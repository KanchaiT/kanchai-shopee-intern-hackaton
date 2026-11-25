#!/usr/bin/env python3
"""Thai/English meeting audio -> transcript.txt + summary.md.

    python openthai.py recording.mp3

1. ffmpeg converts the audio to 16 kHz mono wav (locally).
2. Speech-to-text, one of two backends:
     iapp  (default): iApp's hosted OpenThai ASR Realtime WebSocket API.
                      THE AUDIO IS SENT TO iApp (api.iapp.co.th). Key in IAPP_API_KEY.
     local:           your own OpenThai 2.0 server on 127.0.0.1 (audio stays on this machine).
3. Only the text transcript is sent to Shopee CompassLLM, which writes a Thai summary
   (key decisions, action items, open questions). Long transcripts are summarised in
   chunks, then the chunk summaries are merged.

Requires: Python 3.11+, ffmpeg, `pip install requests google-genai websockets`, and
COMPASS_API_KEY (+ IAPP_API_KEY for the iapp backend), in the environment or in a .env file
next to this script. See README.md.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import queue
import shutil
import signal
import ssl
import subprocess
import sys
import tempfile
import threading
import time
import wave
from pathlib import Path

import requests


def _load_env(path: Path = Path(__file__).resolve().parent / ".env") -> None:
    """Minimal .env loader (KEY=VALUE lines). Real environment variables win; blanks are skipped."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        v = v.strip().strip("'\"")
        if v:
            os.environ.setdefault(k.removeprefix("export ").strip(), v)


_load_env()

DEFAULT_ASR_URL = "http://127.0.0.1:18081"
MEETINGS_DIR = Path(__file__).resolve().parent / "meetings"
IAPP_URL = "wss://api.iapp.co.th/v3/store/speech/speech-to-text/realtime-v2"
IAPP_CHUNK_SECONDS = 0.1  # the service wants 100-200 ms binary messages
IAPP_PACE = 1.0  # 1.0 = send at real-time speed, which the service requires
IAPP_BUSY_RETRIES = 5
COMPASS_BASE_URL = os.environ.get("COMPASS_BASE_URL", "").rstrip("/")  # e.g. https://<compass-host>/compass-api/v1
DEFAULT_MODEL = "gemini-3.8-flash"
DEFAULT_CHUNK_CHARS = 30000
MAX_MERGE_ROUNDS = 5


class MeetingError(Exception):
    """A problem the user can fix; printed without a traceback."""


# ------------------------------------------------------------------ audio -> wav


def convert_to_wav(src: Path, dst: Path) -> None:
    if shutil.which("ffmpeg") is None:
        raise MeetingError("ffmpeg not found. Install it (macOS: `brew install ffmpeg`).")
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
           "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(dst)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise MeetingError(f"ffmpeg could not read {src.name}:\n{p.stderr.strip()[-500:]}")


# --------------------------------------------------------------------------- ASR


def _fmt_time(sec: float) -> str:
    sec = int(sec)
    return f"{sec // 3600:02d}:{sec % 3600 // 60:02d}:{sec % 60:02d}"


def _num(seg: dict, *keys: str) -> float | None:
    for k in keys:
        v = seg.get(k)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return float(v)
    return None


def parse_asr_response(data: dict) -> list[str]:
    """Turn the server's JSON into transcript lines.

    The model card documents a `segments` list and a `display_text` key but not the exact
    schema, so this accepts the plausible variants and fails loudly if none match.
    """
    lines: list[str] = []
    segments = data.get("segments")
    if isinstance(segments, list):
        for seg in segments:
            if isinstance(seg, str):
                text, start = seg, None
            elif isinstance(seg, dict):
                text = seg.get("display_text") or seg.get("text") or ""
                start = _num(seg, "start", "start_time", "start_s", "begin")
            else:
                continue
            text = str(text).strip()
            if text:
                lines.append(f"[{_fmt_time(start)}] {text}" if start is not None else text)
    if not lines:
        text = str(data.get("display_text") or data.get("text") or "").strip()
        if text:
            lines.append(text)
    if not lines:
        raise MeetingError(
            "The ASR server returned no text. Response keys: "
            f"{sorted(data)[:10]}. Was there speech in the recording?")
    return lines


def transcribe_local(wav: Path, asr_url: str, lang: str, mode: str) -> list[str]:
    url = asr_url.rstrip("/") + "/v1/transcribe"
    try:
        with open(wav, "rb") as f:
            r = requests.post(url, files={"file": (wav.name, f, "audio/wav")},
                              data={"lang": lang, "mode": mode}, timeout=(10, 3 * 3600))
    except requests.ConnectionError:
        raise MeetingError(
            f"Cannot reach the ASR server at {asr_url}. Start it first (see README.md), "
            "or pass --asr-url.") from None
    except requests.Timeout:
        raise MeetingError("The ASR server took too long to respond.") from None
    if r.status_code != 200:
        raise MeetingError(f"ASR server returned HTTP {r.status_code}: {r.text[:300]}")
    try:
        return parse_asr_response(r.json())
    except ValueError:
        raise MeetingError(f"ASR server did not return JSON: {r.text[:300]}") from None


# ------------------------------------------------------- iApp realtime (hosted)


class _Busy(Exception):
    """iApp has no free live session right now."""


def _pcm_chunks(pcm: bytes):
    step = int(16000 * 2 * IAPP_CHUNK_SECONDS)
    for i in range(0, len(pcm), step):
        yield pcm[i:i + step]


class Mic:
    """Microphone as an iterator of 100 ms 16 kHz mono int16 chunks. Nothing is saved to disk."""

    def __init__(self):
        self.q: queue.Queue[bytes] = queue.Queue()
        self.stop = threading.Event()

    def __enter__(self):
        try:
            import sounddevice as sd
        except ImportError:
            raise MeetingError("Missing dependency for --live: pip install sounddevice") from None
        try:
            self.stream = sd.RawInputStream(
                samplerate=16000, channels=1, dtype="int16", blocksize=int(16000 * IAPP_CHUNK_SECONDS),
                callback=lambda indata, frames, t, status: self.q.put(bytes(indata)))
            self.stream.start()
        except Exception as e:  # no device, or macOS microphone permission denied
            raise MeetingError(
                f"Cannot open the microphone: {e}. On macOS, allow microphone access for your "
                "terminal in System Settings > Privacy & Security > Microphone.") from None
        return self

    def __exit__(self, *exc):
        self.stream.stop()
        self.stream.close()

    def chunks(self):
        while not self.stop.is_set():
            try:
                yield self.q.get(timeout=0.2)
            except queue.Empty:
                pass
        while not self.q.empty():  # audio captured just before the stop
            yield self.q.get_nowait()


def _iapp_stream(chunks, key: str, url: str, latency_ms: int, pace: float = 0.0,
                 total_s: float = 0.0, on_line=None) -> list[str]:
    """Stream `chunks` to iApp. pace=1.0 sends at real-time speed (needed for files);
    pace=0 sends as they come (a live mic is already real time)."""
    from websockets.exceptions import ConnectionClosed, InvalidStatus
    from websockets.sync.client import connect

    kwargs = {}
    if url.startswith("wss://"):
        # python.org builds of Python on macOS ship without CA certificates; use certifi's
        # bundle (already installed as a dependency of requests) instead of the system default.
        try:
            import certifi
            kwargs["ssl"] = ssl.create_default_context(cafile=certifi.where())
        except ImportError:
            pass
    try:
        ws = connect(url, additional_headers={"apikey": key}, open_timeout=15, max_size=None, **kwargs)
    except InvalidStatus as e:
        code = e.response.status_code
        if code == 401:
            raise MeetingError("iApp rejected the API key (HTTP 401). Check IAPP_API_KEY.") from None
        if code == 402:
            raise MeetingError("iApp account has too few credits (HTTP 402).") from None
        raise MeetingError(f"iApp refused the connection (HTTP {code}).") from None
    except ssl.SSLCertVerificationError as e:
        raise MeetingError(
            f"TLS certificate check failed ({e.verify_message}). If this Python has no CA bundle, "
            "run: pip install certifi. On a company network that inspects TLS, point SSL_CERT_FILE "
            "at your company's CA bundle.") from None
    except (OSError, TimeoutError) as e:
        raise MeetingError(f"Cannot reach iApp at {url}: {e}") from None

    lines: list[str] = []
    stop = threading.Event()
    send_err: list[Exception] = []
    done = False

    def sender():
        try:
            t0 = time.monotonic()
            for n, chunk in enumerate(chunks):
                if stop.is_set():
                    return
                ws.send(chunk)
                if total_s and n and n % 600 == 0:  # every minute of audio
                    print(f"  {_fmt_time(n * IAPP_CHUNK_SECONDS)} / {_fmt_time(total_s)}", flush=True)
                if pace:  # absolute schedule (not sleep-per-chunk) so pacing does not drift
                    wait = t0 + (n + 1) * IAPP_CHUNK_SECONDS * pace - time.monotonic()
                    if wait > 0:
                        time.sleep(wait)
            ws.send("end")
        except ConnectionClosed:
            pass  # the receiver reports why
        except Exception as e:
            send_err.append(e)
            ws.close()  # unblock the receiver

    try:
        ws.send(json.dumps({"language_code": "th", "sample_rate": 16000, "encoding": "pcm_s16le",
                            "latency_ms": latency_ms}))
        first = json.loads(ws.recv(timeout=30))
        _raise_iapp_error(first)
        if first.get("type") != "ready":
            raise MeetingError(f"Unexpected first message from iApp: {str(first)[:200]}")
        print("iApp connected; transcribing.", flush=True)
        threading.Thread(target=sender, daemon=True).start()
        for raw in ws:
            if isinstance(raw, bytes):
                continue
            m = json.loads(raw)
            _raise_iapp_error(m)
            if m.get("type") == "final":
                text = str(m.get("text", "")).strip()
                if text:
                    start = m.get("start")
                    line = f"[{_fmt_time(start)}] {text}" if isinstance(start, (int, float)) else text
                    lines.append(line)
                    if on_line:
                        on_line(line)
            elif m.get("type") == "done":
                done = True
                break
    except ConnectionClosed as e:
        raise MeetingError(f"iApp closed the connection early ({e}).") from None
    except TimeoutError:
        raise MeetingError("iApp did not answer in time.") from None
    finally:
        stop.set()
        ws.close()
    if send_err:
        raise MeetingError(f"Failed while sending audio: {send_err[0]}")
    if not done:
        raise MeetingError("iApp connection ended before the transcript was complete.")
    if not lines:
        raise MeetingError("iApp returned no text. Was there speech?")
    return lines


def _raise_iapp_error(m: dict) -> None:
    if m.get("type") != "error":
        return
    code, msg = m.get("code"), m.get("message")
    if code == "BUSY":
        raise _Busy
    hints = {"TOO_LONG": "iApp limits a session to 4 hours of audio; split the recording.",
             "TOO_FAST": "Audio was sent faster than real time.",
             "TIMEOUT": "No audio or settings arrived in time."}
    raise MeetingError(f"iApp error {code}: {msg}. {hints.get(code, '')}".strip())


def _with_busy_retry(call):
    try:
        import websockets  # noqa: F401
        from websockets.sync.client import connect  # noqa: F401
    except ImportError:
        raise MeetingError("Missing dependency: pip install 'websockets>=12'") from None
    for attempt in range(1, IAPP_BUSY_RETRIES + 1):
        try:
            return call()
        except _Busy:
            if attempt == IAPP_BUSY_RETRIES:
                raise MeetingError("iApp has no free live session (BUSY). Try again later.") from None
            print(f"  iApp is busy; retrying in {5 * attempt}s", flush=True)
            time.sleep(5 * attempt)
    raise AssertionError  # unreachable


def transcribe_iapp(wav: Path, key: str, url: str = IAPP_URL, latency_ms: int = 560) -> list[str]:
    with wave.open(str(wav), "rb") as w:
        if (w.getframerate(), w.getnchannels(), w.getsampwidth()) != (16000, 1, 2):
            raise MeetingError("Internal error: audio is not 16 kHz mono 16-bit after conversion.")
        pcm = w.readframes(w.getnframes())
    if not pcm:
        raise MeetingError("The recording contains no audio.")
    return _with_busy_retry(lambda: _iapp_stream(
        _pcm_chunks(pcm), key, url, latency_ms, pace=IAPP_PACE, total_s=len(pcm) / 32000))


def transcribe_iapp_live(mic: Mic, key: str, url: str, latency_ms: int, on_line) -> list[str]:
    return _with_busy_retry(lambda: _iapp_stream(mic.chunks(), key, url, latency_ms, on_line=on_line))


# ------------------------------------------------------------------ summarising

SUMMARY_SECTIONS = """\
## การตัดสินใจสำคัญ (Key decisions)
## งานที่ต้องทำ (Action items)  - ระบุผู้รับผิดชอบและกำหนดส่งหากมีการพูดถึง
## คำถามที่ยังค้างอยู่ (Open questions)"""

COMMON_RULES = """\
The transcript comes from automatic speech recognition of a Thai/English mixed meeting, so it
may contain ASR errors (misheard words, wrong spellings of names and technical terms, missing
words). Infer the correct word from context where you can. Never invent facts, owners or
deadlines that the transcript does not support; if something is unclear, say so.
English words may appear transliterated in Thai script (e.g. "ดีพลอย" for "deploy"); restore
the English term where you can tell what it is.
Write in Thai, but keep English technical terms, product names and code identifiers as-is.
Output Markdown only."""


def chunk_text(lines: list[str], max_chars: int) -> list[str]:
    chunks, cur, size = [], [], 0
    for line in lines:
        # A single line longer than the limit is split so no chunk exceeds it.
        pieces = [line[i:i + max_chars] for i in range(0, len(line), max_chars)] or [line]
        for piece in pieces:
            if cur and size + len(piece) + 1 > max_chars:
                chunks.append("\n".join(cur))
                cur, size = [], 0
            cur.append(piece)
            size += len(piece) + 1
    if cur:
        chunks.append("\n".join(cur))
    return chunks


def make_client():
    key = os.environ.get("COMPASS_API_KEY")
    if not key:
        raise MeetingError("COMPASS_API_KEY is not set. Add COMPASS_API_KEY=... to the .env file next to openthai.py "
            "(or export it). Use --skip-summary to get only the transcript.")
    if not COMPASS_BASE_URL:
        raise MeetingError("COMPASS_BASE_URL is not set. Add COMPASS_BASE_URL=https://<compass-host>/compass-api/v1 "
                           "to .env (see .env.example). Use --skip-summary to get only the transcript.")
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        raise MeetingError("Missing dependency: pip install google-genai") from None
    return genai.Client(
        api_key=key,
        http_options=types.HttpOptions(api_version="v1", base_url=COMPASS_BASE_URL))


def generate(client, model: str, prompt: str, retries: int = 4) -> str:
    """One Compass call, retrying rate limits and transient server errors."""
    for attempt in range(1, retries + 1):
        try:
            text = (client.models.generate_content(model=model, contents=prompt).text or "").strip()
            if text:
                return text
            err: Exception = MeetingError("Compass returned an empty response.")
        except Exception as e:  # SDK raises its own error types; classify by status code
            code = getattr(e, "code", None)
            if code in (400, 401, 403, 404):
                raise MeetingError(f"Compass rejected the request (HTTP {code}): {e}") from None
            err = e
        if attempt == retries:
            raise MeetingError(f"Compass failed after {retries} attempts: {err}") from None
        time.sleep(2 ** attempt)
    raise AssertionError  # unreachable


def summarize(lines: list[str], client, model: str, chunk_chars: int) -> str:
    chunks = chunk_text(lines, chunk_chars)
    if len(chunks) == 1:
        return generate(client, model,
                        f"{COMMON_RULES}\n\nSummarise this meeting transcript using exactly these "
                        f"sections:\n{SUMMARY_SECTIONS}\n\nTranscript:\n{chunks[0]}")

    print(f"Transcript is long: summarising in {len(chunks)} chunks, then merging.", flush=True)
    partials = []
    for i, c in enumerate(chunks, 1):
        print(f"  chunk {i}/{len(chunks)}", flush=True)
        partials.append(generate(
            client, model,
            f"{COMMON_RULES}\n\nThis is part {i} of {len(chunks)} of a meeting transcript. "
            f"Summarise ONLY this part using these sections:\n{SUMMARY_SECTIONS}\n"
            f"Keep every decision, action item (with owner and deadline if mentioned) and open "
            f"question; later parts will be merged with yours.\n\nTranscript part:\n{c}"))

    for _ in range(MAX_MERGE_ROUNDS):  # merge, recursing if the partials are themselves too long
        groups = chunk_text(["\n\n---\n\n".join(partials)], chunk_chars)
        if len(groups) == 1:
            break
        print(f"  merging {len(partials)} partial summaries in {len(groups)} groups", flush=True)
        partials = [generate(client, model, _merge_prompt(g, "some of the")) for g in groups]
    else:
        raise MeetingError("Partial summaries are still too long to merge; raise --chunk-chars.")
    return generate(client, model, _merge_prompt("\n\n---\n\n".join(partials), "all"))


def _merge_prompt(body: str, which: str) -> str:
    return (f"{COMMON_RULES}\n\nBelow are partial summaries of {which} consecutive parts of one "
            f"meeting. Merge them into one summary using exactly these sections:\n"
            f"{SUMMARY_SECTIONS}\nRemove duplicates, keep every distinct decision, action item "
            f"(with owner if mentioned) and open question, and drop items a later part resolved."
            f"\n\nPartial summaries:\n{body}")


# --------------------------------------------------------------------------- CLI


def new_meeting_dir(base: Path) -> Path:
    """Create <base>/<YYYY-MM-DD_HH-MM-SS>/ for this run (a numeric suffix if it already exists)."""
    label = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base.mkdir(parents=True, exist_ok=True)
    for n in range(1, 100):
        d = base / (label if n == 1 else f"{label}_{n}")
        try:
            d.mkdir()
            return d
        except FileExistsError:
            continue
    raise MeetingError(f"Could not create a meeting folder in {base}.")


def _discard_if_empty(d: Path) -> None:
    """Drop a meeting folder that a failed run left with nothing useful in it."""
    try:
        if all(f.stat().st_size == 0 for f in d.iterdir()):
            for f in d.iterdir():
                f.unlink()
            d.rmdir()
    except OSError:
        pass


def run_live(args: argparse.Namespace) -> None:
    if args.backend != "iapp":
        raise MeetingError("--live works with the iapp backend only (for local live transcription "
                           "use seari.py).")
    key = os.environ.get("IAPP_API_KEY")
    if not key:
        raise MeetingError("IAPP_API_KEY is not set.")
    client = None if args.skip_summary else make_client()
    with Mic() as mic:  # open the mic first so a mic problem leaves no empty folder behind
        out = new_meeting_dir(Path(args.out_dir))
        path = out / "transcript.txt"
        try:
            with open(path, "w", encoding="utf-8") as f:
                def on_line(line: str):  # written as it arrives, so a crash or dropped link loses nothing
                    f.write(line + "\n")
                    f.flush()
                    print(f"  {line}", flush=True)

                def first_ctrl_c(signum, frame):  # first Ctrl+C = finish cleanly; second = hard stop
                    mic.stop.set()
                    signal.signal(signal.SIGINT, signal.default_int_handler)
                    print("\nStopping; finishing the last sentences ... (Ctrl+C again to abort)", flush=True)

                signal.signal(signal.SIGINT, first_ctrl_c)
                print(f"Listening to the microphone; audio is streamed to iApp (api.iapp.co.th) and NOT "
                      f"saved. Saving to {out}\nPress Ctrl+C to finish.", flush=True)
                lines = transcribe_iapp_live(mic, key, args.iapp_url, args.latency_ms, on_line)
        except BaseException:
            _discard_if_empty(out)
            raise
    print(f"Wrote {path} ({len(lines)} sentences)")

    if client is None:
        return
    print(f"Summarising with Compass ({args.model}) ...", flush=True)
    (out / "summary.md").write_text(summarize(lines, client, args.model, args.chunk_chars) + "\n",
                                    encoding="utf-8")
    print(f"Wrote {out / 'summary.md'}")


def run(args: argparse.Namespace) -> None:
    if args.chunk_chars < 2000:
        raise MeetingError("--chunk-chars must be at least 2000.")
    if args.live:
        return run_live(args)
    src = Path(args.audio)
    if not src.is_file():
        raise MeetingError(f"File not found: {src}")
    # Fail on a missing key before spending minutes on transcription.
    client = None if args.skip_summary else make_client()
    iapp_key = None
    if args.backend == "iapp":
        iapp_key = os.environ.get("IAPP_API_KEY")
        if not iapp_key:
            raise MeetingError("IAPP_API_KEY is not set (or use --backend local).")

    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "audio.wav"
        print(f"Converting {src.name} ...", flush=True)
        convert_to_wav(src, wav)
        t0 = time.monotonic()
        if args.backend == "iapp":
            print("Transcribing with iApp realtime. NOTE: the audio is being sent to iApp "
                  "(api.iapp.co.th). It runs at real-time speed, so it takes as long as the "
                  "recording.", flush=True)
            lines = transcribe_iapp(wav, iapp_key, args.iapp_url, args.latency_ms)
        else:
            print(f"Transcribing locally ({args.asr_url}, lang={args.lang}) ...", flush=True)
            lines = transcribe_local(wav, args.asr_url, args.lang, args.mode)
    out = new_meeting_dir(Path(args.out_dir))  # only now, so failed runs leave no empty folder
    (out / "transcript.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out / 'transcript.txt'} ({len(lines)} segments, {time.monotonic() - t0:.0f}s)")

    if client is None:
        return
    print(f"Summarising with Compass ({args.model}) ...", flush=True)
    summary = summarize(lines, client, args.model, args.chunk_chars)
    (out / "summary.md").write_text(summary + "\n", encoding="utf-8")
    print(f"Wrote {out / 'summary.md'}")


def main() -> int:
    p = argparse.ArgumentParser(description="Thai/English meeting audio -> transcript.txt + summary.md")
    p.add_argument("audio", nargs="?", help="recording (mp3, wav, m4a, ...); omit with --live")
    p.add_argument("--live", action="store_true",
                   help="listen to the microphone instead of reading a file; nothing is recorded to "
                        "disk. Ctrl+C finishes and writes transcript.txt and summary.md")
    p.add_argument("-o", "--out-dir", default=str(MEETINGS_DIR),
                   help="parent folder (default: ./meetings next to this script). Each run creates a "
                        "<YYYY-MM-DD_HH-MM-SS>/ folder inside it holding transcript.txt and summary.md")
    p.add_argument("--backend", default=os.environ.get("ASR_BACKEND", "iapp"), choices=["iapp", "local"],
                   help="iapp = hosted OpenThai realtime (audio leaves this machine); "
                        "local = your own server (env ASR_BACKEND; default iapp)")
    p.add_argument("--iapp-url", default=IAPP_URL)
    p.add_argument("--latency-ms", type=int, default=560, choices=[80, 320, 560, 1120],
                   help="iapp only: lower = faster but slightly less accurate")
    p.add_argument("--asr-url", default=os.environ.get("ASR_URL", DEFAULT_ASR_URL),
                   help=f"local backend only: server URL (env ASR_URL; default {DEFAULT_ASR_URL})")
    p.add_argument("--lang", default="th-mix", choices=["th-mix", "th-TH"], help="local backend only")
    p.add_argument("--mode", default="standard", choices=["standard", "conversation"],
                   help="local backend only. Not 'meeting': that mode skips the pause cutting")
    p.add_argument("--model", default=os.environ.get("COMPASS_MODEL", DEFAULT_MODEL))
    p.add_argument("--chunk-chars", type=int, default=DEFAULT_CHUNK_CHARS,
                   help="max transcript characters per Compass call")
    p.add_argument("--skip-summary", action="store_true", help="only write transcript.txt")
    args = p.parse_args()
    if args.live == bool(args.audio):
        p.error("give either an audio file or --live (not both, not neither)")
    try:
        run(args)
    except MeetingError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
