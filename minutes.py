"""Minute: a voice assistant that waits quietly until you call it.

  * WAITING: a local Whisper model (mlx-whisper, on the Mac's GPU) listens only for the
    phrases below. Nothing is sent to the cloud, saved, or summarised.
  * Say "Hello Minute ..." -> a blink-blink alert plays and a Gemini Live voice conversation
    starts. If you ask in the same breath ("Hello Minute, what time is it?") the question is
    sent as text the moment the session opens, so the answer starts sooner.
  * Say "Stop Minute" -> the same alert plays, the conversation ends, and it goes back to
    WAITING for the next "Hello Minute".
  * Say "Finished meeting" (or press Ctrl+C) -> the program exits.

Setup:
    pip install websockets sounddevice numpy mlx-whisper
    # optional, for Calendar:
    pip install google-api-python-client google-auth-oauthlib

    .env: COMPASS_API_KEY (Gemini Live runs through CompassLLM), SEARI_SSH_HOSTS (optional),
          COMPASS_BASE_URL (https://<compass-host>/compass-api/v1), COMPASS_LIVE_MODEL (default gemini-3.8-live),
          SEARI_STT_MODEL (default mlx-community/whisper-large-v3-turbo),
          SEARI_STT_LANGUAGE (default "en": the local recogniser only decodes English)
    Calendar (today / create / cancel) lives in gg_calendar_function.py. One-time Google
    sign-in:  python gg_calendar_function.py auth   (see that file for the credential paths)

    python minutes.py           # add --debug to print what the local recogniser hears

Use headphones, otherwise the mic hears the speaker and the alert.
"""

import asyncio
import base64
import collections
import concurrent.futures
import contextlib
import datetime as dt
import json
import os
import re
import ssl
import subprocess
import sys
import threading
import time
import zlib
from urllib.parse import urlencode

import numpy as np
import sounddevice as sd
from websockets.asyncio.client import connect as ws_connect
from websockets.exceptions import ConnectionClosedOK, InvalidStatus

import gg_calendar_function as gcal


def _load_env(path: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")):
    """Minimal .env loader: KEY=VALUE lines; real environment variables win; blanks are skipped."""
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            v = v.strip().strip("'\"")
            if v:
                os.environ.setdefault(k.strip(), v)


_load_env()

COMPASS_BASE_URL = os.environ.get("COMPASS_BASE_URL", "").rstrip("/")  # e.g. https://<compass-host>/compass-api/v1
COMPASS_REALTIME_URL = os.environ.get("COMPASS_REALTIME_URL") or (
    re.sub(r"^https?://", "wss://", COMPASS_BASE_URL) + "/realtime" if COMPASS_BASE_URL else "")
MODEL = os.environ.get("COMPASS_LIVE_MODEL", "gemini-3.8-live")
STT_MODEL = os.environ.get("SEARI_STT_MODEL", "mlx-community/whisper-large-v3-turbo")
STT_LANGUAGE = os.environ.get("SEARI_STT_LANGUAGE") or "en"  # English only: no per-utterance language guessing
SSH_HOSTS = {h.strip() for h in os.environ.get("SEARI_SSH_HOSTS", "").split(",") if h.strip()}
HERE = os.path.dirname(os.path.abspath(__file__))
CMD_TIMEOUT = 60
MAX_OUTPUT = 4000

SEND_RATE = 16000  # mic -> Whisper / Gemini
RECV_RATE = 24000  # Gemini -> speaker
FRAME = 512  # 32 ms of mic audio per callback

# Voice-activity segmentation for local transcription.
PREROLL = 0.3  # seconds of audio kept before speech starts, so the first syllable isn't cut
SILENCE_END = 0.6  # seconds of quiet that end an utterance (lower = faster wake, choppier text)
MAX_SEGMENT = 15.0  # force-cut long monologues so transcripts keep flowing
MIN_SPEECH = 0.3  # ignore blips shorter than this

SAID_GAP = 1.2  # seconds without new words before what you said is printed as one line
TALK_IDLE = 60.0  # seconds of silence before a forgotten conversation ends itself (with the alert)
BEEP_SECONDS = 0.28
# Echo guard: while Gemini's voice plays (plus this tail for room reverb and output buffering)
# the mic is NOT sent to Gemini, otherwise it hears itself through the speaker and answers itself.
ECHO_TAIL = 0.7
# Local 'Stop Minute' detection while Gemini is talking: text this long is almost certainly
# Gemini's own voice coming back through the speaker, not you saying a short command.
BARGE_IN_MAX_CHARS = 40
STT_PROMPT = "Hello Minute. Hey Minute. Stop Minute. Finished meeting."  # biases Whisper toward the command words

# --- Command phrases. Whisper may spell or mishear them, so each accepts several variants. -------
# The name as Whisper might write it: minute(s), minit, minnit, minuet, mini, minnie, "min it", "mi-nit".
_NAME = r"(?:min+(?:ute|ut|uet|it|ite|ie|y|i)s?|mi[\s-]?nit(?:e|s)?|min[\s-]it)\b"
_SEP = r"[\s,.!?-]*"

# Wake: a greeting + the name:  "hey minute", "hi minutes", "hello, minit", "hey there minnie", "yo minute".
# ("okay minutes" and "a minute" are deliberately NOT accepted: they are everyday meeting talk.)
WAKE_RE = re.compile(
    r"\b(?:hey+|hay+|hei|heya|hi+|hiya|hello|hullo|yo)" + _SEP + r"(?:there" + _SEP + r")?" + _NAME, re.I)

# Stop (ends the chat): "stop minute", "end minute", "bye minute", "thank you minute", "that's all
# minute", "minute stop", "minute, please stop".
_STOP_WORDS = r"(?:stop+|stahp|end|quit|exit|bye|good\s?bye|enough|thanks?|thank\s+you|that'?s\s+all)"
_STOP_AFTER = r"(?:stop+|stahp|bye|good\s?bye|enough|that'?s\s+all)"
STOP_RE = re.compile(
    r"\b" + _STOP_WORDS + _SEP + r"(?:the\s+)?" + _NAME
    + r"|\b" + _NAME + _SEP + r"(?:please" + _SEP + r")?" + _STOP_AFTER + r"\b", re.I)

# Finish (quits the whole program): "finished meeting", "finish the meeting", "end meeting",
# "close the meeting", "meeting finished", "meeting is over", "meeting adjourned".
# This one is only honoured for a SHORT utterance (FINISH_MAX_CHARS), so a long sentence that merely
# mentions it ("we finished the meeting yesterday and ...") can never quit the program.
FINISH_RE = re.compile(
    r"\b(?:finish(?:ed)?|end(?:ed)?|close[sd]?|adjourn(?:ed)?)" + _SEP + r"(?:with\s+)?(?:the\s+|this\s+|our\s+)?meeting\b"
    r"(?!\s*(?:notes?|minutes?|room|invites?|link|agenda|summary|recording|series|time))"
    r"|\bmeeting\s*(?:finish(?:ed)?|ended|adjourned)\b"
    r"|\bmeeting(?:\s+is|'s)\s+(?:finish(?:ed)?|ended|adjourned|over|done)\b", re.I)
FINISH_MAX_CHARS = 40


def is_finish(text: str) -> bool:
    return bool(FINISH_RE.search(text)) and len(text.strip()) <= FINISH_MAX_CHARS


# Whisper's classic hallucinations on near-silence.
HALLUCINATIONS = {
    "thank you.", "thank you", "thanks for watching!", "thanks for watching.", "you", ".",
}

SYSTEM_PROMPT = (
    "You are Minute, a voice assistant the user calls up during a meeting. Speak your answers "
    "aloud, in one to three short sentences. The user speaks English and you speak English only: "
    "always reply in English, even if you are addressed in another language, and if you cannot "
    "understand them, ask them in English to repeat. "
    "Speech recognition can mishear. Use the tools only when the user asks you to act on the "
    "computer, and read back any command that changes things before running it. For the "
    "calendar: use calendar_today to answer what meetings the user has (summarise them briefly, "
    "by time); to create or cancel an appointment, say back the title, date and time and any "
    "guests first, then call the tool. The user confirms every change in the terminal. If "
    "calendar_cancel_event says several meetings match, ask which one. Never say the phrase "
    "'Stop Minute' yourself."
)

# ---------------------------------------------------------------- confirmation

_pending_confirm: asyncio.Future | None = None


CONFIRM_TIMEOUT = 90.0  # seconds to wait for your y/N before the action is cancelled


async def confirm(prompt: str, timeout: float = CONFIRM_TIMEOUT) -> bool:
    """Ask in the terminal. Type y and press Enter to approve; anything else, or no answer, cancels."""
    global _pending_confirm
    loop = asyncio.get_running_loop()
    _pending_confirm = loop.create_future()
    print(f"\n[CONFIRM] {prompt}\n  Type y and press Enter to approve, anything else cancels "
          f"(auto-cancels in {timeout:.0f}s) [y/N] ", end="", flush=True)
    try:
        answer = await asyncio.wait_for(_pending_confirm, timeout)
    except TimeoutError:
        print(f"\n  (no answer in {timeout:.0f}s: cancelled)", flush=True)
        return False
    finally:
        _pending_confirm = None
    approved = answer.strip().lower() in ("y", "yes")
    print(f"  -> you answered {answer.strip()!r}: {'approved, going ahead' if approved else 'cancelled'}", flush=True)
    return approved


_on_enter = None  # set by Agent: pressing Enter (when no y/N prompt is open) ends the conversation


def _on_stdin_line(line: str):
    if _pending_confirm is not None and not _pending_confirm.done():
        _pending_confirm.set_result(line)
    elif _on_enter is not None and not line.strip():  # only a bare Enter ends the chat, not a stray 'y'
        _on_enter()


def _stdin_thread(loop: asyncio.AbstractEventLoop):
    # Daemon thread, so a blocked readline never delays exit.
    for line in sys.stdin:
        loop.call_soon_threadsafe(_on_stdin_line, line)


# ----------------------------------------------------------------------- tools


def _run(args: list[str] | str, shell: bool) -> dict:
    try:
        p = subprocess.run(args, shell=shell, capture_output=True, text=True, timeout=CMD_TIMEOUT)
        return {
            "exit_code": p.returncode,
            "stdout": p.stdout[-MAX_OUTPUT:],
            "stderr": p.stderr[-MAX_OUTPUT:],
        }
    except subprocess.TimeoutExpired:
        return {"error": f"timed out after {CMD_TIMEOUT}s"}


async def run_shell(command: str) -> dict:
    if not await confirm(f"Run locally: {command}"):
        return {"error": "user declined"}
    return await asyncio.to_thread(_run, command, True)


async def ssh_exec(host: str, command: str) -> dict:
    if host not in SSH_HOSTS:
        return {"error": f"host '{host}' not in SEARI_SSH_HOSTS allowlist"}
    if not await confirm(f"Run on {host}: {command}"):
        return {"error": "user declined"}
    # BatchMode: never prompt for a password mid-conversation.
    return await asyncio.to_thread(
        _run, ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host, command], False
    )


async def read_file(path: str) -> dict:
    try:
        with open(os.path.expanduser(path), errors="replace") as f:
            return {"content": f.read(MAX_OUTPUT)}
    except OSError as e:
        return {"error": str(e)}


async def write_file(path: str, content: str) -> dict:
    path = os.path.expanduser(path)
    if not await confirm(f"Write {len(content)} chars to {path}"):
        return {"error": "user declined"}
    try:
        with open(path, "w") as f:
            f.write(content)
        return {"ok": True}
    except OSError as e:
        return {"error": str(e)}


async def calendar_today(date: str = "today") -> dict:
    """Read-only: the meetings on one day from the user's Google Calendar."""
    return await asyncio.to_thread(gcal.today, date or "today")


async def calendar_create_event(summary: str, start_iso: str, end_iso: str, description: str = "",
                                guests: str = "", room: str = "") -> dict:
    """start_iso / end_iso are local-time ISO 8601, e.g. 2026-11-24T15:00:00 (Bangkok unless an offset is given)."""
    extra = (f" with {guests}" if guests else "") + (f" in room {room}" if room else "")
    note = " (invitations will be emailed)" if guests or room else ""
    if not await confirm(f"Create '{summary}' {start_iso} -> {end_iso}{extra}{note}"):
        return {"error": "user declined"}
    return await asyncio.to_thread(gcal.create, summary, start_iso, end_iso, guests or "", room or "",
                                   "", description or "")


async def calendar_cancel_event(title: str = "", date: str = "today", event_id: str = "") -> dict:
    """Find the one appointment first, show it to the user, and delete only after they confirm."""
    found = await asyncio.to_thread(gcal.find_to_cancel, event_id or "", title or "", date or "today")
    if "error" in found:
        return found
    notify = f" ({found['guests']} guest(s) will be notified)" if found.get("guests") else ""
    if not await confirm(f"Cancel '{found['title']}' {found['start']} -> {found['end']}{notify}"):
        return {"error": "user declined"}
    return await asyncio.to_thread(gcal.cancel, found["event_id"])


TOOLS = {
    "run_shell": (run_shell, "Run a shell command on this Mac (zsh). Asks the user to confirm.",
                  {"command": ("string", "The command line")}),
    "ssh_exec": (ssh_exec, "Run a command on an allowlisted remote host over SSH.",
                 {"host": ("string", "Host alias"), "command": ("string", "Command to run")}),
    "read_file": (read_file, "Read a text file from this Mac.", {"path": ("string", "File path")}),
    "write_file": (write_file, "Write a text file on this Mac. Asks the user to confirm.",
                   {"path": ("string", "File path"), "content": ("string", "Full file content")}),
    "calendar_today": (calendar_today, "List the user's meetings on one day from Google Calendar "
                       "(default today). Use this for 'what meetings do I have today/tomorrow'.",
                       {"date": ("string", "today, tomorrow, or YYYY-MM-DD (default today)")}),
    "calendar_create_event": (calendar_create_event,
                              "Create a Google Calendar appointment, optionally with guests and/or a "
                              "meeting room from the room list. Asks the user to confirm.",
                              {"summary": ("string", "Title"),
                               "start_iso": ("string", "Local start, ISO 8601, e.g. 2026-11-26T15:00:00"),
                               "end_iso": ("string", "Local end, ISO 8601"),
                               "description": ("string", "Optional notes"),
                               "guests": ("string", "Optional guest emails, comma-separated"),
                               "room": ("string", "Optional meeting room name")}),
    "calendar_cancel_event": (calendar_cancel_event,
                              "Cancel an appointment the user organised. Give its title and date, or "
                              "the event_id from calendar_today. Asks the user to confirm.",
                              {"title": ("string", "Part of the meeting title"),
                               "date": ("string", "today, tomorrow, or YYYY-MM-DD (default today)"),
                               "event_id": ("string", "Event id, if known")}),
}
OPTIONAL_PARAMS = {"date", "description", "guests", "room", "title", "event_id"}


def _declarations() -> list[dict]:
    decls = []
    for name, (_, desc, params) in TOOLS.items():
        decls.append({
            "name": name,
            "description": desc,
            "parameters": {
                "type": "OBJECT",
                "properties": {p: {"type": t.upper(), "description": d} for p, (t, d) in params.items()},
                "required": [p for p in params if p not in OPTIONAL_PARAMS],
            },
        })
    for d in decls:  # the room list lives in rooms.csv, so tell the model which rooms exist right now
        if d["name"] == "calendar_create_event":
            names = ", ".join(gcal.room_names()) or "none configured"
            d["parameters"]["properties"]["room"]["description"] = f"Optional meeting room. One of: {names}."
    return decls


async def dispatch(fc: dict) -> dict:
    """Run one tool call from the model ({"id", "name", "args"}); returns the functionResponse."""
    name, args = fc.get("name", ""), fc.get("args") or {}
    print(f"\n[tool] {name}({json.dumps(args)})")
    fn = TOOLS.get(name, (None,))[0]
    if fn is None:
        result = {"error": f"unknown tool {name}"}
    else:
        try:
            result = await fn(**args)
        except Exception as e:
            result = {"error": f"{type(e).__name__}: {e}"}
    print(f"[tool result] {json.dumps(result)[:300]}")
    return {"id": fc.get("id"), "name": name, "response": result}


# ------------------------------------------------ Gemini Live through CompassLLM (WebSocket)


def _ssl_context() -> ssl.SSLContext:
    """python.org Python on macOS ships without CA certificates, so use certifi's bundle
    (or the file named by SSL_CERT_FILE, e.g. a company CA bundle)."""
    cafile = os.environ.get("SSL_CERT_FILE")
    if not cafile:
        try:
            import certifi
            cafile = certifi.where()
        except ImportError:
            pass
    return ssl.create_default_context(cafile=cafile)


class LiveSession:
    """One Gemini Live session over CompassLLM's realtime WebSocket (raw JSON messages)."""

    def __init__(self, ws):
        self.ws = ws

    async def recv(self) -> dict:
        msg = json.loads(await self.ws.recv())
        if msg.get("type") == "error" or "error" in msg:
            raise RuntimeError(f"Compass error: {json.dumps(msg)[:300]}")
        return msg

    async def stream(self):
        """Every server message, until the connection is closed normally."""
        try:
            while True:
                yield await self.recv()
        except ConnectionClosedOK:
            return

    async def _send(self, obj: dict) -> None:
        await self.ws.send(json.dumps(obj))

    async def send_text(self, text: str) -> None:
        await self._send({"clientContent": {"turns": [{"role": "user", "parts": [{"text": text}]}],
                                            "turnComplete": True}})

    async def send_audio(self, pcm16: bytes) -> None:
        await self._send({"realtimeInput": {"audio": {"data": base64.b64encode(pcm16).decode(),
                                                      "mimeType": f"audio/pcm;rate={SEND_RATE}"}}})

    async def send_tool_responses(self, responses: list[dict]) -> None:
        await self._send({"toolResponse": {"functionResponses": responses}})


@contextlib.asynccontextmanager
async def compass_live(setup: dict):
    """Open a Gemini Live session via CompassLLM and wait for setupComplete."""
    key = os.environ.get("COMPASS_API_KEY")
    if not key:
        raise RuntimeError("COMPASS_API_KEY is not set (put it in .env).")
    if not COMPASS_REALTIME_URL:
        raise RuntimeError("COMPASS_BASE_URL is not set (put it in .env; see .env.example).")
    url = f"{COMPASS_REALTIME_URL}?{urlencode({'model': MODEL})}"
    kwargs = {"ssl": _ssl_context()} if url.startswith("wss://") else {}
    try:
        ws = await ws_connect(url, additional_headers={"Authorization": f"Bearer {key}"},
                              open_timeout=15, max_size=None, **kwargs)
    except InvalidStatus as e:
        code = e.response.status_code
        hint = {401: "check COMPASS_API_KEY", 403: "this key may not have access to the model",
                404: f"model '{MODEL}' or the URL was not found"}.get(code)
        raise RuntimeError(f"Compass refused the connection (HTTP {code})" + (f": {hint}" if hint else "")) from None
    except ssl.SSLCertVerificationError as e:
        raise RuntimeError(f"TLS certificate check failed ({e.verify_message}). On a network that inspects TLS, "
                           "point SSL_CERT_FILE at your company's CA bundle.") from None
    except (OSError, TimeoutError) as e:
        raise RuntimeError(f"Cannot reach Compass at {COMPASS_REALTIME_URL}: {e}") from None
    try:
        session = LiveSession(ws)
        await session._send({"setup": setup})
        try:
            while "setupComplete" not in await asyncio.wait_for(session.recv(), 30):
                pass
        except TimeoutError:
            raise RuntimeError("Compass did not finish session setup within 30 s.") from None
        yield session
    finally:
        await ws.close()


# ------------------------------------------------------------ local speech-to-text


class Segmenter:
    """Energy-based voice activity detection: cuts the mic stream into utterances."""

    def __init__(self):
        self.noise: float | None = None
        self.pre: collections.deque = collections.deque(maxlen=max(1, int(PREROLL * SEND_RATE / FRAME)))
        self.reset()

    def reset(self):
        self.pre.clear()
        self.buf: list[np.ndarray] = []
        self.speech = 0
        self.silence = 0

    def feed(self, frame: np.ndarray) -> np.ndarray | None:
        """Returns a finished utterance (int16 array) or None."""
        rms = float(np.sqrt(np.mean((frame.astype(np.float32) / 32768.0) ** 2))) + 1e-9
        if self.noise is None:
            self.noise = rms
        is_speech = rms > max(self.noise * 3.0, 0.004)
        # Track the room's noise floor: fast on quiet frames, very slowly during "speech"
        # so a fan switching on doesn't look like someone talking forever.
        self.noise = (0.95 * self.noise + 0.05 * rms) if not is_speech else (0.999 * self.noise + 0.001 * rms)

        if not self.buf:
            if is_speech:
                self.buf = [*self.pre, frame]
                self.speech, self.silence = 1, 0
            else:
                self.pre.append(frame)
            return None

        self.buf.append(frame)
        if is_speech:
            self.speech += 1
            self.silence = 0
        else:
            self.silence += 1
        sec = FRAME / SEND_RATE
        if self.silence * sec >= SILENCE_END or len(self.buf) * sec >= MAX_SEGMENT:
            seg, enough = np.concatenate(self.buf), self.speech * sec >= MIN_SPEECH
            self.reset()
            return seg if enough else None
        return None

    def take_partial(self) -> np.ndarray | None:
        """Whatever is mid-utterance right now (used at shutdown)."""
        seg = np.concatenate(self.buf) if self.buf and self.speech * FRAME / SEND_RATE >= MIN_SPEECH else None
        self.reset()
        return seg


def _transcribe(audio_i16: np.ndarray) -> str:
    """Runs on the single STT thread (MLX GPU streams are per-thread)."""
    import mlx_whisper

    r = mlx_whisper.transcribe(
        audio_i16.astype(np.float32) / 32768.0,
        path_or_hf_repo=STT_MODEL,
        language=STT_LANGUAGE,
        temperature=0.0,  # one greedy pass: fastest, and no fallback loops
        condition_on_previous_text=False,
        initial_prompt=STT_PROMPT,
        verbose=None,
    )
    parts = [
        s["text"] for s in r.get("segments", [])
        if s.get("no_speech_prob", 0.0) < 0.6 and s.get("avg_logprob", 0.0) > -1.0
    ]
    text = "".join(parts).strip()
    if text.lower() in HALLUCINATIONS or _is_garbage(text):
        return ""
    return text


def _is_garbage(text: str) -> bool:
    """Whisper locked to English sometimes emits a long run of repeated junk for speech it cannot
    decode (e.g. another language). Highly repetitive text compresses far better than real speech."""
    raw = text.encode()
    return len(raw) > 20 and len(raw) / len(zlib.compress(raw)) > 2.4


# ----------------------------------------------------------------------- agent


def datetime_now_text() -> str:
    """Current local time for the model (it cannot know the date, and needs it for 'tomorrow at 3')."""
    now = dt.datetime.now(gcal.TZ)
    return f"{now:%A %Y-%m-%d %H:%M} in {gcal.TZ} (UTC{now:%z})"


def make_beep() -> bytes:
    """Two short 'blink blink' tones (24 kHz int16). Same sound for start and stop."""
    def tone(sec: float, hz: float = 1046.0) -> np.ndarray:
        t = np.arange(int(RECV_RATE * sec)) / RECV_RATE
        fade = np.minimum(1.0, np.minimum(t, sec - t) * 150)  # ~7 ms fades, so no clicks
        return 0.3 * np.sin(2 * np.pi * hz * t) * fade

    wave = np.concatenate([tone(0.1), np.zeros(int(RECV_RATE * 0.08)), tone(0.1)])
    return (wave * 32767).astype(np.int16).tobytes()


BEEP = make_beep()


class Agent:
    def __init__(self):
        self.debug = "--debug" in sys.argv
        self.talking = False
        self.done = asyncio.Event()

        self.mic_q: asyncio.Queue[bytes] = asyncio.Queue(maxsize=400)
        self.agent_audio_q: asyncio.Queue[bytes] = asyncio.Queue(maxsize=400)
        self.seg_q: asyncio.Queue[tuple[np.ndarray, bool]] = asyncio.Queue()  # (audio, Gemini was speaking)
        self.play_q: asyncio.Queue[bytes] = asyncio.Queue()
        self.segmenter = Segmenter()
        self.stt_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="stt")

        self.conv_stop = asyncio.Event()  # belongs to the current conversation
        self.conv_task: asyncio.Task | None = None
        self.mute_until = 0.0  # ignore the mic while the alert plays
        self.playing_until = 0.0  # monotonic time at which everything queued for playback has finished
        self.last_activity = time.monotonic()
        self.model_speaking = False
        self.tool_busy = False
        self._recent = ""  # rolling text window for spotting the stop phrase during a conversation
        self._said = ""  # what you are saying to the assistant right now (assembled from Gemini's fragments)
        self._said_at = 0.0
        self._reply = ""  # what the assistant is saying right now (assembled from fragments)

    # -- helpers ---------------------------------------------------------------

    def show_said(self):
        """Print what you said to the assistant, one line per utterance."""
        text = " ".join(self._said.split())
        self._said = ""
        if text:
            print(f"  you: {text}", flush=True)

    def show_reply(self):
        """Print what the assistant said, one line per answer."""
        text = " ".join(self._reply.split())
        self._reply = ""
        if text:
            print(f"  minute: {text}", flush=True)

    def _note_said(self, text: str):
        if not self._said.strip():  # the local recogniser heard it; Gemini's transcript has not shown it
            self._said = text

    def beep(self):
        self.mute_until = time.monotonic() + BEEP_SECONDS + 0.15
        self.play_q.put_nowait(BEEP)

    def touch(self):
        self.last_activity = time.monotonic()

    def queue_speech(self, data: bytes):
        """Queue Gemini's audio and remember when it will finish playing (for the echo guard)."""
        now = time.monotonic()
        self.playing_until = max(now, self.playing_until) + len(data) / (RECV_RATE * 2)
        self.play_q.put_nowait(data)

    def flush_playback(self):
        self._drain(self.play_q)
        self.playing_until = time.monotonic()

    @staticmethod
    def _drain(q: asyncio.Queue):
        while not q.empty():
            q.get_nowait()

    # -- audio I/O -------------------------------------------------------------

    async def mic_capture(self):
        loop = asyncio.get_running_loop()

        def put(b: bytes):
            try:
                self.mic_q.put_nowait(b)
            except asyncio.QueueFull:
                pass

        def cb(indata, frames, t, status):
            loop.call_soon_threadsafe(put, indata.tobytes())

        with sd.InputStream(samplerate=SEND_RATE, channels=1, dtype="int16", blocksize=FRAME, callback=cb):
            await asyncio.Event().wait()

    async def mic_router(self):
        """WAITING: mic -> local VAD/Whisper (command words only).
        TALKING: mic -> Gemini Live, AND ALSO -> local VAD/Whisper, so 'Stop Minute' still works
        if Gemini's own transcription misses it, while it is speaking, or while a prompt is open."""
        while True:
            data = await self.mic_q.get()
            if time.monotonic() < self.mute_until:  # do not listen to our own alert
                if not self.talking:
                    self.segmenter.reset()
                continue
            raw = data
            gemini_speaking = time.monotonic() < self.playing_until + ECHO_TAIL
            if self.talking:
                try:
                    self.agent_audio_q.put_nowait(bytes(len(raw)) if gemini_speaking else raw)  # silence, not echo
                except asyncio.QueueFull:
                    pass
            seg = self.segmenter.feed(np.frombuffer(raw, dtype=np.int16))
            if seg is not None:
                self.seg_q.put_nowait((seg, self.talking and gemini_speaking))

    async def speaker(self):
        with sd.OutputStream(samplerate=RECV_RATE, channels=1, dtype="int16") as out:
            while True:
                data = await self.play_q.get()
                await asyncio.to_thread(out.write, np.frombuffer(data, dtype=np.int16))

    # -- WAITING: local command-word detection ---------------------------------

    async def stt_worker(self):
        loop = asyncio.get_running_loop()
        while True:
            seg, gemini_was_speaking = await self.seg_q.get()
            was_talking = self.talking
            text = await loop.run_in_executor(self.stt_pool, _transcribe, seg)
            if not text:
                continue
            if self.talking:
                self.on_local_text_while_talking(text, gemini_was_speaking)
            elif not was_talking:  # drop leftovers that finish after a conversation ended/began
                self.on_waiting_text(text)

    def on_local_text_while_talking(self, text: str, gemini_was_speaking: bool):
        """Second, independent 'Stop Minute' detector (local Whisper on the mic)."""
        if self.debug:
            print(f"  (local heard{' while Gemini spoke' if gemini_was_speaking else ''}) {text}", flush=True)
        if is_finish(text) and not (gemini_was_speaking and len(text) > BARGE_IN_MAX_CHARS):
            print("\n🛑 'Finished meeting' heard. Goodbye.", flush=True)
            self.done.set()
            self._note_said(text)
            self.end_conversation("finished meeting")
        elif STOP_RE.search(text):
            if gemini_was_speaking and len(text) > BARGE_IN_MAX_CHARS:
                return  # a long passage while Gemini talks is Gemini's own voice, not a command
            self._note_said(text)
            self.end_conversation("you said 'Stop Minute'")

    def on_waiting_text(self, text: str):
        if self.debug:
            print(f"  heard: {text}", flush=True)
        if is_finish(text):
            print("\n🛑 'Finished meeting' heard. Goodbye.", flush=True)
            self.done.set()
        elif m := WAKE_RE.search(text):
            print(f"  you: {text.strip()}", flush=True)
            self.start_conversation(text[m.end():].strip(" ,.!?"))

    # -- conversation lifecycle ------------------------------------------------

    def start_conversation(self, question: str):
        self.talking = True
        self.touch()
        self._recent = ""
        self._reply = ""
        self._drain(self.agent_audio_q)
        self._drain(self.seg_q)
        self.segmenter.reset()
        self.beep()
        self.conv_stop = asyncio.Event()
        print("\n🟢 Conversation started (say 'Stop Minute' to end it)", flush=True)
        self.conv_task = asyncio.create_task(self._conversation(question, self.conv_stop))

    def end_conversation(self, why: str, stop: asyncio.Event | None = None):
        if not self.talking or (stop is not None and stop is not self.conv_stop):
            return
        self.show_said()
        self.show_reply()
        self.talking = False
        self.conv_stop.set()
        self.model_speaking = False
        self.flush_playback()  # cut off any speech still queued, then the alert
        self._drain(self.agent_audio_q)
        self._drain(self.seg_q)
        self.segmenter.reset()
        self._recent = ""
        if not self.done.is_set():
            self.beep()
            print(f"\n⚪ Conversation ended ({why}). Waiting for 'Hello Minute'.", flush=True)

    def _setup(self) -> dict:
        """The Compass realtime `setup` message (same flat layout as Compass's own example)."""
        return {
            "responseModalities": ["AUDIO"],
            "systemInstruction": {"parts": [{"text": f"{SYSTEM_PROMPT}\nIt is now {datetime_now_text()}."}]},
            "tools": [{"functionDeclarations": _declarations()}],
            "inputAudioTranscription": {},  # what you said (shown on screen, and used to hear 'Stop Minute')
            "outputAudioTranscription": {},  # what the assistant said (shown on screen)
        }

    async def _conversation(self, question: str, stop: asyncio.Event):
        t0 = time.monotonic()
        try:
            async with compass_live(self._setup()) as session:
                if stop.is_set():
                    return
                print(f"  (Gemini Live connected via Compass in {time.monotonic() - t0:.2f}s)", flush=True)
                opening = (f"The user just asked you: {question}\nAnswer it now." if question else
                           "The user just called you by name. Greet them with one very short phrase "
                           "such as 'Yes?', then wait.")
                await session.send_text(opening)
                tasks = [asyncio.create_task(self._send(session, stop)),
                         asyncio.create_task(self._receive(session, stop))]
                waiters = [asyncio.create_task(stop.wait()), asyncio.create_task(self.done.wait())]
                try:
                    await asyncio.wait([*tasks, *waiters], return_when=asyncio.FIRST_COMPLETED)
                finally:
                    for t in (*tasks, *waiters):
                        t.cancel()
                    results = await asyncio.gather(*tasks, *waiters, return_exceptions=True)
                for r in results:
                    if isinstance(r, Exception):
                        raise r
        except asyncio.CancelledError:
            raise
        except Exception as e:
            print(f"\n(Gemini Live error: {type(e).__name__}: {e})", flush=True)
        finally:
            self.end_conversation("session closed", stop)  # no-op if it already ended

    async def _send(self, session, stop: asyncio.Event):
        while not stop.is_set():
            data = await self.agent_audio_q.get()
            await session.send_audio(data)

    async def _receive(self, session, stop: asyncio.Event):
        async for msg in session.stream():
            if stop.is_set():
                return
            if "goAway" in msg:
                print("\n(Gemini is closing the connection)", flush=True)
                return

            sc = msg.get("serverContent")
            if sc:
                for part in (sc.get("modelTurn") or {}).get("parts", []):
                    audio = part.get("inlineData") or {}
                    if audio.get("mimeType", "").startswith("audio/pcm") and audio.get("data"):
                        self.show_said()  # you finished; the assistant is answering
                        self.model_speaking = True
                        self.touch()
                        self.queue_speech(base64.b64decode(audio["data"]))
                if sc.get("interrupted"):  # barge-in: drop queued speech
                    self.flush_playback()
                heard = (sc.get("inputTranscription") or {}).get("text")
                if heard:
                    self.on_heard_while_talking(heard, stop)
                    if stop.is_set():
                        return
                if said := (sc.get("outputTranscription") or {}).get("text"):
                    self._reply += said
                if sc.get("turnComplete"):
                    self.model_speaking = False
                    self.show_reply()
                    self.touch()

            calls = (msg.get("toolCall") or {}).get("functionCalls")
            if calls:
                self.tool_busy = True
                try:
                    responses = [await dispatch(fc) for fc in calls]
                finally:
                    self.tool_busy = False
                    self.touch()
                await session.send_tool_responses(responses)

    def on_heard_while_talking(self, text: str, stop: asyncio.Event):
        if self.debug:
            print(f"  (Gemini heard) {text}", flush=True)
        self.touch()
        self._said += text
        self._said_at = time.monotonic()
        self._recent = (self._recent + text)[-120:]
        # (No 'finish meeting' here: Gemini's text arrives in fragments, so the short-utterance safety
        # rule cannot be applied. The local detector in on_local_text_while_talking handles it.)
        if STOP_RE.search(self._recent):
            self._recent = ""
            self.end_conversation("you said 'Stop Minute'", stop)

    async def ticker(self):
        """Ends a forgotten conversation (so the cloud session does not stay open for nothing)."""
        while True:
            await asyncio.sleep(0.5)
            if self._said and time.monotonic() - self._said_at > SAID_GAP:
                self.show_said()
            if (self.talking and not self.model_speaking and not self.tool_busy
                    and time.monotonic() > self.playing_until
                    and _pending_confirm is None and time.monotonic() - self.last_activity > TALK_IDLE):
                self.end_conversation("quiet for too long")

    # -- main loop -------------------------------------------------------------

    async def run(self):
        loop = asyncio.get_running_loop()
        print(f"Loading speech-to-text model {STT_MODEL} (first run downloads it)...", flush=True)
        t0 = time.monotonic()
        await loop.run_in_executor(self.stt_pool, _transcribe, np.zeros(SEND_RATE, dtype=np.int16))
        print(f"Model ready in {time.monotonic() - t0:.1f}s.", flush=True)
        print("Waiting. Say 'Hello Minute' to talk to me, 'Stop Minute' (or press Enter) to end "
              "the chat, 'Finished meeting' to quit.", flush=True)

        global _on_enter
        _on_enter = lambda: self.end_conversation("you pressed Enter")  # no-op unless talking
        threading.Thread(target=_stdin_thread, args=(loop,), daemon=True).start()
        tasks = [asyncio.create_task(c) for c in
                 (self.mic_capture(), self.mic_router(), self.speaker(), self.stt_worker(), self.ticker())]
        waiter = asyncio.create_task(self.done.wait())
        try:
            finished, _ = await asyncio.wait([*tasks, waiter], return_when=asyncio.FIRST_COMPLETED)
            for t in finished:
                if t is not waiter and t.exception():
                    raise t.exception()
        finally:
            extra = [self.conv_task] if self.conv_task else []
            for t in (*tasks, waiter, *extra):
                t.cancel()
            await asyncio.gather(*tasks, waiter, *extra, return_exceptions=True)


async def main():
    agent = Agent()
    try:
        await agent.run()
    except asyncio.CancelledError:  # Ctrl+C
        pass
    finally:
        agent.stt_pool.shutdown(wait=False)
        print("Session terminated.")


if __name__ == "__main__":
    if "COMPASS_API_KEY" not in os.environ or not COMPASS_REALTIME_URL:
        sys.exit("Set COMPASS_API_KEY and COMPASS_BASE_URL first (in .env; see .env.example).")
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
