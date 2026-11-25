"""minutes_02.py: an English-only AI voice agent that stays silent until you wake it.

  * WAITING: it says nothing. A small speech model running on this Mac (mlx-whisper) listens only
    for the wake phrase; nothing is sent to the cloud and nothing is saved.
  * Say "Hey Minutes" (optionally with your question: "Hey Minutes, what time is it?") and a short
    beep plays, then Gemini 3.8 Live (through CompassLLM) starts a voice conversation. It speaks first.
  * Say "Stop Minutes" (or press Enter) to end the conversation. It goes silent again and waits for the
    next "Hey Minutes". It also goes back to sleep after IDLE seconds of silence. Ctrl+C quits.

It hears and speaks English only: the local recogniser only decodes English, and Gemini is told to
reply only in English.

Setup:
    pip install websockets sounddevice numpy mlx-whisper
    .env: COMPASS_API_KEY=... and COMPASS_BASE_URL=https://<compass-host>/compass-api/v1
                                         (optional: COMPASS_LIVE_MODEL, COMPASS_REALTIME_URL,
                                          MINUTES_STT_MODEL)
    python minutes_02.py                 (add --debug to see what the local recogniser hears)

Use headphones if you can. With speakers the microphone hears the agent's voice; the code mutes the
microphone while the agent speaks to stop it answering itself, but you then cannot talk over it.
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
import sys
import threading
import time
import zlib
from urllib.parse import urlencode

import numpy as np
import sounddevice as sd
from websockets.asyncio.client import connect as ws_connect
from websockets.exceptions import ConnectionClosedOK, InvalidStatus


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
                os.environ.setdefault(k.removeprefix("export ").strip(), v)


_load_env()

# ------------------------------------------------------------------ settings

COMPASS_BASE_URL = os.environ.get("COMPASS_BASE_URL", "").rstrip("/")  # e.g. https://<compass-host>/compass-api/v1
COMPASS_REALTIME_URL = os.environ.get("COMPASS_REALTIME_URL") or (
    re.sub(r"^https?://", "wss://", COMPASS_BASE_URL) + "/realtime" if COMPASS_BASE_URL else "")
LIVE_MODEL = os.environ.get("COMPASS_LIVE_MODEL", "gemini-3.8-live")
STT_MODEL = os.environ.get("MINUTES_STT_MODEL", "mlx-community/whisper-large-v3-turbo")
STT_PROMPT = "Hey Minutes. Stop Minutes."  # nudges Whisper to spell the command words correctly

SEND_RATE = 16000  # microphone -> Whisper / Gemini
RECV_RATE = 24000  # Gemini -> speaker
FRAME = 512  # 32 ms of microphone audio per block

# Voice-activity segmentation for the local recogniser.
PREROLL = 0.3  # seconds kept before speech starts, so the first syllable is not cut
SILENCE_END = 0.6  # seconds of quiet that end an utterance
MAX_SEGMENT = 15.0  # force-cut very long speech
MIN_SPEECH = 0.3  # ignore blips shorter than this

IDLE = 60.0  # seconds of silence before a forgotten conversation ends itself
ECHO_TAIL = 0.7  # keep the microphone muted this long after the agent stops speaking
SAID_GAP = 1.2  # a pause this long closes what you were saying, so it can be printed
BARGE_IN_MAX_CHARS = 40  # while the agent speaks, only SHORT local text counts as a command from you
BEEP_SECONDS = 0.12

# The name as Whisper might write it: minute(s), minit, minnit, minuet, mini, minnie, "min it".
_NAME = r"(?:min+(?:ute|ut|uet|it|ite|ie|y|i)s?|mi[\s-]?nit(?:e|s)?|min[\s-]it)\b"
_SEP = r"[\s,.!?-]*"
# Wake: a greeting + the name ("hey minutes", "hey minute", "hello minutes"...). Words like
# "okay minutes" or "a minute" are deliberately not accepted: they are ordinary meeting talk.
WAKE_RE = re.compile(r"\b(?:hey+|hay+|hei|heya|hi+|hiya|hello|hullo|yo)" + _SEP + r"(?:there" + _SEP + r")?" + _NAME, re.I)
# Stop: "stop minutes", "bye minutes", "goodbye minutes", "thank you minutes", "minutes, please stop"...
_STOP_WORDS = r"(?:stop+|stahp|end|quit|exit|bye|good\s?bye|enough|thanks?|thank\s+you|that'?s\s+all)"
STOP_RE = re.compile(
    r"\b" + _STOP_WORDS + _SEP + r"(?:the\s+)?" + _NAME
    + r"|\b" + _NAME + _SEP + r"(?:please" + _SEP + r")?(?:stop+|stahp|bye|good\s?bye|enough|that'?s\s+all)\b", re.I)

HALLUCINATIONS = {"thank you.", "thank you", "thanks for watching!", "thanks for watching.", "you", "."}  # Whisper on silence

SYSTEM_PROMPT = (
    "You are Minutes, a friendly voice assistant. Speak in natural, spoken English: one to three short "
    "sentences, no lists or markdown. You hear and speak English only: always reply in English, even if "
    "you are addressed in another language, and if you cannot understand the user, ask them in English "
    "to repeat. Speech recognition can mishear, so if something seems odd, check what they meant. "
    "Never say the phrase 'Stop Minutes' yourself."
)


# ------------------------------------------------------- local speech recogniser


class Segmenter:
    """Energy-based voice activity detection: cuts the microphone stream into utterances."""

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
        # Track the room's noise floor: fast on quiet frames, very slowly during speech.
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


def transcribe(audio_i16: np.ndarray) -> str:
    """English-only speech to text. Runs on the single STT thread (MLX GPU streams are per-thread)."""
    import mlx_whisper

    r = mlx_whisper.transcribe(
        audio_i16.astype(np.float32) / 32768.0,
        path_or_hf_repo=STT_MODEL,
        language="en",  # English only: no language guessing
        temperature=0.0,
        condition_on_previous_text=False,
        initial_prompt=STT_PROMPT,
        verbose=None,
    )
    text = "".join(
        s["text"] for s in r.get("segments", [])
        if s.get("no_speech_prob", 0.0) < 0.6 and s.get("avg_logprob", 0.0) > -1.0
    ).strip()
    raw = text.encode()
    garbage = len(raw) > 20 and len(raw) / len(zlib.compress(raw)) > 2.4  # repeated junk = speech it could not decode
    return "" if text.lower() in HALLUCINATIONS or garbage else text


# --------------------------------------------- Gemini Live through CompassLLM


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
        await self._send({"clientContent": {"turns": [{"role": "user", "parts": [{"text": text}]}], "turnComplete": True}})

    async def send_audio(self, pcm16: bytes) -> None:
        await self._send({"realtimeInput": {"audio": {"data": base64.b64encode(pcm16).decode(),
                                                      "mimeType": f"audio/pcm;rate={SEND_RATE}"}}})


@contextlib.asynccontextmanager
async def compass_live(setup: dict):
    """Open a Gemini Live session via CompassLLM and wait for setupComplete."""
    key = os.environ.get("COMPASS_API_KEY")
    if not key:
        raise RuntimeError("COMPASS_API_KEY is not set (put it in .env).")
    if not COMPASS_REALTIME_URL:
        raise RuntimeError("COMPASS_BASE_URL is not set (put it in .env; see .env.example).")
    url = f"{COMPASS_REALTIME_URL}?{urlencode({'model': LIVE_MODEL})}"
    kwargs = {"ssl": _ssl_context()} if url.startswith("wss://") else {}
    try:
        ws = await ws_connect(url, additional_headers={"Authorization": f"Bearer {key}"},
                              open_timeout=15, max_size=None, **kwargs)
    except InvalidStatus as e:
        code = e.response.status_code
        hint = {401: "check COMPASS_API_KEY", 403: "this key may not have access to the model",
                404: f"model '{LIVE_MODEL}' or the URL was not found"}.get(code)
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


# ----------------------------------------------------------------------- agent


def make_beep() -> bytes:
    """One short beep (24 kHz int16): 'I heard you, connecting'."""
    t = np.arange(int(RECV_RATE * BEEP_SECONDS)) / RECV_RATE
    fade = np.minimum(1.0, np.minimum(t, BEEP_SECONDS - t) * 150)  # ~7 ms fades, no clicks
    return (0.3 * np.sin(2 * np.pi * 1046.0 * t) * fade * 32767).astype(np.int16).tobytes()


BEEP = make_beep()


def now_text() -> str:
    now = dt.datetime.now().astimezone()
    return f"{now:%A %Y-%m-%d %H:%M} (UTC{now:%z})"


class Agent:
    def __init__(self):
        self.debug = "--debug" in sys.argv
        self.talking = False
        self.done = asyncio.Event()

        self.mic_q: asyncio.Queue[bytes] = asyncio.Queue(maxsize=400)
        self.agent_audio_q: asyncio.Queue[bytes] = asyncio.Queue(maxsize=400)  # microphone -> Gemini
        self.seg_q: asyncio.Queue[tuple[np.ndarray, bool]] = asyncio.Queue()  # (audio, agent was speaking)
        self.play_q: asyncio.Queue[bytes] = asyncio.Queue()  # Gemini / beep -> speaker
        self.segmenter = Segmenter()
        self.stt_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="stt")

        self.conv_stop = asyncio.Event()  # belongs to the current conversation
        self.conv_task: asyncio.Task | None = None
        self.mute_until = 0.0  # ignore the microphone while the beep plays
        self.playing_until = 0.0  # when everything queued for playback will have finished
        self.last_activity = time.monotonic()
        self._recent = ""  # rolling window of what Gemini heard, for spotting the stop phrase
        self._said, self._said_at = "", 0.0  # what you are saying right now (assembled from fragments)
        self._reply = ""  # what the agent is saying right now

    # -- small helpers ---------------------------------------------------------

    @staticmethod
    def _drain(q: asyncio.Queue):
        while not q.empty():
            q.get_nowait()

    def touch(self):
        self.last_activity = time.monotonic()

    def beep(self):
        self.mute_until = time.monotonic() + BEEP_SECONDS + 0.15
        self.play_q.put_nowait(BEEP)

    def queue_speech(self, data: bytes):
        """Queue the agent's audio and remember when it will finish playing (for the echo guard)."""
        self.playing_until = max(time.monotonic(), self.playing_until) + len(data) / (RECV_RATE * 2)
        self.play_q.put_nowait(data)

    def flush_playback(self):
        self._drain(self.play_q)
        self.playing_until = time.monotonic()

    def show_said(self):
        text = " ".join(self._said.split())
        self._said = ""
        if text:
            print(f"  you: {text}", flush=True)

    def show_reply(self):
        text = " ".join(self._reply.split())
        self._reply = ""
        if text:
            print(f"  minutes: {text}", flush=True)

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
        """Microphone -> local recogniser (always), and -> Gemini (only while a conversation is on)."""
        while True:
            data = await self.mic_q.get()
            if time.monotonic() < self.mute_until:  # do not listen to our own beep
                if not self.talking:
                    self.segmenter.reset()
                continue
            agent_speaking = time.monotonic() < self.playing_until + ECHO_TAIL
            if self.talking:
                try:  # while the agent speaks send silence, not its own voice coming back from the speaker
                    self.agent_audio_q.put_nowait(bytes(len(data)) if agent_speaking else data)
                except asyncio.QueueFull:
                    pass
            seg = self.segmenter.feed(np.frombuffer(data, dtype=np.int16))
            if seg is not None:
                self.seg_q.put_nowait((seg, self.talking and agent_speaking))

    async def speaker(self):
        with sd.OutputStream(samplerate=RECV_RATE, channels=1, dtype="int16") as out:
            while True:
                data = await self.play_q.get()
                await asyncio.to_thread(out.write, np.frombuffer(data, dtype=np.int16))

    # -- local recogniser: wake and stop phrases -------------------------------

    async def stt_worker(self):
        loop = asyncio.get_running_loop()
        while True:
            seg, agent_was_speaking = await self.seg_q.get()
            was_talking = self.talking
            text = await loop.run_in_executor(self.stt_pool, transcribe, seg)
            if not text:
                continue
            if self.debug:
                print(f"  (local heard{' while agent spoke' if agent_was_speaking else ''}) {text}", flush=True)
            if self.talking:
                self.on_local_text_while_talking(text, agent_was_speaking)
            elif not was_talking:  # ignore a leftover that finished after a conversation just ended
                self.on_waiting_text(text)

    def on_waiting_text(self, text: str):
        if m := WAKE_RE.search(text):
            print(f"  you: {text.strip()}", flush=True)
            self.start_conversation(text[m.end():].strip(" ,.!?"))

    def on_local_text_while_talking(self, text: str, agent_was_speaking: bool):
        """A second, independent 'Stop Minutes' detector, in case Gemini's transcript misses it."""
        if STOP_RE.search(text) and not (agent_was_speaking and len(text) > BARGE_IN_MAX_CHARS):
            if not self._said.strip():
                self._said = text
            self.end_conversation("you said 'Stop Minutes'")

    # -- conversation lifecycle ------------------------------------------------

    def start_conversation(self, question: str):
        self.talking = True
        self.touch()
        self._recent, self._said, self._reply = "", "", ""
        self._drain(self.agent_audio_q)
        self._drain(self.seg_q)
        self.segmenter.reset()
        self.beep()
        self.conv_stop = asyncio.Event()
        print("\n🟢 Awake. Say 'Stop Minutes' (or press Enter) to end the conversation.", flush=True)
        self.conv_task = asyncio.create_task(self._conversation(question, self.conv_stop))

    def end_conversation(self, why: str, stop: asyncio.Event | None = None):
        if not self.talking or (stop is not None and stop is not self.conv_stop):
            return
        self.show_said()
        self.show_reply()
        self.talking = False
        self.conv_stop.set()
        self.flush_playback()  # cut off any speech still queued
        self._drain(self.agent_audio_q)
        self._drain(self.seg_q)
        self.segmenter.reset()
        self._recent = ""
        print(f"\n⚪ Asleep ({why}). Silent until you say 'Hey Minutes'.", flush=True)

    def _setup(self) -> dict:
        """The Compass realtime `setup` message."""
        return {
            "responseModalities": ["AUDIO"],
            "systemInstruction": {"parts": [{"text": f"{SYSTEM_PROMPT}\nIt is now {now_text()}."}]},
            "inputAudioTranscription": {},  # what you said (shown on screen, and used to hear 'Stop Minutes')
            "outputAudioTranscription": {},  # what the agent said (shown on screen)
        }

    async def _conversation(self, question: str, stop: asyncio.Event):
        t0 = time.monotonic()
        try:
            async with compass_live(self._setup()) as session:
                if stop.is_set():
                    return
                print(f"  (connected in {time.monotonic() - t0:.1f}s)", flush=True)
                await session.send_text(
                    f"The user just said: {question}\nAnswer it now." if question else
                    "The user just woke you by saying 'Hey Minutes'. Greet them in one short English sentence "
                    "and ask how you can help.")
                tasks = [asyncio.create_task(self._send(session, stop)), asyncio.create_task(self._receive(session, stop))]
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
            print(f"\n(Compass error: {type(e).__name__}: {e})", flush=True)
        finally:
            self.end_conversation("session closed", stop)  # no-op if it already ended

    async def _send(self, session: LiveSession, stop: asyncio.Event):
        while not stop.is_set():
            await session.send_audio(await self.agent_audio_q.get())

    async def _receive(self, session: LiveSession, stop: asyncio.Event):
        async for msg in session.stream():
            if stop.is_set():
                return
            if "goAway" in msg:
                print("\n(Compass is closing the connection)", flush=True)
                return
            sc = msg.get("serverContent")
            if not sc:
                continue
            for part in (sc.get("modelTurn") or {}).get("parts", []):
                audio = part.get("inlineData") or {}
                if audio.get("mimeType", "").startswith("audio/pcm") and audio.get("data"):
                    self.show_said()  # you finished; the agent is answering
                    self.touch()
                    self.queue_speech(base64.b64decode(audio["data"]))
            if sc.get("interrupted"):  # you cut in: drop queued speech
                self.flush_playback()
            if heard := (sc.get("inputTranscription") or {}).get("text"):
                self.on_heard(heard, stop)
                if stop.is_set():
                    return
            if said := (sc.get("outputTranscription") or {}).get("text"):
                self._reply += said
            if sc.get("turnComplete"):
                self.show_reply()
                self.touch()

    def on_heard(self, text: str, stop: asyncio.Event):
        """A fragment of what Gemini heard you say."""
        self.touch()
        self._said += text
        self._said_at = time.monotonic()
        self._recent = (self._recent + text)[-120:]
        if STOP_RE.search(self._recent):
            self._recent = ""
            self.end_conversation("you said 'Stop Minutes'", stop)

    async def ticker(self):
        """Prints what you said after a pause, and puts a forgotten conversation back to sleep."""
        while True:
            await asyncio.sleep(0.5)
            now = time.monotonic()
            if self._said and now - self._said_at > SAID_GAP:
                self.show_said()
            if self.talking and now > self.playing_until and now - self.last_activity > IDLE:
                self.end_conversation("quiet for too long")

    # -- main loop -------------------------------------------------------------

    def _enter_pressed(self):
        self.end_conversation("you pressed Enter")  # does nothing unless a conversation is on

    async def run(self):
        loop = asyncio.get_running_loop()
        print(f"Loading the speech model {STT_MODEL} (the first run downloads it)...", flush=True)
        t0 = time.monotonic()
        await loop.run_in_executor(self.stt_pool, transcribe, np.zeros(SEND_RATE, dtype=np.int16))
        print(f"Ready in {time.monotonic() - t0:.1f}s. I am silent until you say 'Hey Minutes'. (Ctrl+C to quit)", flush=True)

        def stdin_thread():  # Enter key = end the conversation
            for _ in sys.stdin:
                loop.call_soon_threadsafe(self._enter_pressed)

        threading.Thread(target=stdin_thread, daemon=True).start()
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
        print("Bye.")


if __name__ == "__main__":
    if "COMPASS_API_KEY" not in os.environ or not COMPASS_REALTIME_URL:
        sys.exit("Set COMPASS_API_KEY and COMPASS_BASE_URL first (in .env; see .env.example).")
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
