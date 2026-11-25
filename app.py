#!/usr/bin/env python3
"""app.py: a web interface for the meeting assistant.

    python app.py            # then open http://127.0.0.1:8765 (it opens by itself)

It runs the same two programs as main.py and shows them in your browser:
  * minutes.py          the voice assistant   ("Hello Minute" to talk, "Stop Minute" to end the chat)
  * openthai.py --live  the live note-taker    (transcript.txt + summary.md in meetings/<date-time>/)

The page has Start / End meeting buttons with a loading percentage, the live conversation, the live
notes, and Approve / Cancel buttons for actions that need your confirmation (instead of typing y/N).

Security: it listens on 127.0.0.1 only, and every request needs a secret token that only the page
itself receives, plus a matching Host/Origin. A website you happen to visit therefore cannot press
"Approve" for you. No extra packages: it uses only the standard library and main.py's helpers.

    python app.py --port 9000 --no-browser
"""

from __future__ import annotations

import argparse
import json
import queue
import re
import secrets
import signal
import subprocess
import sys
import threading
import time
import webbrowser
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

import main as mt  # reuses Child (progress parsing), build_children and preflight

HERE = Path(__file__).resolve().parent
UI_FILE = HERE / "ui.html"
STOP_TIMEOUT = 600  # seconds to let the note-taker finish and summarise before giving up
CONFIRM_TIMEOUT = 90  # keep in step with CONFIRM_TIMEOUT in minutes.py (only used for the countdown)
MAX_SUMMARY_CHARS = 30000
MAX_BODY = 4096

NOTE_RE = re.compile(r"^\s*\[(\d\d:\d\d:\d\d)\]\s+(.*\S)\s*$")
YOU_RE = re.compile(r"^\s*you:\s*(.*\S)\s*$")
REPLY_RE = re.compile(r"^\s*minutes?:\s*(.*\S)\s*$", re.I)
ENDED_RE = re.compile(r"Conversation ended \((.*?)\)")


# ------------------------------------------------------------------------ hub


class Hub:
    """Everything the browser shows, plus a fan-out of live updates to connected pages."""

    def __init__(self):
        self.lock = threading.RLock()
        self.clients: set[queue.Queue] = set()
        self.reset()

    def _fresh_state(self) -> dict:
        return {"phase": "idle", "pct": 0, "label": "", "awake": False, "confirm": None, "warning": None,
                "problems": [], "error": None, "message": None, "started_at": None, "ready_at": None,
                "saved_to": None, "summary": None, "skip_summary": False}

    def reset(self):
        with self.lock:
            self.state = self._fresh_state()
            self.conversation: list[dict] = []
            self.notes: list[dict] = []
            self.log: deque[str] = deque(maxlen=400)

    def snapshot(self) -> dict:
        with self.lock:
            return {"type": "snapshot", "state": dict(self.state), "conversation": list(self.conversation),
                    "notes": list(self.notes), "log": list(self.log)}

    def subscribe(self) -> tuple[queue.Queue, dict]:
        q: queue.Queue = queue.Queue()
        with self.lock:  # snapshot and registration together, so no update falls between them
            snap = self.snapshot()
            self.clients.add(q)
        return q, snap

    def unsubscribe(self, q: queue.Queue):
        with self.lock:
            self.clients.discard(q)

    def _push(self, ev: dict):
        for q in list(self.clients):
            if q.qsize() > 5000:  # a page that stopped reading
                self.clients.discard(q)
            else:
                q.put_nowait(ev)

    def push_snapshot(self):
        with self.lock:
            self._push(self.snapshot())

    def set_state(self, **kw):
        with self.lock:
            self.state.update(kw)
            self._push({"type": "state", **kw})

    def add_conv(self, kind: str, text: str):
        ev = {"kind": kind, "text": text, "t": time.time()}
        with self.lock:
            self.conversation.append(ev)
            del self.conversation[:-1500]
            self._push({"type": "conv", **ev})

    def add_note(self, stamp: str, text: str):
        ev = {"time": stamp, "text": text}
        with self.lock:
            self.notes.append(ev)
            del self.notes[:-5000]
            self._push({"type": "note", **ev})

    def add_log(self, line: str):
        with self.lock:
            self.log.append(line)
            self._push({"type": "log", "line": line})


# -------------------------------------------------------------------- meeting


class Meeting:
    """Runs the assistant and the note-taker, and turns their output into hub updates."""

    def __init__(self, hub: Hub):
        self.hub = hub
        self.lock = threading.Lock()
        self.thread: threading.Thread | None = None
        self.stop_requested = threading.Event()
        self.terminal_signal = False  # Ctrl+C in the terminal also reaches the children by itself
        self.assistant: mt.Child | None = None
        self.notes: mt.Child | None = None
        self._partial: dict[str, str] = {}

    def running(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    # -- commands from the browser ---------------------------------------------

    def start(self, skip_summary: bool) -> dict:
        with self.lock:
            if self.running():
                return {"error": "A meeting is already running."}
            mt._load_env()
            args = SimpleNamespace(skip_summary=skip_summary, debug=False, latency_ms=560, out_dir=None, check=False)
            problems = mt.preflight(args, lambda pct, label: None)
            self.hub.reset()
            if problems:
                self.hub.set_state(phase="failed", problems=problems, error="Cannot start yet.")
                self.hub.push_snapshot()
                return {"error": "Cannot start yet.", "problems": problems}
            self.stop_requested.clear()
            self.terminal_signal = False
            self.hub.set_state(phase="loading", pct=10, label="starting", started_at=time.time(), skip_summary=skip_summary)
            self.hub.push_snapshot()
            self.thread = threading.Thread(target=self._supervise, args=(args,), daemon=True)
            self.thread.start()
            return {"ok": True}

    def stop(self) -> dict:
        if not self.running():
            return {"error": "No meeting is running."}
        self.stop_requested.set()
        return {"ok": True}

    def terminal_stop(self):
        """Ctrl+C in the terminal: the children got the signal too, so only wind down."""
        self.terminal_signal = True
        self.stop_requested.set()

    def answer(self, approve: bool) -> dict:
        if not self.hub.state["confirm"]:
            return {"error": "Nothing is waiting for your confirmation."}
        return self._send_line("y" if approve else "n")

    def end_conversation(self) -> dict:
        st = self.hub.state
        if st["phase"] != "running" or not st["awake"] or st["confirm"]:
            return {"error": "There is no conversation to end."}
        return self._send_line("")  # a bare Enter ends the chat

    def _send_line(self, text: str) -> dict:
        proc = self.assistant.proc if self.assistant else None
        if proc is None or proc.poll() is not None or proc.stdin is None:
            return {"error": "The assistant is not running."}
        try:
            proc.stdin.write((text + "\n").encode())
            proc.stdin.flush()
        except (BrokenPipeError, OSError, ValueError):
            return {"error": "The assistant is not running."}
        return {"ok": True}

    # -- reading the programs' output ------------------------------------------

    def _lines(self, child: mt.Child, text: str):
        buf = self._partial.get(child.name, "") + text.replace("\r", "\n")
        *lines, self._partial[child.name] = buf.split("\n")
        for line in lines:
            if line.strip():
                self._on_line(child, line.rstrip())

    def _on_line(self, child: mt.Child, line: str):
        hub = self.hub
        hub.add_log(f"[{child.name}] {line}")
        s = line.strip()
        if child.name == "notes":
            if m := NOTE_RE.match(line):
                hub.add_note(m.group(1), m.group(2))
            return
        if m := YOU_RE.match(line):
            hub.add_conv("you", m.group(1))
        elif m := REPLY_RE.match(line):
            hub.add_conv("minute", m.group(1))
        elif "Conversation started" in s:
            hub.set_state(awake=True)
            hub.add_conv("status", "Conversation started")
        elif m := ENDED_RE.search(s):
            hub.set_state(awake=False, confirm=None)
            hub.add_conv("status", f"Conversation ended ({m.group(1)})")
        elif s.startswith("[tool result]"):
            hub.add_conv("tool_result", s[len("[tool result]"):].strip())
        elif s.startswith("[tool]"):
            hub.add_conv("tool", s[len("[tool]"):].strip())
        elif s.startswith("[CONFIRM]"):
            prompt = s[len("[CONFIRM]"):].strip()
            hub.set_state(confirm={"prompt": prompt, "deadline": time.time() + CONFIRM_TIMEOUT})
            hub.add_conv("status", f"Waiting for your confirmation: {prompt}")
        elif "-> you answered" in s:
            hub.set_state(confirm=None)
            hub.add_conv("status", "You approved it." if "approved" in s else "Cancelled.")
        elif "no answer in" in s and "cancelled" in s:
            hub.set_state(confirm=None)
            hub.add_conv("status", "No answer: cancelled.")
        elif "Finished meeting" in s and "heard" in s:
            hub.add_conv("status", "You said “Finished meeting”.")
        elif s.startswith("(Gemini Live error") or s.startswith("(Gemini is closing"):
            hub.add_conv("error", s.strip("()"))

    # -- the supervisor thread (same logic as main.py, reporting to the browser) --

    def _supervise(self, args):
        hub = self.hub
        assistant, notes = mt.build_children(args)
        assistant.stdin = subprocess.PIPE  # the browser answers y/N and Enter through this pipe
        self.assistant, self.notes = assistant, notes
        children = (assistant, notes)
        self._partial = {}
        q: queue.Queue = queue.Queue()
        try:
            for c in children:
                c.start(q)
        except OSError as e:
            hub.set_state(phase="failed", error=f"Could not start the programs: {e}")
            return

        phase, last_pct, last_label = "loading", 10, "starting"
        deadline: float | None = None
        warned_notes = interrupted_notes = False
        outcome = ""

        def pct_now() -> int:
            return int(10 + 60 * assistant.progress + 30 * notes.progress)

        def begin_stopping(outcome_: str, interrupt_assistant: bool):
            nonlocal phase, deadline, interrupted_notes, outcome
            phase, outcome = "stopping", outcome_
            deadline = time.monotonic() + (15 if outcome_ in ("failed", "cancelled") else STOP_TIMEOUT)
            if not self.terminal_signal:  # from the terminal, Ctrl+C already reached them
                if notes.alive:
                    notes.interrupt()
                    interrupted_notes = True
                if interrupt_assistant:
                    assistant.interrupt()
            hub.set_state(phase="stopping", awake=False, confirm=None,
                          label="finishing the notes" if outcome_ not in ("failed", "cancelled") else "stopping")

        while True:
            try:
                kind, ch, text = q.get(timeout=0.2)
            except queue.Empty:
                kind = None
            if kind == "data":
                ch.feed(text)
                self._lines(ch, text)
            elif kind == "eof":
                ch.eof = True

            if phase == "loading":
                dead = [c for c in children if c.eof]
                if dead:
                    lines = []
                    for c in dead:
                        lines.append(f"The {c.name} stopped before it was ready (exit code {c.proc.wait()}). Last output:")
                        lines += [f"  {x}" for x in list(c.tail)[-10:]]
                    hub.set_state(error="\n".join(lines))
                    begin_stopping("failed", True)
                elif self.stop_requested.is_set():
                    begin_stopping("cancelled", True)
                elif all(c.progress >= 1.0 for c in children):
                    phase = "running"
                    hub.set_state(phase="running", pct=100, label="ready", ready_at=time.time())
                else:
                    pct = max(last_pct, min(99, pct_now()))
                    label = f"assistant: {assistant.label} · notes: {notes.label}"
                    if (pct, label) != (last_pct, last_label):
                        last_pct, last_label = pct, label
                        hub.set_state(pct=pct, label=label)

            elif phase == "running":
                if notes.eof and not warned_notes and not assistant.eof:
                    warned_notes = True
                    hub.set_state(warning="The note-taker stopped, so the transcript has stopped. "
                                          "The assistant keeps running. Check the log below.")
                if assistant.eof:
                    code = assistant.proc.wait()
                    ok = code == 0
                    begin_stopping("finished" if ok else "assistant stopped unexpectedly", False)
                    if not ok:
                        hub.set_state(error=f"The assistant stopped unexpectedly (exit code {code}). Check the log below.")
                elif self.stop_requested.is_set():
                    begin_stopping("ended", True)

            if phase == "stopping":
                if all(c.eof for c in children):
                    break
                if deadline and time.monotonic() > deadline:
                    for c in children:
                        c.kill()
                    hub.set_state(error="The programs did not finish in time and were stopped.")
                    break

        for c in children:
            if c.proc.poll() is None:
                c.kill()
        self._finish(outcome, notes, interrupted_notes)

    def _finish(self, outcome: str, notes: mt.Child, interrupted_notes: bool):
        hub = self.hub
        if outcome in ("failed", "cancelled"):
            hub.set_state(phase="failed" if outcome == "failed" else "idle", awake=False, confirm=None,
                          pct=0, label="", message="Cancelled." if outcome == "cancelled" else None)
            return
        saved, summary = None, None
        if notes.saved_to and (Path(notes.saved_to) / "transcript.txt").exists():
            folder = Path(notes.saved_to)
            saved = str(folder)
            f = folder / "summary.md"
            if f.exists():
                summary = f.read_text(encoding="utf-8", errors="replace")[:MAX_SUMMARY_CHARS]
        msg = None
        if saved is None:
            msg = "No transcript was saved."
        elif summary is None and not hub.state["skip_summary"]:
            msg = "The transcript was saved, but no summary was created. Check the log below."
        hub.set_state(phase="done", awake=False, confirm=None, saved_to=saved, summary=summary, message=msg)


# ------------------------------------------------------------------------ http


def make_handler(hub: Hub, meeting: Meeting, token: str, port: int):
    allowed_hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
    html = UI_FILE.read_text(encoding="utf-8").replace("__TOKEN__", token)

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "MeetingUI"

        def log_message(self, *args):  # keep the terminal quiet
            pass

        # -- helpers ---------------------------------------------------------
        def _host_ok(self) -> bool:
            if self.headers.get("Host", "") not in allowed_hosts:  # blocks DNS-rebinding
                return False
            origin = self.headers.get("Origin")
            return origin is None or origin.removeprefix("http://") in allowed_hosts and origin.startswith("http://")

        def _token_ok(self, supplied: str | None) -> bool:
            return supplied is not None and secrets.compare_digest(supplied, token)

        def _send(self, code: int, body: bytes, ctype: str = "application/json", extra: dict | None = None):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, obj: dict):
            self._send(code, json.dumps(obj).encode())

        # -- routes ----------------------------------------------------------
        def do_GET(self):
            if not self._host_ok():
                return self._json(403, {"error": "forbidden"})
            url = urlparse(self.path)
            if url.path == "/":
                csp = ("default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
                       "connect-src 'self'; img-src data:; base-uri 'none'; form-action 'none'; frame-ancestors 'none'")
                return self._send(200, html.encode(), "text/html; charset=utf-8", {"Content-Security-Policy": csp})
            if url.path == "/events":
                if not self._token_ok((parse_qs(url.query).get("t") or [None])[0]):
                    return self._json(403, {"error": "forbidden"})
                return self._events()
            self._json(404, {"error": "not found"})

        def do_POST(self):
            if not self._host_ok() or not self._token_ok(self.headers.get("X-Token")):
                return self._json(403, {"error": "forbidden"})
            if "application/json" not in self.headers.get("Content-Type", ""):
                return self._json(415, {"error": "send JSON"})
            try:
                n = int(self.headers.get("Content-Length", "0"))
                if not 0 <= n <= MAX_BODY:
                    raise ValueError
                body = json.loads(self.rfile.read(n) or b"{}")
                if not isinstance(body, dict):
                    raise ValueError
            except (ValueError, json.JSONDecodeError):
                return self._json(400, {"error": "bad request"})
            path = urlparse(self.path).path
            if path == "/api/start":
                result = meeting.start(bool(body.get("skip_summary")))
            elif path == "/api/stop":
                result = meeting.stop()
            elif path == "/api/confirm":
                result = meeting.answer(body.get("approve") is True)
            elif path == "/api/end_conversation":
                result = meeting.end_conversation()
            else:
                return self._json(404, {"error": "not found"})
            self._json(409 if "error" in result else 200, result)

        def _events(self):
            """Server-sent events: a snapshot first, then every change as it happens."""
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            q, snap = hub.subscribe()
            try:
                self.wfile.write(f"data: {json.dumps(snap)}\n\n".encode())
                self.wfile.flush()
                while True:
                    try:
                        ev = q.get(timeout=15)
                        chunk = f"data: {json.dumps(ev)}\n\n"
                    except queue.Empty:
                        chunk = ": keep-alive\n\n"
                    self.wfile.write(chunk.encode())
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
            finally:
                hub.unsubscribe(q)
            self.close_connection = True

    return Handler


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def serve(port: int, open_browser: bool) -> int:
    hub = Hub()
    meeting = Meeting(hub)
    token = secrets.token_urlsafe(24)
    try:
        server = Server(("127.0.0.1", port), make_handler(hub, meeting, token, port))
    except OSError as e:
        print(f"Cannot listen on 127.0.0.1:{port} ({e}). Is it already running? Try --port.", file=sys.stderr)
        return 1
    url = f"http://127.0.0.1:{port}/"
    exiting = threading.Event()

    def graceful_exit():
        if meeting.running():
            meeting.terminal_stop()
            meeting.thread.join(STOP_TIMEOUT + 30)
        server.shutdown()

    def on_sigint(*_):
        if exiting.is_set():
            return  # a second Ctrl+C already reached the children; just let things finish
        exiting.set()
        print("\nShutting down (finishing the notes first if a meeting is running)...", flush=True)
        threading.Thread(target=graceful_exit, daemon=True).start()

    signal.signal(signal.SIGINT, on_sigint)
    print(f"Meeting interface: {url}   (Ctrl+C to quit)", flush=True)
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    finally:
        server.server_close()
    print("Bye.")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Web interface for the meeting assistant.")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--no-browser", action="store_true", help="do not open the page automatically")
    a = p.parse_args()
    if not UI_FILE.is_file():
        print(f"Missing {UI_FILE.name} next to app.py.", file=sys.stderr)
        return 1
    return serve(a.port, not a.no_browser)


if __name__ == "__main__":
    sys.exit(main())
