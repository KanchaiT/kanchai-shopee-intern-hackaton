#!/usr/bin/env python3
"""main.py: start a meeting with ONE command.

    python main.py

Runs two programs at the same time and shows a loading percentage until both are ready:

  * minute.py           the voice assistant: say "Hey Minutes" to talk, "Stop Minute" to end
                        the chat. (Local wake-word detection; Gemini Live only while talking.)
  * openthai.py --live  the note-taker: streams the microphone to iApp and writes
                        meetings/<date-time>/transcript.txt and summary.md.

Say "Finished meeting" (or press Ctrl+C) to end the meeting: the assistant stops, the note-taker
finishes its last sentences, writes the summary, and then this program exits.

    python main.py --check            only run the pre-flight checks
    python main.py --skip-summary     transcript only (no COMPASS_API_KEY needed)
    python main.py --debug            the assistant prints what its local recogniser hears
"""

from __future__ import annotations

import argparse
import codecs
import importlib.util
import os
import queue
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

HERE = Path(__file__).resolve().parent
MINUTE_SCRIPT = HERE / "minutes.py"
NOTES_SCRIPT = HERE / "openthai.py"
STOP_TIMEOUT = 600  # seconds to let the note-taker finish and summarise before giving up
BAR = 30
SPIN = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

PACKAGES = [("numpy", "numpy"), ("sounddevice", "sounddevice"), ("mlx_whisper", "mlx-whisper"),
            ("google.genai", "google-genai"), ("websockets", "'websockets>=12'"), ("requests", "requests")]


def _load_env(path: Path = HERE / ".env") -> None:
    """Minimal .env loader (the two programs load it themselves; this is for the pre-flight)."""
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


# ------------------------------------------------------------------- pre-flight


def preflight(args, report) -> list[str]:
    """Quick checks so a problem shows up in a second, not after the model has loaded.
    `report(pct, label)` draws progress (this phase covers 0-10%). Returns the problems found."""
    problems: list[str] = []

    def scripts():
        for f in (MINUTE_SCRIPT, NOTES_SCRIPT):
            if not f.is_file():
                problems.append(f"Missing file: {f.name}")

    def packages():
        missing = []
        for mod, pip_name in PACKAGES:
            try:
                ok = importlib.util.find_spec(mod) is not None
            except (ImportError, ValueError):
                ok = False
            if not ok:
                missing.append(pip_name)
        if missing:
            problems.append(f"Missing Python packages: pip install {' '.join(missing)}")

    def keys():
        need = [("COMPASS_API_KEY", "the voice assistant (Gemini Live through Compass) and the summary"),
                ("COMPASS_BASE_URL", "the Compass address, e.g. https://<compass-host>/compass-api/v1"),
                ("IAPP_API_KEY", "live transcription (iApp)")]
        for name, what in need:
            if not os.environ.get(name):
                problems.append(f"{name} is not set (needed for {what}). Put it in .env.")

    def audio():
        try:
            import sounddevice as sd
        except ImportError:
            return  # already reported above
        for kind, label in (("input", "microphone"), ("output", "speaker")):
            try:
                sd.query_devices(kind=kind)
            except Exception as e:
                problems.append(f"No {label} available ({e}).")

    checks = [("checking files", scripts), ("checking packages", packages),
              ("checking API keys", keys), ("checking microphone and speaker", audio)]
    for i, (label, fn) in enumerate(checks):
        report(int(10 * i / len(checks)), label)
        fn()
    report(10, "checks passed" if not problems else "checks failed")
    return problems


# --------------------------------------------------------------------- progress


class Progress:
    def __init__(self):
        self.tty = sys.stdout.isatty()
        self.t0 = time.monotonic()
        self.last = (-1, "")
        self.tick = 0

    def show(self, pct: int, label: str) -> None:
        pct = max(0, min(100, pct))
        if not self.tty:  # plain log lines when piped to a file
            if (pct, label) != self.last:
                print(f"  loading {pct:3d}%  {label}", flush=True)
                self.last = (pct, label)
            return
        self.tick += 1
        n = round(BAR * pct / 100)
        line = (f"  [{'█' * n}{'░' * (BAR - n)}] {pct:3d}%  {SPIN[self.tick % len(SPIN)]} {label}"
                f"  {time.monotonic() - self.t0:.0f}s")
        cols = shutil.get_terminal_size((100, 20)).columns
        sys.stdout.write("\r\033[K" + line[:cols - 1])
        sys.stdout.flush()

    def clear(self) -> None:
        if self.tty:
            sys.stdout.write("\r\033[K")
            sys.stdout.flush()


# --------------------------------------------------------------------- children


class Child:
    """One program run as a subprocess; its output is parsed for 'ready' milestones."""

    def __init__(self, name: str, tag: str, argv: list[str], start: tuple[float, str],
                 milestones: list[tuple[str, float, str]], stdin):
        self.name, self.tag, self.argv, self.stdin = name, tag, argv, stdin
        self.progress, self.label = start
        self.milestones = milestones  # (line starts with, progress 0-1, label shown while waiting)
        self.tail: deque[str] = deque(maxlen=25)
        self.partial = ""
        self.at_line_start = True
        self.eof = False
        self.proc: subprocess.Popen | None = None
        self.saved_to: str | None = None

    def start(self, q: queue.Queue) -> None:
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        self.proc = subprocess.Popen(self.argv, stdin=self.stdin, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, cwd=HERE, env=env, bufsize=0)
        threading.Thread(target=self._read, args=(q,), daemon=True).start()

    def _read(self, q: queue.Queue) -> None:
        dec = codecs.getincrementaldecoder("utf-8")("replace")
        fd = self.proc.stdout.fileno()
        while True:
            try:
                b = os.read(fd, 4096)
            except OSError:
                break
            if not b:
                break
            q.put(("data", self, dec.decode(b)))
        q.put(("eof", self, ""))

    def feed(self, text: str) -> None:
        """Record output for error reports and advance progress on milestone lines."""
        self.partial += text
        *lines, self.partial = self.partial.split("\n")
        for raw in lines:
            line = raw.split("\r")[-1].rstrip()
            if not line.strip():
                continue
            self.tail.append(line)
            m = re.search(r"Saving to (.+)$", line)
            if m:
                self.saved_to = m.group(1).strip()
            for prefix, prog, label in self.milestones:
                if line.strip().startswith(prefix) and prog > self.progress:
                    self.progress, self.label = prog, label

    @property
    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def interrupt(self) -> None:
        if self.alive:
            try:
                self.proc.send_signal(signal.SIGINT)
            except ProcessLookupError:
                pass

    def kill(self) -> None:
        if self.alive:
            self.proc.terminate()
            try:
                self.proc.wait(5)
            except subprocess.TimeoutExpired:
                self.proc.kill()


def emit(child: Child, text: str, color: bool) -> None:
    """Print child output with a [tag] in front of every line (partial lines, e.g. a y/N prompt,
    are shown immediately)."""
    tag = {"assistant": "\033[36m[assistant]\033[0m ", "notes": "\033[35m[notes]\033[0m "}[child.name] if color \
        else f"[{child.name}] "
    out = []
    pieces = text.replace("\r", "").split("\n")
    for i, piece in enumerate(pieces):
        if piece:
            if child.at_line_start:
                out.append(tag)
            out.append(piece)
            child.at_line_start = False
        if i < len(pieces) - 1:  # a newline followed this piece
            out.append("\n")
            child.at_line_start = True
    sys.stdout.write("".join(out))
    sys.stdout.flush()


# ------------------------------------------------------------------------- main


def build_children(args) -> tuple[Child, Child]:
    minute = Child(
        "assistant", "assistant",
        [sys.executable, "-u", str(MINUTE_SCRIPT), *(["--debug"] if args.debug else [])],
        (0.05, "starting"),
        [("Loading speech-to-text model", 0.15, "loading speech model (first run downloads ~1.6 GB)"),
         ("Model ready in", 0.9, "speech model ready"),
         ("Waiting.", 1.0, "ready")],
        stdin=None)  # inherits the terminal: it asks y/N questions and reads Enter
    notes_argv = [sys.executable, "-u", str(NOTES_SCRIPT), "--live", "--latency-ms", str(args.latency_ms)]
    if args.skip_summary:
        notes_argv.append("--skip-summary")
    if args.out_dir:
        notes_argv += ["-o", args.out_dir]
    notes = Child(
        "notes", "notes", notes_argv, (0.1, "opening microphone"),
        [("Listening to the microphone", 0.5, "connecting to iApp"),
         ("iApp connected", 1.0, "ready")],
        stdin=subprocess.DEVNULL)
    return minute, notes


def run(args) -> int:
    _load_env()
    prog = Progress()
    print("Starting meeting tools...\n", flush=True)
    problems = preflight(args, prog.show)
    prog.clear()
    if problems:
        print("Cannot start:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return 1
    if args.check:
        print("  All checks passed.")
        return 0

    stop_flag = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop_flag.set())  # the terminal also signals the children

    minute, notes = build_children(args)
    children = (minute, notes)
    q: queue.Queue = queue.Queue()
    for c in children:
        c.start(q)

    color = sys.stdout.isatty()
    phase, started, last_pct = "loading", time.monotonic(), 10
    stop_deadline = None
    notes_interrupted = False
    warned_notes = False
    exit_code = 0
    reason = ""

    def pct_now() -> int:
        return int(10 + 60 * minute.progress + 30 * notes.progress)

    while True:
        try:
            kind, ch, text = q.get(timeout=0.2)
        except queue.Empty:
            kind = None
        if kind == "data":
            ch.feed(text)
            if phase != "loading":
                emit(ch, text, color)
        elif kind == "eof":
            ch.eof = True

        if phase == "loading":
            dead = [c for c in children if c.eof]
            if dead:
                prog.clear()
                for c in dead:
                    print(f"\n{c.name} stopped before it was ready (exit code "
                          f"{c.proc.wait()}). Last output:", file=sys.stderr)
                    for line in list(c.tail)[-12:]:
                        print(f"  | {line}", file=sys.stderr)
                for c in children:
                    c.interrupt()
                phase, exit_code, reason = "stopping", 1, "startup failed"
                stop_deadline = time.monotonic() + 15
            elif stop_flag.is_set():
                prog.clear()
                print("\nCancelled.")
                phase, exit_code, reason = "stopping", 130, "cancelled"
                stop_deadline = time.monotonic() + 15
            elif all(c.progress >= 1.0 for c in children):
                prog.show(100, "ready")
                prog.clear()
                took = time.monotonic() - started
                print(f"  [{'█' * BAR}] 100%  ✅ Ready in {took:.0f}s\n")
                print("  Say \"Hey Minutes\"       talk to the assistant")
                print("  Say \"Stop Minute\"       end the chat (or press Enter)")
                print("  Say \"Finished meeting\"  end the meeting (or press Ctrl+C)")
                print("  Notes are being written live; the summary comes at the end.\n", flush=True)
                phase = "running"
            else:
                last_pct = max(last_pct, min(99, pct_now()))
                prog.show(last_pct, f"assistant: {minute.label} · notes: {notes.label}")

        elif phase == "running":
            if notes.eof and not warned_notes and not minute.eof:
                warned_notes = True
                exit_code = 1
                print("\n[main] The note-taker stopped (see above). The assistant keeps running, but "
                      "the transcript has stopped.\n", flush=True)
            if minute.eof:
                code = minute.proc.wait()
                reason = "meeting finished" if code == 0 else f"assistant stopped unexpectedly (exit {code})"
                if code != 0:
                    exit_code = 1
                print(f"\n[main] {reason}. Finishing the notes...", flush=True)
                phase = "stopping"
            elif stop_flag.is_set():
                reason = "Ctrl+C"
                print("\n[main] Ctrl+C. Finishing the notes...", flush=True)
                phase = "stopping"
            if phase == "stopping":
                stop_deadline = time.monotonic() + STOP_TIMEOUT
                # Ctrl+C already reached everything via the terminal; otherwise ask the notes politely.
                if not stop_flag.is_set() and notes.alive:
                    notes.interrupt()
                    notes_interrupted = True

        if phase == "stopping":
            if all(c.eof for c in children):
                break
            if stop_deadline and time.monotonic() > stop_deadline:
                print("\n[main] Timed out waiting for the programs to finish; stopping them.", file=sys.stderr)
                for c in children:
                    c.kill()
                exit_code = exit_code or 1
                break

    for c in children:  # collect exit codes
        if c.proc.poll() is None:
            c.kill()
    if phase == "stopping" and reason not in ("startup failed", "cancelled"):
        nc = notes.proc.returncode
        if nc not in (0, None) and not notes_interrupted and not stop_flag.is_set():
            exit_code = exit_code or 1
        folder = Path(notes.saved_to) if notes.saved_to else None
        print()
        if folder and (folder / "transcript.txt").exists():
            print(f"  Meeting saved in {folder}")
            print(f"    transcript.txt  {'✔' if (folder / 'transcript.txt').stat().st_size else '(empty)'}")
            print(f"    summary.md      {'✔' if (folder / 'summary.md').exists() else '(not created)'}")
        else:
            print("  No transcript was saved.")
    return exit_code


def main() -> int:
    p = argparse.ArgumentParser(description="Run minute.py (assistant) and openthai.py --live (notes) together.")
    p.add_argument("--check", action="store_true", help="only run the pre-flight checks")
    p.add_argument("--skip-summary", action="store_true", help="transcript only; no COMPASS_API_KEY needed")
    p.add_argument("--debug", action="store_true", help="assistant prints what its local recogniser hears")
    p.add_argument("--latency-ms", type=int, default=560, choices=[80, 320, 560, 1120])
    p.add_argument("-o", "--out-dir", default=None, help="parent folder for meeting folders (default ./meetings)")
    return run(p.parse_args())


if __name__ == "__main__":
    sys.exit(main())
