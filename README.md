# Minutes: a voice-activated meeting assistant

Minutes sits in on your meeting. It writes a **live transcript and a summary**, and when you say
**"Hello Minute"** it wakes up as a **voice assistant** you can talk to, one that can also check, book and
cancel Google Calendar appointments for you. It is controlled from a small web page on your Mac.

```
microphone ─┬─► speech model on your Mac ─► listens ONLY for "Hello Minute"
            │         └─ when woken ─► Gemini Live (via CompassLLM) ─► voice conversation + calendar tools
            └─► iApp OpenThai (live speech-to-text) ─► meetings/<date-time>/transcript.txt
                                                              └─► CompassLLM ─► summary.md
```

| File | What it is |
|---|---|
| `app.py` + `ui.html` | **Web interface** (start / end, progress, live conversation and notes, Approve / Cancel buttons) |
| `main.py` | Same thing in the terminal, with a loading bar |
| `minutes.py` | The voice assistant: wake word, conversation, calendar tools, echo guard |
| `minutes_02.py` | A small stand-alone, English-only voice agent (no tools) |
| `openthai.py` | The note-taker: live microphone or an audio file → transcript + summary |
| `gg_calendar_function.py`, `seari_calendar.py` | Google Calendar: today / create / cancel, meeting rooms |
| `rooms.example.csv`, `.env.example` | Templates for your local settings |

## Requirements

- An **Apple Silicon Mac** (the wake-word model uses `mlx-whisper`), Python 3.11 or newer (developed on 3.15).
- A **CompassLLM** API key and address (Gemini Live for the assistant, Gemini Flash for the summary).
- An **iApp** API key ([OpenThai ASR Realtime](https://iapp.co.th/docs/speech/speech-to-text/realtime)) for the live transcript.
- Optional: a Google OAuth *Desktop app* client (for the calendar), and `ffmpeg` if you want to transcribe audio *files*.

## Setup

```bash
git clone <this repository> && cd <this repository>
python3 -m venv env && source env/bin/activate
pip install -r requirements.txt        # about 1.5 GB: PyTorch comes with mlx-whisper
cp .env.example .env                   # then edit .env: COMPASS_API_KEY, COMPASS_BASE_URL, IAPP_API_KEY
python main.py --check                 # checks files, packages, keys, microphone and speaker
```

The first meeting downloads the speech model (about 1.6 GB, once). macOS will ask to let your terminal use the
**microphone**: allow it.

**Google Calendar (optional).** In Google Cloud, create an OAuth client of type *Desktop app*, download it and save it as
`credentials.json` next to the code. Then sign in once:

```bash
python gg_calendar_function.py auth     # opens a browser; sign in with the Google account to use
cp rooms.example.csv rooms.csv          # then edit it: your meeting rooms (see "Calendar and rooms" below)
```

## Run

```bash
python app.py                           # web interface: opens http://127.0.0.1:8765
python main.py                          # or the terminal version (--skip-summary, --debug, --check)
```

In the web page: **Start meeting**, wait for the loading bar, then talk. **End meeting** (or Ctrl+C in the
terminal) finishes the notes first, writes the summary, then stops. Use **headphones**: with speakers the microphone also
hears the assistant, and the transcript will contain its voice. `python app.py --port 9000 --no-browser` changes the port or
stops it opening the browser; the "Transcript only" box skips the summary.

You can also run the parts on their own:

```bash
python minutes.py                       # just the voice assistant
python minutes_02.py                    # the small stand-alone English-only voice agent
python openthai.py --live               # just the live notes; Ctrl+C to finish
python openthai.py recording.m4a        # notes from an audio file (needs ffmpeg)
```

## Talking to it

The assistant hears and speaks **English only**, and stays silent until woken.

| Say | What happens |
|---|---|
| **"Hello Minute"** (also "Hey Minute", "Hi Minute"…) | A short beep, then a voice conversation. Add your question in the same breath if you like. |
| **"Stop Minute"** (also "Goodbye Minute", "Thank you Minute"…) or the **End conversation** button | Back to waiting. |
| **"Finished meeting"** | Ends the whole meeting: the notes finish and the summary is written. |

Anything that changes something (booking or cancelling an appointment, running a command) needs your OK. In the web page
press **Approve** or **Cancel** (Esc also cancels); in the terminal type `y` and Enter. An unanswered request cancels
itself after 90 seconds.

## Calendar and rooms

Ask: "What meetings do I have today?", "Book *Sprint review* tomorrow 3 to 4 with `amy@example.com` in *Example Room*",
"Cancel the *Sprint review* tomorrow". Only meetings **you organised** can be cancelled; guests are emailed. Booking the same
appointment twice never duplicates it. The time zone defaults to Asia/Bangkok (`GCAL_TZ`).

Meeting rooms live in **`rooms.csv`** (one row per room; any spreadsheet or text editor):

```csv
name,calendar_id,location
example-room,c_xxxxxxxxxxxxxxxx@resource.calendar.google.com,"Example Tower, 5F"
```

- `name` is what you say. Case, spaces and dashes don't matter. `location` is optional.
- `calendar_id` is the **Calendar ID** in Google Calendar → the room's settings → *Integrate calendar*.
- Changes apply at once, no restart. `python gg_calendar_function.py rooms --check` asks Google whether each ID is real and
  free: run it after adding a room, it catches a mistyped ID.

The calendar functions also work without the assistant: `python gg_calendar_function.py today | create | cancel | rooms`.

## The meeting notes

Each meeting gets its own folder, `meetings/<date-time>/`, with `transcript.txt` (written live, one line per sentence) and
`summary.md` (key decisions, action items with owners, open questions; Thai with English technical terms kept). A long
transcript is summarised in chunks and then merged.

- **Live** (`--live`): streams the microphone to iApp and saves **no audio**. Ctrl+C once finishes cleanly.
- **A recording**: `python openthai.py recording.mp3`. iApp requires real-time speed, so a 1-hour file takes 1 hour.
- iApp's realtime service is **Thai only**: English words come back in Thai script (the summary step restores them where it can).
  Limits: 4 hours per session, a limited number of live sessions, and billing for the whole time connected. Check iApp's
  current pricing.
- `--backend local` uses your own OpenThai 2.0 server instead of iApp, so audio never leaves your machine. It needs a separate
  Python 3.12 environment with PyTorch + NeMo; follow the
  [model card](https://huggingface.co/iapp/OpenThai2.0-ASR-Streaming) (`hf download iapp/OpenThai2.0-ASR-Streaming`, then
  `python -m uvicorn app:app --port 18081 --host 127.0.0.1` in its `serve/` folder; the server has no authentication, keep it on
  localhost).

## Configuration

Everything is set in `.env` (see `.env.example`), or in the environment.

| Setting | Meaning |
|---|---|
| `COMPASS_API_KEY`, `COMPASS_BASE_URL` | CompassLLM key and address (`https://<host>/compass-api/v1`). **Required.** |
| `IAPP_API_KEY` | iApp key, for the live transcript. **Required** (unless `--backend local`). |
| `COMPASS_LIVE_MODEL` / `COMPASS_MODEL` | Voice model (`gemini-3.8-live`) / summary model (`gemini-3.8-flash`). |
| `COMPASS_REALTIME_URL` | Only if the realtime WebSocket is not `<COMPASS_BASE_URL>/realtime`. |
| `SEARI_STT_MODEL` | Local wake-word model (default `mlx-community/whisper-large-v3-turbo`). |
| `SEARI_SSH_HOSTS` | Hosts the assistant may run SSH commands on (empty = none). |
| `GCAL_TZ`, `GCAL_CREDENTIALS`, `GCAL_TOKEN`, `GCAL_ROOMS` | Calendar time zone, OAuth client file, saved sign-in, room list. |

`python openthai.py --help` and `python app.py --help` list the command-line options.

## Privacy and security

- **Wake-word listening is local.** Nothing is sent anywhere until you say "Hello Minute". Then your voice for that conversation goes
  to CompassLLM (Gemini). With the default backend, the **meeting audio is sent to iApp** for the live transcript, and only the
  *text* transcript goes to CompassLLM for the summary. iApp's documentation does not say how long they keep audio; check
  before using sensitive meetings, or use `--backend local`.
- **Secrets stay out of git.** `.gitignore` excludes `.env`, `credentials.json`, the saved Google sign-in (`*-token.json`), `rooms.csv`
  and the whole `meetings/` folder (your transcripts). Keep it that way.
- **The web page is local-only.** It listens on `127.0.0.1`, and every request needs a secret token that only the page itself receives,
  with matching host and origin checks, so a website you visit cannot press *Approve* for you.

## Troubleshooting

| You see | Do this |
|---|---|
| `X is not set` | Fill it in `.env`; run `python main.py --check`. |
| `certificate verify failed` | Python's own certificate list is missing; this project uses `certifi`. On a network that inspects TLS, set `SSL_CERT_FILE` to your company's CA file. |
| "Google Calendar is not signed in yet" | `python gg_calendar_function.py auth`. |
| Google says access blocked | Add your account as a test user on the OAuth consent screen, or use the organisation's account. |
| Nothing happens after `[CONFIRM]` in the terminal | Type `y` and Enter (or use the web page's buttons). |
| Port 8765 in use | `python app.py --port 9000`. |
| No microphone / silence | System Settings → Privacy & Security → Microphone → allow your terminal. |

## Known limits

- Live ASR accuracy on meetings is imperfect (about 21% character error rate on the model's meeting benchmark): check names and
  numbers in the summary. There are no speaker labels.
- It only hears *your* microphone. To capture other people on a call you need a loopback device (for example BlackHole) as the input.
- There is no automated test suite in this repository yet.
