"""Google Calendar for the voice assistant: today's meetings, create and cancel appointments.

Used by minutes.py (as assistant tools) and also runnable by itself:

    python gg_calendar_function.py auth                 # one-time Google sign-in (opens a browser)
    python gg_calendar_function.py rooms [--check]      # list the rooms in rooms.csv (--check asks Google)
    python gg_calendar_function.py today [--date YYYY-MM-DD]
    python gg_calendar_function.py create --title "Sprint review" \\
            --start 2026-11-26T15:00 --end 2026-11-26T16:00 [--guest a@b.com] [--room "room name"]
    python gg_calendar_function.py cancel --title "Sprint review" [--date 2026-11-26]   (or --id ID)

It acts as whichever Google account approved the token. Sign-in comes from seari_calendar.py
(same scopes as create_stonehenge.py / cancel_stonehenge.py). Settings (all optional):

    GCAL_CREDENTIALS  OAuth client JSON   (default: ./credentials.json, else ~/Downloads/client_secret_*.json)
    GCAL_TOKEN        saved sign-in       (default: ./seari-token.json, else ../work/seari-token.json)
    GCAL_TZ           time zone           (default: Asia/Bangkok)
    GCAL_ROOMS        room list CSV       (default: ./rooms.csv)

Every function below returns a plain dict and never raises: problems come back as {"error": "..."}.
"""

from __future__ import annotations

import argparse
import csv
import functools
import json
import os
import re
import sys
from datetime import date, datetime, time, timedelta
from pathlib import Path
from uuid import NAMESPACE_URL, UUID, uuid5
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
TZ = ZoneInfo(os.environ.get("GCAL_TZ", "Asia/Bangkok"))

ROOMS_FILE = Path(os.environ.get("GCAL_ROOMS", HERE / "rooms.csv"))  # name,calendar_id,location
EMAIL_RE = re.compile(r"^[^@\s,;]+@[^@\s,;]+\.[^@\s,;]+$")


class CalendarError(Exception):
    """A problem the user can act on. The message is written to be read aloud."""


# ------------------------------------------------------------------- sign-in


def credentials_path() -> Path:
    if os.environ.get("GCAL_CREDENTIALS"):
        return Path(os.environ["GCAL_CREDENTIALS"]).expanduser()
    local = HERE / "credentials.json"
    if local.exists():
        return local
    found = sorted((Path.home() / "Downloads").glob("client_secret_*.json"))
    return found[0] if found else local


def token_path() -> Path:
    if os.environ.get("GCAL_TOKEN"):
        return Path(os.environ["GCAL_TOKEN"]).expanduser()
    for p in (HERE / "seari-token.json", HERE.parent / "work" / "seari-token.json"):
        if p.exists():
            return p
    return HERE / "seari-token.json"


def get_service(interactive: bool = False):
    """The Calendar API client. Only `auth` (interactive) may open a browser; the assistant never
    pops one up in the middle of a conversation."""
    try:
        from seari_calendar import calendar_service
    except ImportError:
        raise CalendarError("The Google Calendar libraries are missing. In the terminal run: "
                            "pip install google-api-python-client google-auth-oauthlib") from None
    token, creds = token_path(), credentials_path()
    if not token.exists():
        if not interactive:
            raise CalendarError("Google Calendar is not signed in yet. In the terminal run: "
                                "python gg_calendar_function.py auth")
        if not creds.exists():
            raise CalendarError(f"Google OAuth client file not found. Put it at {HERE / 'credentials.json'} "
                                "or set GCAL_CREDENTIALS.")
    token.parent.mkdir(parents=True, exist_ok=True)
    return calendar_service(creds, token)


def _explain(e: Exception) -> str:
    try:
        from google.auth.exceptions import GoogleAuthError
        from googleapiclient.errors import HttpError
    except ImportError:
        return f"{type(e).__name__}: {e}"
    if isinstance(e, HttpError):
        try:
            detail = json.loads(e.content.decode())["error"]["message"]
        except Exception:
            detail = e.reason if hasattr(e, "reason") else str(e)
        status = e.resp.status
        hint = {401: "Google sign-in expired; run: python gg_calendar_function.py auth. ",
                403: "Google refused (permission or scope). ",
                429: "Google is rate-limiting; try again shortly. "}.get(status, "")
        return f"{hint}Google Calendar error {status}: {detail}"
    if isinstance(e, GoogleAuthError):
        return ("Google sign-in failed or expired. Delete the saved token file and run: "
                "python gg_calendar_function.py auth")
    if isinstance(e, (ValueError, OSError)):
        return str(e)
    return f"{type(e).__name__}: {e}"


def _guard(fn):
    @functools.wraps(fn)
    def wrapper(*a, **k):
        try:
            return fn(*a, **k)
        except CalendarError as e:
            return {"error": str(e)}
        except Exception as e:  # network, Google, bad input: report, never crash the assistant
            return {"error": _explain(e)}
    return wrapper


# -------------------------------------------------------------------- helpers


def parse_when(text: str) -> datetime:
    """ISO 8601 date-time. No offset means the configured time zone (Bangkok by default)."""
    try:
        d = datetime.fromisoformat(str(text).strip())
    except ValueError:
        raise CalendarError(f"I cannot read the time '{text}'. Use a form like 2026-11-26T15:00.") from None
    return d if d.tzinfo else d.replace(tzinfo=TZ)


def resolve_day(text: str | None) -> date:
    t = (text or "today").strip().lower()
    today = datetime.now(TZ).date()
    named = {"": 0, "today": 0, "tomorrow": 1, "yesterday": -1}
    if t in named:
        return today + timedelta(days=named[t])
    try:
        return date.fromisoformat(t)
    except ValueError:
        raise CalendarError(f"I cannot read the date '{text}'. Use today, tomorrow or YYYY-MM-DD.") from None


def day_bounds(d: date) -> tuple[datetime, datetime]:
    start = datetime.combine(d, time.min, TZ)
    return start, start + timedelta(days=1)


def _emails(text) -> list[str]:
    items = [t for t in re.split(r"[,\s;]+", text if isinstance(text, str) else " ".join(text or ())) if t]
    bad = [t for t in items if not EMAIL_RE.match(t)]
    if bad:
        raise CalendarError(f"These do not look like email addresses: {', '.join(bad)}.")
    return list(dict.fromkeys(items))


def _norm(name: str) -> str:
    """'Meeting Room-2', 'meeting room 2' and 'meeting-room-2' all look the same."""
    return re.sub(r"[^a-z0-9]+", " ", name.casefold()).strip()


def load_rooms(path: Path | None = None) -> dict[str, tuple[str, str, str]]:
    """Read the room list (rooms.csv: name,calendar_id,location) -> {normalised name: (calendar_id, location, name)}.
    It is read on every call, so adding a row takes effect immediately, with no restart."""
    path = path or ROOMS_FILE
    if not path.is_file():
        raise CalendarError(f"The room list {path.name} was not found. Copy rooms.example.csv to {path.name} and edit it "
                            "(columns: name,calendar_id,location).")
    lines = [(no, row) for no, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1)
             if line.strip() and not line.lstrip().startswith("#")
             for row in csv.reader([line], skipinitialspace=True)]
    if not lines:
        return {}
    header = [h.strip().casefold() for h in lines[0][1]]
    if "name" not in header or "calendar_id" not in header:
        raise CalendarError(f"{path.name} line {lines[0][0]}: the first row must be a header with the columns "
                            "name,calendar_id,location.")
    col = {h: i for i, h in enumerate(header)}
    rooms: dict[str, tuple[str, str, str]] = {}
    for no, row in lines[1:]:
        get = lambda c: row[col[c]].strip() if c in col and col[c] < len(row) else ""
        name, cal, loc = get("name"), get("calendar_id"), get("location")
        if not name or not _norm(name):
            raise CalendarError(f"{path.name} line {no}: the room needs a name.")
        if not EMAIL_RE.match(cal):
            raise CalendarError(f"{path.name} line {no}: '{name}' needs a calendar_id such as "
                                "c_xxxxxxxx@resource.calendar.google.com.")
        if _norm(name) in rooms:
            raise CalendarError(f"{path.name} line {no}: the room name '{name}' is used twice.")
        rooms[_norm(name)] = (cal, loc, name)
    return rooms


def room_names() -> list[str]:
    """The names in the room list, for telling the voice agent which rooms exist. Never raises."""
    try:
        return [r[2] for r in load_rooms().values()]
    except Exception:
        return []


def _room(name: str) -> tuple[str | None, str | None]:
    """A room name from rooms.csv (or a room's calendar address) -> (calendar_id, location)."""
    name = (name or "").strip()
    if not name:
        return None, None
    if EMAIL_RE.match(name):
        return name, None
    rooms = load_rooms()
    if _norm(name) in rooms:
        cal, loc, _ = rooms[_norm(name)]
        return cal, loc or None
    known = ", ".join(r[2] for r in rooms.values()) or "none yet"
    raise CalendarError(f"I do not know the room '{name}'. Rooms I know: {known}. "
                        f"Add more to {ROOMS_FILE.name}, or give the room's calendar address.")


def _local(iso: str) -> str:
    return datetime.fromisoformat(iso).astimezone(TZ).isoformat(timespec="minutes")


def _is_mine(e: dict) -> bool:
    return bool(e.get("organizer", {}).get("self") or e.get("creator", {}).get("self"))


def _my_response(e: dict) -> str | None:
    return next((a.get("responseStatus") for a in e.get("attendees", []) if a.get("self")), None)


def _summary(e: dict) -> dict:
    s, en = e.get("start", {}), e.get("end", {})
    all_day = "dateTime" not in s
    out = {
        "event_id": e["id"],
        "title": e.get("summary") or "(no title)",
        "all_day": all_day,
        "start": s.get("date") if all_day else _local(s["dateTime"]),
        "end": en.get("date") if all_day else _local(en["dateTime"]),
        "time": "all day" if all_day else
                f"{datetime.fromisoformat(s['dateTime']).astimezone(TZ):%H:%M}-"
                f"{datetime.fromisoformat(en['dateTime']).astimezone(TZ):%H:%M}",
        "organized_by_me": _is_mine(e),
        "my_response": _my_response(e),
        "guests": sum(1 for a in e.get("attendees", []) if not a.get("self") and not a.get("resource")),
    }
    for key, src in (("location", e.get("location")), ("video_link", e.get("hangoutLink")),
                     ("link", e.get("htmlLink"))):
        if src:
            out[key] = src
    return out


def _get_event(service, event_id: str) -> dict | None:
    from googleapiclient.errors import HttpError
    try:
        return service.events().get(calendarId="primary", eventId=event_id).execute()
    except HttpError as e:
        if e.resp.status != 404:
            raise
        return None


def _list_range(service, start: datetime, end: datetime) -> list[dict]:
    """Events overlapping [start, end) on the primary calendar, minus cancelled and declined ones."""
    events, token = [], None
    while True:
        kw = dict(calendarId="primary", timeMin=start.isoformat(), timeMax=end.isoformat(),
                  singleEvents=True, orderBy="startTime", maxResults=100)
        if token:
            kw["pageToken"] = token
        res = service.events().list(**kw).execute()
        events += res.get("items", [])
        token = res.get("nextPageToken")
        if not token:
            break
    return [e for e in events if e.get("status") != "cancelled" and _my_response(e) != "declined"]


# ------------------------------------------------------------- check meetings


@_guard
def today(day: str = "today", *, service=None) -> dict:
    """The meetings on one day (default: today) from the primary calendar."""
    d = resolve_day(day)
    service = service or get_service()
    events = [_summary(e) for e in _list_range(service, *day_bounds(d))]
    return {"date": d.isoformat(), "weekday": d.strftime("%A"), "timezone": str(TZ),
            "count": len(events), "events": events}


# ------------------------------------------------------------ create appointment


def _request_id(title: str, start: datetime, end: datetime, room: str | None, guests: list[str]) -> UUID:
    """Same appointment -> same ID, so asking twice (or a retry) never books it twice."""
    key = f"seari/{room or '-'}/{start.isoformat()}/{end.isoformat()}/{title.casefold()}/" \
          f"{','.join(sorted(g.casefold() for g in guests))}"
    return uuid5(NAMESPACE_URL, key)


@_guard
def create(title: str, start: str, end: str, guests: str = "", room: str = "", location: str = "",
           description: str = "", *, service=None) -> dict:
    """Create an appointment (optionally with guests and/or a meeting room)."""
    from googleapiclient.errors import HttpError

    title = (title or "").strip()
    if not title:
        raise CalendarError("The appointment needs a title.")
    s, e = parse_when(start), parse_when(end)
    if e <= s:
        raise CalendarError("The end time must be after the start time.")
    guest_list = _emails(guests)
    room_email, room_location = _room(room)
    service = service or get_service()

    rid = _request_id(title, s, e, room_email, guest_list)
    event = _get_event(service, rid.hex)
    while event and event.get("status") == "cancelled":  # a deleted event keeps its ID; use the next
        rid = uuid5(rid, "rebook")
        event = _get_event(service, rid.hex)

    created, overlaps = event is None, []
    if created:
        if room_email:
            busy = service.freebusy().query(body={
                "timeMin": s.isoformat(), "timeMax": e.isoformat(), "items": [{"id": room_email}]}).execute()
            info = busy.get("calendars", {}).get(room_email, {})
            if info.get("errors") or "busy" not in info:
                raise CalendarError("I cannot check that room. Verify its email and your company access.")
            if info["busy"]:
                raise CalendarError("That room is busy at the requested time.")
        overlaps = [ev.get("summary") or "(no title)" for ev in _list_range(service, s, e)]

        attendees = ([{"email": room_email, "resource": True}] if room_email else [])
        attendees += [{"email": g} for g in guest_list if g.casefold() != (room_email or "").casefold()]
        body = {"id": rid.hex, "summary": title,
                "start": {"dateTime": s.isoformat()}, "end": {"dateTime": e.isoformat()}}
        if attendees:
            body["attendees"] = attendees
        if location or room_location:
            body["location"] = location or room_location
        if description:
            body["description"] = description
        try:
            event = service.events().insert(calendarId="primary", body=body,
                                            sendUpdates="all" if attendees else "none").execute()
        except HttpError as err:
            if err.resp.status != 409:  # 409 = it was created a moment ago (retry); read it back
                raise
            event = _get_event(service, rid.hex)
            if event is None:
                raise

    out = {"created": created, **_summary(event)}
    if not created:
        out["note"] = "This exact appointment already existed, so nothing was duplicated."
    if overlaps:
        out["overlaps_with"] = overlaps
    if room_email:
        out["room_response"] = next((a.get("responseStatus", "needsAction") for a in event.get("attendees", [])
                                     if a.get("email", "").casefold() == room_email.casefold()), "needsAction")
    return out


# ------------------------------------------------------------ cancel appointment


@_guard
def find_to_cancel(event_id: str = "", title: str = "", day: str = "today", *, service=None) -> dict:
    """Identify exactly one appointment to cancel, without changing anything (so the caller can
    ask the user to confirm). Only appointments you organise can be cancelled."""
    service = service or get_service()
    if event_id:
        event = _get_event(service, event_id.strip())
        if event is None or event.get("status") == "cancelled":
            raise CalendarError("I cannot find that appointment; it may already be cancelled.")
    elif title.strip():
        d = resolve_day(day)
        wanted = title.strip().casefold()
        same_day = _list_range(service, *day_bounds(d))
        hits = [e for e in same_day if wanted in (e.get("summary") or "").casefold()]
        if not hits:
            titles = ", ".join(f"'{e.get('summary') or '(no title)'}'" for e in same_day) or "none"
            raise CalendarError(f"No meeting matching '{title}' on {d.isoformat()}. Meetings that day: {titles}.")
        if len(hits) > 1:
            options = "; ".join(f"'{e.get('summary')}' at {_summary(e)['time']}" for e in hits)
            raise CalendarError(f"Several meetings match '{title}': {options}. Which one?")
        event = hits[0]
    else:
        raise CalendarError("Tell me the meeting's title (and date), or its event id.")

    if not _is_mine(event):
        raise CalendarError(f"You are not the organiser of '{event.get('summary')}', so cancelling it would only "
                            "remove it from your own calendar. Decline it in Google Calendar instead.")
    s = _summary(event)
    return {k: s[k] for k in ("event_id", "title", "start", "end", "time", "guests") if k in s}


@_guard
def cancel(event_id: str, *, service=None) -> dict:
    """Cancel (delete) one appointment you organise; guests are notified."""
    from googleapiclient.errors import HttpError

    service = service or get_service()
    event = _get_event(service, event_id)
    if event is None or event.get("status") == "cancelled":
        return {"cancelled": False, "reason": "It was already cancelled or does not exist."}
    if not _is_mine(event):
        raise CalendarError("You are not the organiser of that appointment, so I will not delete it.")
    notify = "all" if any(not a.get("self") for a in event.get("attendees", [])) else "none"
    try:
        service.events().delete(calendarId="primary", eventId=event["id"], sendUpdates=notify).execute()
    except HttpError as err:
        if err.resp.status not in (404, 410):
            raise
        return {"cancelled": False, "reason": "It was already cancelled."}
    s = _summary(event)
    return {"cancelled": True, "title": s["title"], "start": s["start"], "end": s["end"],
            "guests_notified": notify == "all"}


# ---------------------------------------------------------------------- rooms


@_guard
def rooms(check: bool = False, *, service=None) -> dict:
    """The rooms in rooms.csv. With check=True, ask Google whether each one can really be reached
    (catches a mistyped calendar_id) and whether it is free for the next hour."""
    listing = load_rooms()
    out = [{"name": n, "calendar_id": c, "location": loc} for c, loc, n in listing.values()]
    if check and out:
        service = service or get_service()
        start = datetime.now(TZ)
        for r in out:
            res = service.freebusy().query(body={"timeMin": start.isoformat(),
                                                 "timeMax": (start + timedelta(hours=1)).isoformat(),
                                                 "items": [{"id": r["calendar_id"]}]}).execute()
            info = res.get("calendars", {}).get(r["calendar_id"], {})
            if info.get("errors") or "busy" not in info:
                why = (info.get("errors") or [{}])[0].get("reason", "unknown")
                r["status"] = f"CANNOT BE REACHED ({why}): check the calendar_id and your access"
            else:
                r["status"] = "busy now" if info["busy"] else "free for the next hour"
    return {"file": str(ROOMS_FILE), "count": len(out), "rooms": out}


# -------------------------------------------------------------------------- CLI


def main() -> int:
    p = argparse.ArgumentParser(description="Google Calendar: today / create / cancel.")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("auth", help="one-time Google sign-in (opens a browser)")
    r = sub.add_parser("rooms", help="list the rooms in rooms.csv")
    r.add_argument("--check", action="store_true", help="ask Google whether each room can be reached and is free")
    t = sub.add_parser("today", help="list a day's meetings")
    t.add_argument("--date", default="today", help="today, tomorrow or YYYY-MM-DD")
    c = sub.add_parser("create", help="create an appointment")
    c.add_argument("--title", required=True)
    c.add_argument("--start", required=True, help="e.g. 2026-11-26T15:00 (Bangkok time unless an offset is given)")
    c.add_argument("--end", required=True)
    c.add_argument("--guest", action="append", default=[], help="guest email; repeat as needed")
    c.add_argument("--room", default="", help="a room name from rooms.csv, or a room's calendar address")
    c.add_argument("--location", default="")
    c.add_argument("--description", default="")
    x = sub.add_parser("cancel", help="cancel an appointment you organise")
    x.add_argument("--id", default="", help="event id (from `today`)")
    x.add_argument("--title", default="")
    x.add_argument("--date", default="today")
    a = p.parse_args()

    if a.cmd == "auth":
        try:
            get_service(interactive=True)
        except Exception as e:
            print(f"Error: {e if isinstance(e, CalendarError) else _explain(e)}", file=sys.stderr)
            return 1
        print(f"Signed in. Token saved to {token_path()}")
        return 0
    if a.cmd == "rooms":
        result = rooms(a.check)
    elif a.cmd == "today":
        result = today(a.date)
    elif a.cmd == "create":
        result = create(a.title, a.start, a.end, ", ".join(a.guest), a.room, a.location, a.description)
    else:
        found = find_to_cancel(a.id, a.title, a.date)
        result = found if "error" in found else cancel(found["event_id"])
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 1 if "error" in result else 0


if __name__ == "__main__":
    sys.exit(main())
