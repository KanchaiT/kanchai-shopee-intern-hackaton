import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from uuid import UUID, uuid4

from google.auth.exceptions import GoogleAuthError
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


SCOPES = [
    "https://www.googleapis.com/auth/calendar.events.owned",
    "https://www.googleapis.com/auth/calendar.events.freebusy",
]


def calendar_service(credentials_path, token_path):
    credentials = None
    if token_path.exists():
        credentials = Credentials.from_authorized_user_file(str(token_path), SCOPES)

    if credentials is None:
        flow = InstalledAppFlow.from_client_secrets_file(str(credentials_path), SCOPES)
        credentials = flow.run_local_server(port=0)
        token_path.write_text(credentials.to_json(), encoding="utf-8")
    elif not credentials.valid:
        credentials.refresh(Request())
        token_path.write_text(credentials.to_json(), encoding="utf-8")

    return build("calendar", "v3", credentials=credentials)


def get_event(service, event_id):
    try:
        return service.events().get(calendarId="primary", eventId=event_id).execute()
    except HttpError as error:
        if error.resp.status != 404:
            raise
        return None


def book_room(
    service, *, title, start, end, room_email, request_id, location=None, guests=()
):
    """Request a room. Reuse the same UUID to retry or read its latest response."""
    if start.utcoffset() is None or end.utcoffset() is None:
        raise ValueError("Include a time zone offset in both times, such as +07:00.")
    if end <= start:
        raise ValueError("End time must be after start time.")

    event_id = request_id.hex
    event = get_event(service, event_id)
    if event is None:
        availability = service.freebusy().query(body={
            "timeMin": start.isoformat(),
            "timeMax": end.isoformat(),
            "items": [{"id": room_email}],
        }).execute()
        room = availability.get("calendars", {}).get(room_email, {})
        if room.get("errors") or "busy" not in room:
            raise ValueError("Cannot check the room. Verify its email and your company access.")
        if room["busy"]:
            raise ValueError("The room is busy at the requested time.")

        attendees = [{"email": room_email, "resource": True}]
        attendees.extend(
            {"email": email}
            for email in dict.fromkeys(guests)
            if email.casefold() != room_email.casefold()
        )
        event_body = {
            "id": event_id,
            "summary": title,
            "start": {"dateTime": start.isoformat()},
            "end": {"dateTime": end.isoformat()},
            "attendees": attendees,
        }
        if location:
            event_body["location"] = location

        try:
            event = service.events().insert(
                calendarId="primary", body=event_body, sendUpdates="all"
            ).execute()
        except HttpError as error:
            if error.resp.status != 409:
                raise
            event = get_event(service, event_id)
            if event is None:
                raise

    room_status = "needsAction"
    for attendee in event.get("attendees", []):
        if attendee.get("email", "").casefold() == room_email.casefold():
            room_status = attendee.get("responseStatus", "needsAction")
            break

    return {
        "event_id": event["id"],
        "event_url": event.get("htmlLink"),
        "room_email": room_email,
        "room_response": room_status,
        "room_confirmed": room_status == "accepted" and event.get("status") != "cancelled",
    }


def main():
    parser = argparse.ArgumentParser(description="Request a Google Workspace meeting room.")
    parser.add_argument("--title", required=True)
    parser.add_argument("--start", type=datetime.fromisoformat, required=True)
    parser.add_argument("--end", type=datetime.fromisoformat, required=True)
    parser.add_argument("--room", required=True, help="Google Workspace room resource email")
    parser.add_argument("--location", help="Room name to display in the event")
    parser.add_argument("--guest", action="append", default=[], help="Guest email. Repeat as needed.")
    parser.add_argument("--request-id", type=UUID, help="Reuse the printed UUID to check or retry.")
    parser.add_argument("--credentials", type=Path, default=Path("credentials.json"))
    parser.add_argument("--token", type=Path, default=Path("seari-token.json"))
    args = parser.parse_args()

    request_id = args.request_id or uuid4()
    print(f"Request ID: {request_id}", file=sys.stderr, flush=True)
    try:
        service = calendar_service(args.credentials, args.token)
        result = book_room(
            service,
            title=args.title,
            start=args.start,
            end=args.end,
            room_email=args.room,
            request_id=request_id,
            location=args.location,
            guests=args.guest,
        )
    except (GoogleAuthError, HttpError, OSError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
