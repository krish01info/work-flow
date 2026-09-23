"""
Run this script once to get a fresh DRIVE_REFRESH_TOKEN.
It will open a browser for you to authorize, then print the new token.
"""
from dotenv import load_dotenv
load_dotenv()

import os
from google_auth_oauthlib.flow import InstalledAppFlow

CLIENT_ID     = os.getenv("GOOGLE_CLIENT_ID")
CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")

if not CLIENT_ID or not CLIENT_SECRET:
    raise SystemExit("GOOGLE_CLIENT_ID / GOOGLE_CLIENT_SECRET missing from .env")

client_config = {
    "installed": {
        "client_id":     CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "auth_uri":      "https://accounts.google.com/o/oauth2/auth",
        "token_uri":     "https://oauth2.googleapis.com/token",
        "redirect_uris": ["urn:ietf:wg:oauth:2.0:oob", "http://localhost"],
    }
}

flow = InstalledAppFlow.from_client_config(
    client_config,
    scopes=["https://www.googleapis.com/auth/drive.file"],
)

creds = flow.run_local_server(port=0)

print("\n" + "="*60)
print("SUCCESS! Add this to your .env file:")
print("="*60)
print(f"\nDRIVE_REFRESH_TOKEN={creds.refresh_token}\n")
