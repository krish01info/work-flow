"""Helper to discover your INSTAGRAM_ACCOUNT_ID from your INSTAGRAM_ACCESS_TOKEN.

Run:
    python get_instagram_id.py
"""

import os
import sys
import requests
from dotenv import load_dotenv

load_dotenv()

token = os.getenv("INSTAGRAM_ACCESS_TOKEN")
if not token:
    token = input("Enter your Meta / Instagram Access Token: ").strip()

if not token:
    print("Error: No access token provided.")
    sys.exit(1)

print("\nQuerying Meta Graph API for connected Instagram Business / Creator accounts...\n")
url = "https://graph.facebook.com/v20.0/me/accounts"
params = {
    "fields": "id,name,instagram_business_account{id,username}",
    "access_token": token,
}

try:
    resp = requests.get(url, params=params, timeout=20)
    data = resp.json()

    if "error" in data:
        print(f"Meta API Error: {data['error'].get('message')}")
        sys.exit(1)

    pages = data.get("data", [])
    if not pages:
        print("No Facebook Pages found associated with this access token.")
        print("Ensure your token has 'pages_show_list' and 'instagram_basic' permissions.")
        sys.exit(1)

    found = False
    for p in pages:
        page_name = p.get("name")
        page_id   = p.get("id")
        ig_acc    = p.get("instagram_business_account")

        print(f"Facebook Page: {page_name} (ID: {page_id})")
        if ig_acc:
            ig_id = ig_acc.get("id")
            ig_username = ig_acc.get("username", "unknown")
            print(f"  -> Connected Instagram Account: @{ig_username}")
            print(f"  -> INSTAGRAM_ACCOUNT_ID = {ig_id}\n")
            found = True
        else:
            print("  -> No Instagram Professional account connected to this Page.\n")

    if found:
        print("SUCCESS! Copy the INSTAGRAM_ACCOUNT_ID and INSTAGRAM_ACCESS_TOKEN to your .env file and GitHub Secrets.")
    else:
        print("Ensure your Instagram account is switched to Professional (Creator or Business) and linked to your Facebook Page.")

except Exception as e:
    print(f"Request failed: {e}")
