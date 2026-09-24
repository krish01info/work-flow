# Tech Stack — Reel Builder

> A breakdown of every technology used in `reelbuilder.py`, what role it plays, and how it fits into the pipeline.

---

## Overview

| Layer | Technology | Role |
|---|---|---|
| Orchestration | LangGraph | Runs the pipeline as a state machine |
| Video Processing | FFmpeg | Encodes, scales, crops, and muxes video |
| Social API | Meta Graph API | Fetches trending audio, creates & publishes Reels |
| Cloud Storage | Google Drive API | Hosts the final video at a public URL |
| Runtime | Python 3.11 | Core language |
| Environment | python-dotenv | Loads secrets from `.env` |
| HTTP | requests | All API calls to Meta |
| CI/CD | GitHub Actions | Automates the full pipeline on a schedule |

---

## 1. LangGraph

**What it is:**  
A graph-based orchestration framework built on top of LangChain. It lets you define a pipeline as a directed graph of nodes, where each node is a Python function that reads and writes to a shared state object.

**How we use it:**

```python
graph = StateGraph(ReelState)
graph.add_node("find_trending_audio", find_trending_audio)
graph.add_node("generate_caption", generate_caption)
# ... more nodes
graph.set_entry_point("find_trending_audio")
graph.add_edge("find_trending_audio", "generate_caption")
# ...
app = graph.compile()
app.invoke(initial_state)
```

**Why LangGraph instead of plain Python functions?**
- Each node is isolated — a failure in one node doesn't silently corrupt others
- State is typed (`ReelState` TypedDict) so every field is explicit
- Adding, removing, or reordering pipeline steps is a one-line change
- Designed for agentic workflows that may later use LLMs or branching logic

**Package:** `langgraph`

---

## 2. FFmpeg

**What it is:**  
The industry-standard open-source tool for video and audio processing. Called as a subprocess from Python.

**How we use it — 5 distinct passes:**

### Pass 1 — Normalize (scale + crop)
```
ffmpeg -i asset.mp4
  -vf "scale=1080:1920:force_original_aspect_ratio=increase,
       crop=1080:1920,setsar=1"
  -r 30 -c:v libx264 -preset medium -pix_fmt yuv420p
  -c:a aac -b:a 192k
  normalized.mp4
```
Scales source video to 1080×1920 (9:16) by scaling up and center-cropping — no black bars.

### Pass 2 — Extract loop segment (video only)
```
ffmpeg -ss {loop_start} -i normalized.mp4
  -t {loop_duration} -an -c:v libx264
  loop_segment.mp4
```
Cuts the tail of the asset to use as a seamless loop filler.

### Pass 3 — Concatenate
```
ffmpeg -f concat -safe 0 -i concat.txt -c copy concatenated.mp4
```
Joins the full asset + repeated loop segments into one continuous video.

### Pass 4 — Audio extraction + volume boost
```
ffmpeg -i normalized.mp4 -t 5.0
  -vn -af "volume=1.8" -acodec aac -b:a 192k
  asset_audio.aac
```
Extracts only the first 5 seconds of the original clip audio, boosted 1.8×.

### Pass 5 — Final mux
```
ffmpeg -i concatenated.mp4 -i mixed_audio.aac
  -t {final_duration}
  -vf "scale=1080:1920:force_original_aspect_ratio=increase,
       crop=1080:1920,setsar=1"
  -c:v libx264 -pix_fmt yuv420p -r 30
  -c:a aac -b:a 192k -shortest
  final_reel.mp4
```
Combines video + audio into the final output with a second scale+crop pass to guarantee exact dimensions.

**Why subprocess instead of a Python FFmpeg wrapper?**  
Full control over every flag. Python wrappers like `moviepy` abstract too much and don't expose the exact filters needed for reliable Instagram-spec output.

---

## 3. Meta Graph API

**What it is:**  
Facebook/Instagram's official REST API for programmatic content management.

**Endpoints we use:**

| Endpoint | Method | Purpose |
|---|---|---|
| `/ig_audio` | `GET` | Fetch trending catalog music tracks |
| `/ig_hashtag_search` | `GET` | Search for a hashtag by name → get its ID |
| `/{hashtag_id}` | `GET` | Get `media_count` for a hashtag (trending signal) |
| `/{ig_user_id}/media` | `POST` | Create a Reel media container |
| `/{container_id}` | `GET` | Poll processing status |
| `/{ig_user_id}/media_publish` | `POST` | Publish the container to Instagram |

**Authentication:**  
Long-lived access token passed as `access_token` param on every request. Stored in `META_ACCESS_TOKEN` env var.

**API version:**  
Configurable via `META_GRAPH_VERSION` env var (default `v24.0`).

**Package:** `requests` (raw HTTP, no Meta SDK)

---

## 4. Google Drive API

**What it is:**  
Google's cloud storage API. Used here solely to host the final `final_reel.mp4` at a public URL — because Meta requires a publicly reachable video URL to create an Instagram container.

**How we use it:**

```python
# Auth: OAuth2 with refresh token (no user interaction needed)
creds = Credentials(
    refresh_token=DRIVE_REFRESH_TOKEN,
    client_id=GOOGLE_CLIENT_ID,
    client_secret=GOOGLE_CLIENT_SECRET,
    ...
)
creds.refresh(GoogleRequest())   # auto-refresh access token

# Upload
service = gdrive_build("drive", "v3", credentials=creds)
service.files().create(body={"name": "final_reel.mp4"}, media_body=...).execute()

# Make public
service.permissions().create(fileId=file_id, body={"type": "anyone", "role": "reader"}).execute()

# Public URL
url = f"https://drive.google.com/uc?export=download&id={file_id}"
```

**Why Google Drive?**  
- Free, no hosting setup needed
- Supports resumable uploads (large video files)
- Direct download URLs work without authentication for files under ~100MB

**Packages:** `google-auth`, `google-auth-httplib2`, `google-api-python-client`

---

## 5. Python 3.11

**Core language.** Key standard library modules used:

| Module | Usage |
|---|---|
| `os` | Reading environment variables |
| `sys` | Exit codes |
| `subprocess` | Running FFmpeg commands |
| `pathlib.Path` | Cross-platform file path handling |
| `json` | Parsing and logging Meta API responses |
| `math` | Calculating loop repeat counts (`math.ceil`) |
| `shutil` | Cleaning up temp directory after build |
| `time` | Polling sleep between container status checks |
| `re` | Stripping `(feat. ...)` from audio titles in caption generation |
| `typing` | `TypedDict`, `Optional`, `Dict`, `Any` for type safety |

---

## 6. python-dotenv

**What it is:**  
Loads key=value pairs from a `.env` file into `os.environ` at startup.

**How we use it:**
```python
from dotenv import load_dotenv
load_dotenv()   # runs once at module level

ASSET_VIDEO = os.getenv("ASSET_VIDEO", "asset.mp4")
```

Keeps all secrets and config out of code. In GitHub Actions, the same variables are injected via the `env:` block in the workflow.

**Package:** `python-dotenv`

---

## 7. requests

**What it is:**  
The standard Python HTTP library for making REST API calls.

**How we use it:**

```python
# All Meta API calls go through two thin wrappers:
def graph_get(path, params):   # wraps requests.get
def graph_post(path, data):    # wraps requests.post

# Also used for downloading catalog audio from Meta CDN:
response = requests.get(download_url, stream=True, timeout=120)
```

Every call has a 60s timeout. Errors raise `RuntimeError` with the full JSON response logged.

**Package:** `requests`

---

## 8. GitHub Actions

**What it is:**  
GitHub's built-in CI/CD platform. Used here as a free cron scheduler to run the pipeline automatically.

**How we use it:**

```yaml
on:
  schedule:
    - cron: '0 */4 * * *'   # every 4 hours = 6 posts/day
  workflow_dispatch: {}      # manual trigger from Actions tab
```

**What the workflow does (in order):**

1. `actions/checkout@v4` — checks out the repo (includes asset video)
2. `actions/setup-python@v5` — sets up Python 3.11 with pip cache
3. `apt-get install ffmpeg` — installs FFmpeg on the Ubuntu runner
4. `pip install -r requirements.txt` — installs Python dependencies
5. Asset existence check — fails fast if the video file is missing
6. `python -m py_compile reelbuilder.py` — syntax check before running
7. `python -u reelbuilder.py` — runs the full pipeline
8. `actions/upload-artifact@v4` — saves `final_reel.mp4` for 7 days (debug)

**Secrets used:**

| Secret | Used for |
|---|---|
| `INSTAGRAM_USER_ID` | Meta API identity |
| `META_ACCESS_TOKEN` | Meta API authentication |
| `GOOGLE_CLIENT_ID` | Google Drive OAuth2 |
| `GOOGLE_CLIENT_SECRET` | Google Drive OAuth2 |
| `DRIVE_REFRESH_TOKEN` | Google Drive token refresh |
| `INSTAGRAM_CAPTION` | Fallback caption if API fails |

---

## Dependency Summary

```
langgraph                  # pipeline orchestration
requests                   # HTTP / Meta API calls
python-dotenv              # .env loading
google-auth                # Google OAuth2
google-auth-httplib2       # Google auth transport
google-api-python-client   # Google Drive API client
```

> FFmpeg is a **system dependency** (not a Python package) — installed via `apt-get` in CI or manually on local machines.
