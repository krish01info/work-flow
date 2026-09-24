# Reel Builder — Implementation Guide

> **File:** `reelbuilder.py`  
> **Purpose:** Fully automated pipeline to build, process, and publish an Instagram Reel from a local asset video using LangGraph, FFmpeg, Meta Graph API, and Google Drive.

---

## Table of Contents

1. [Overview](#overview)
2. [Pipeline Architecture](#pipeline-architecture)
3. [Configuration & Environment Variables](#configuration--environment-variables)
4. [Node-by-Node Breakdown](#node-by-node-breakdown)
5. [FFmpeg Video Processing](#ffmpeg-video-processing)
6. [Audio Strategy](#audio-strategy)
7. [Graph Wiring](#graph-wiring)
8. [Error Handling](#error-handling)
9. [GitHub Actions Integration](#github-actions-integration)

---

## Overview

`reelbuilder.py` is a **LangGraph state machine** that runs 8 sequential nodes. Each node does one specific job and passes state forward to the next.

```
Asset Video (.mp4)
       │
       ▼
[ LangGraph Pipeline ]
       │
       ▼
final_reel.mp4  ──►  Google Drive  ──►  Instagram (published)
```

---

## Pipeline Architecture

```
find_trending_audio
    → generate_caption
        → calculate_video_duration
            → build_video
                → upload_to_drive
                    → check_public_video_url
                        → create_reel_container
                            → wait_for_container
                                → publish_reel
```

### State Object (`ReelState`)

All nodes read from and write to a shared `TypedDict` state:

| Field | Type | Description |
|---|---|---|
| `asset_video` | `str` | Path to the input asset video |
| `output_video` | `str` | Path to the final output MP4 |
| `asset_duration` | `float` | Duration of the asset video in seconds |
| `final_duration` | `float` | Target duration of the reel (capped at 90s) |
| `audio_id` | `str` | Meta catalog audio ID |
| `audio_title` | `str` | Title of the selected trending audio |
| `audio_metadata` | `dict` | Full Meta API response for the audio track |
| `audio_duration` | `float` | Duration of the catalog track in seconds |
| `caption` | `str` | Auto-generated Instagram caption |
| `video_public_url` | `str` | Public URL of the video (from Google Drive) |
| `container_id` | `str` | Instagram media container ID |
| `media_id` | `str` | Published Instagram media ID |
| `status` | `str` | Final pipeline status |

---

## Configuration & Environment Variables

All config is loaded from `.env` via `python-dotenv`. Every value has a sensible default.

### Required Secrets

| Variable | Description |
|---|---|
| `INSTAGRAM_USER_ID` | Your Instagram Business/Creator account ID |
| `META_ACCESS_TOKEN` | Meta Graph API long-lived access token |
| `GOOGLE_CLIENT_ID` | Google OAuth2 client ID (for Drive upload) |
| `GOOGLE_CLIENT_SECRET` | Google OAuth2 client secret |
| `DRIVE_REFRESH_TOKEN` | Google Drive OAuth2 refresh token |

### Optional / Tunable

| Variable | Default | Description |
|---|---|---|
| `ASSET_VIDEO` | `asset.mp4` | Path to your source video |
| `OUTPUT_VIDEO` | `final_reel.mp4` | Path for the output reel |
| `VIDEO_WIDTH` | `1080` | Output width in pixels |
| `VIDEO_HEIGHT` | `1920` | Output height in pixels (9:16 portrait) |
| `VIDEO_FPS` | `30` | Output frames per second |
| `MUSIC_START_SECONDS` | `5.0` | When catalog music takes over (seconds) |
| `LOOP_START_SECONDS` | `9.5` | Start of the loop segment in the asset |
| `REEL_MAX_DURATION` | `90` | Instagram Reel cap in seconds |
| `ASSET_AUDIO_BOOST` | `1.8` | Volume multiplier for the first 5s asset audio |
| `PUBLISH_REEL` | `false` | Set `true` to actually publish to Instagram |
| `SHARE_TO_FEED` | `false` | Also share the reel to the main feed |
| `INSTAGRAM_CAPTION` | `""` | Fallback caption if auto-generation fails |
| `AUDIO_LIMIT` | `20` | Number of trending tracks to fetch from Meta |

---

## Node-by-Node Breakdown

### Node 1 — `find_trending_audio`

**What it does:**
Calls `GET /ig_audio` on the Meta Graph API to fetch trending catalog music tracks. Picks the most trending track that has a `download_url`. Falls back to the first track if none have a URL.

**Key logic:**
- Parses `duration_in_ms` or `duration_ms` or `duration` from the track (Meta uses different field names across API versions).
- Logs the trending rank (e.g. `#1 of 20`).

**Output fields:** `audio_id`, `audio_title`, `audio_metadata`, `audio_duration`

---

### Node 2 — `generate_caption`

**What it does:**
Generates a short, trending caption — max **6–7 words** in the hook line, followed by hashtags.

**Steps:**
1. Builds a hook from the audio title:
   `"Blinding Lights"` → `"🎵 Blinding Lights on repeat ✨"`
   - Strips `(feat. ...)` / `[prod. ...]` suffixes with regex
   - Keeps max 4 title words so the total hook stays ≤ 7 words

2. Fetches trending hashtag IDs from `GET /ig_hashtag_search` for seed terms:
   `songs, music, viral, reels, trending, fyp, explorepage, newmusic`

3. Fetches `media_count` for each hashtag from `GET /{hashtag_id}?fields=name,media_count`

4. Sorts by `media_count` descending, takes top 4.

5. Appends fixed tags: `#reels #songs #viral`

**Final caption format:**
```
🎵 Blinding Lights on repeat ✨

#viral #reels #music #songs #reels #songs #viral
```

> **Fallback:** If the hashtag API fails (permissions error), falls back gracefully to `#reels #songs #viral`. If `caption` is empty, `create_reel_container` uses the `INSTAGRAM_CAPTION` env var.

**Output fields:** `caption`

---

### Node 3 — `calculate_video_duration`

**What it does:**
Determines the final target duration of the reel.

```
final_duration = min(MUSIC_START_SECONDS + audio_duration, REEL_MAX_DURATION)
```

- Validates that the asset is longer than `MUSIC_START_SECONDS`
- Validates that `LOOP_START_SECONDS >= MUSIC_START_SECONDS`

**Output fields:** `asset_duration`, `final_duration`

---

### Node 4 — `build_video`

The core FFmpeg processing node. Runs entirely in a temp directory (`reel_tmp/`) that is cleaned up after completion.

See [FFmpeg Video Processing](#ffmpeg-video-processing) and [Audio Strategy](#audio-strategy) for full details.

**Output fields:** `output_video`

---

### Node 5 — `upload_to_drive`

**What it does:**
Uploads `final_reel.mp4` to Google Drive and makes it publicly readable.

**Steps:**
1. Builds OAuth2 credentials from `DRIVE_REFRESH_TOKEN`, auto-refreshes the access token.
2. Uploads via resumable upload (handles large files).
3. Sets `anyone → reader` permission.
4. Returns a direct download URL:
   `https://drive.google.com/uc?export=download&id={file_id}`

> Meta requires the video to be at a publicly accessible URL to create an Instagram container.

**Output fields:** `video_public_url`

---

### Node 6 — `check_public_video_url`

Validates that a public URL is available — either from the Drive upload or from the `VIDEO_PUBLIC_URL` env var fallback. Raises a clear error with instructions if missing.

---

### Node 7 — `create_reel_container`

**What it does:**
Calls `POST /{ig_user_id}/media` to create an Instagram media container.

**Payload:**
```json
{
  "media_type": "REELS",
  "video_url": "<drive_url>",
  "caption": "<generated_caption>",
  "share_to_feed": "true|false",
  "audio_configuration": {
    "audio_id": "<trending_audio_id>",
    "audio_volume": 100,
    "video_volume": 100
  }
}
```

- `audio_configuration` attaches the trending catalog track on Instagram.
- `video_volume: 100` keeps the embedded asset audio (first 5s) audible on Instagram.

**Output fields:** `container_id`

---

### Node 8 — `wait_for_container`

**What it does:**
Polls `GET /{container_id}?fields=status_code,status` every 10 seconds, up to 90 attempts (15 minutes total), waiting for Meta to finish processing.

| `status_code` | Meaning |
|---|---|
| `IN_PROGRESS` | Meta is still processing |
| `FINISHED` | Ready to publish |
| `ERROR` | Processing failed |
| `EXPIRED` | Container expired |

---

### Node 9 — `publish_reel`

**What it does:**
If `PUBLISH_REEL=true`, calls `POST /{ig_user_id}/media_publish` with the `container_id` to publish the reel.

If `PUBLISH_REEL=false` (default), exits as a **dry run** — everything is built and uploaded, but nothing is posted.

**Output fields:** `media_id`, `status`

---

## FFmpeg Video Processing

All video processing happens inside `build_video`. Temp files live in `reel_tmp/` and are deleted on completion.

### Step 1 — Normalize

Scales and crops the asset video to exact **1080×1920** (9:16 portrait):

```
scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,setsar=1
```

- `force_original_aspect_ratio=increase` → scales up so both dimensions meet the target
- `crop=1080:1920` → center-crops the overshoot (**fills frame completely, no black bars**)
- `setsar=1` → sets the Sample Aspect Ratio to square pixels

### Step 2 — Extract Loop Segment

Cuts `[LOOP_START_SECONDS → end]` of the normalized video (video only, no audio).
This segment is looped to fill the remaining duration after the asset ends.

### Step 3 — Concatenate

Builds a `concat.txt` list:
```
file 'normalized.mp4'       # full asset
file 'loop_segment.mp4'     # repeated N times to fill final_duration
```
Runs `ffmpeg -f concat` to stitch them together.

### Step 4 — Final Mux

Combines the concatenated video + mixed audio into the final output, applying the same `scale+crop` filter to guarantee exact 1080×1920 output even if the concat step introduced any dimension drift.

---

## Audio Strategy

The audio track in the local MP4 is deliberately **not** the full catalog music.
Instagram handles the music mixing on its own side via `audio_configuration`.

### Audio Timeline

```
0s ─────────── 5s ──────────────────────────── final_duration
│ Asset audio  │          Silence               │
│  (1.8x loud) │  (Instagram overlays catalog)  │
```

### Why Silence After 5s?

When `audio_configuration` is set, Instagram plays the catalog audio on top of the embedded video audio. By putting silence in the local file after 5s, the experience on Instagram is:

- **0–5s:** Original clip audio (boosted 1.8×) + catalog music begins fading in
- **5s+:** Catalog music only — no competing audio from the clip

### Volume Boost

The 5s asset audio is extracted with a `volume` filter:
```
ffmpeg -af "volume=1.8" ...
```
Controlled by `ASSET_AUDIO_BOOST` env var.

| Value | Effect |
|---|---|
| `1.0` | Original level |
| `1.5` | 50% louder |
| `1.8` | **Default** |
| `2.0` | Twice as loud |

---

## Graph Wiring

```
Entry: find_trending_audio
  ↓
generate_caption
  ↓
calculate_video_duration
  ↓
build_video
  ↓
upload_to_drive
  ↓
check_public_video_url
  ↓
create_reel_container
  ↓
wait_for_container
  ↓
publish_reel
  ↓
END
```

---

## Error Handling

| Scenario | Behaviour |
|---|---|
| Missing env vars | `validate_environment()` raises `RuntimeError` before graph starts |
| Asset file not found | `FileNotFoundError` before graph starts |
| Meta API error | `RuntimeError` with full JSON response logged |
| No trending audio found | `RuntimeError` with raw API response |
| Asset shorter than `MUSIC_START_SECONDS` | `RuntimeError` with durations logged |
| `LOOP_START_SECONDS` beyond asset duration | `RuntimeError` |
| Hashtag API fails | Silent warning, falls back to fixed tags |
| Drive upload fails | `RuntimeError` from Google API client |
| Container processing fails/expires | `RuntimeError` with Meta status |
| Container polling timeout (15 min) | `TimeoutError` |
| `PUBLISH_REEL=false` | Clean dry-run exit, status = `DRY_RUN` |
| `KeyboardInterrupt` | Clean exit with code 130 |

---

## GitHub Actions Integration

The workflow (`.github/workflows/generate-video.yml`) automates the full pipeline:

- **Schedule:** Every 4 hours (`0 */4 * * *`) — 6 uploads per day
- **Manual trigger:** Available from the GitHub Actions tab (`workflow_dispatch`)
- **Concurrency:** Only one run at a time (`group: reel-pipeline`)
- **Timeout:** 45 minutes per run

### Key env vars set in the workflow

```yaml
VIDEO_WIDTH:   "1080"
VIDEO_HEIGHT:  "1920"
VIDEO_FPS:     "30"
PUBLISH_REEL:  "true"
SHARE_TO_FEED: "true"
```

All secrets (`META_ACCESS_TOKEN`, `INSTAGRAM_USER_ID`, Drive credentials, `INSTAGRAM_CAPTION`) are injected from GitHub repository secrets.

The final `final_reel.mp4` is uploaded as a GitHub Actions artifact (retained 7 days) for debugging.
