import os
import sys
import time
import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import TypedDict, Optional, Dict, Any

import requests
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END

from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request as GoogleRequest
from googleapiclient.discovery import build as gdrive_build
from googleapiclient.http import MediaFileUpload


# ============================================================
# CONFIG
# ============================================================

load_dotenv()

GRAPH_VERSION = os.getenv("META_GRAPH_VERSION", "v24.0")

IG_USER_ID = os.getenv("INSTAGRAM_USER_ID")
META_ACCESS_TOKEN = os.getenv("META_ACCESS_TOKEN")

ASSET_VIDEO = os.getenv("ASSET_VIDEO", "asset.mp4")
OUTPUT_VIDEO = os.getenv("OUTPUT_VIDEO", "final_reel.mp4")

CAPTION = os.getenv("INSTAGRAM_CAPTION", "")

# How much of the original asset stays before catalog music.
MUSIC_START_SECONDS = float(
    os.getenv("MUSIC_START_SECONDS", "5.0")
)

# The section that should loop forever after the original asset ends.
LOOP_START_SECONDS = float(
    os.getenv("LOOP_START_SECONDS", "9.5")
)

# Output video settings
WIDTH = int(os.getenv("VIDEO_WIDTH", "1080"))
HEIGHT = int(os.getenv("VIDEO_HEIGHT", "1920"))
FPS = int(os.getenv("VIDEO_FPS", "30"))

# Meta Audio API
AUDIO_TYPE = "music"
AUDIO_LIMIT = int(os.getenv("AUDIO_LIMIT", "20"))

# Publishing
SHARE_TO_FEED = os.getenv(
    "SHARE_TO_FEED",
    "false"
).lower() == "true"

# IMPORTANT:
# Keep false until you have confirmed the exact Meta audio
# attachment parameter available to your app/version.
PUBLISH_REEL = (
    os.getenv("PUBLISH_REEL", "false").lower() == "true"
)

# Public URL for the generated video.
#
# Example:
# https://your-public-host.com/final_reel.mp4
#
# Meta needs to be able to fetch this URL.
VIDEO_PUBLIC_URL = os.getenv("VIDEO_PUBLIC_URL")

# Optional configurable audio attachment field.
#
# DO NOT assume this is valid.
#
# Example only:
# META_AUDIO_CREATE_FIELD=audio_id
#
# The code will only use this if explicitly supplied.
META_AUDIO_CREATE_FIELD = os.getenv(
    "META_AUDIO_CREATE_FIELD",
    ""
).strip()


# ============================================================
# VALIDATION
# ============================================================

def require_env(name: str) -> str:
    value = os.getenv(name)

    if not value:
        raise RuntimeError(
            f"Missing environment variable: {name}"
        )

    return value


def validate_environment():
    missing = []

    for name in [
        "INSTAGRAM_USER_ID",
        "META_ACCESS_TOKEN",
    ]:
        if not os.getenv(name):
            missing.append(name)

    if missing:
        raise RuntimeError(
            "Missing environment variables:\n"
            + "\n".join(f"  - {x}" for x in missing)
        )

    if not Path(ASSET_VIDEO).exists():
        raise FileNotFoundError(
            f"Asset video does not exist: {ASSET_VIDEO}"
        )


# ============================================================
# GRAPH STATE
# ============================================================

class ReelState(TypedDict, total=False):

    asset_video: str
    output_video: str

    asset_duration: float
    final_duration: float

    audio_id: str
    audio_title: str
    audio_metadata: Dict[str, Any]
    audio_duration: float

    video_public_url: str

    container_id: str
    media_id: str

    status: str
    error: str


# ============================================================
# HTTP HELPERS
# ============================================================

GRAPH_BASE = "https://graph.facebook.com"


def graph_get(
    path: str,
    params: Optional[Dict[str, Any]] = None
):
    params = params or {}

    params["access_token"] = META_ACCESS_TOKEN

    url = f"{GRAPH_BASE}/{GRAPH_VERSION}/{path.lstrip('/')}"

    response = requests.get(
        url,
        params=params,
        timeout=60,
    )

    try:
        data = response.json()
    except Exception:
        data = {
            "raw": response.text
        }

    if not response.ok:
        raise RuntimeError(
            f"Meta GET failed ({response.status_code}):\n"
            f"{json.dumps(data, indent=2)}"
        )

    return data


def graph_post(
    path: str,
    data: Optional[Dict[str, Any]] = None
):
    data = data or {}

    data["access_token"] = META_ACCESS_TOKEN

    url = f"{GRAPH_BASE}/{GRAPH_VERSION}/{path.lstrip('/')}"

    response = requests.post(
        url,
        data=data,
        timeout=60,
    )

    try:
        result = response.json()
    except Exception:
        result = {
            "raw": response.text
        }

    if not response.ok:
        raise RuntimeError(
            f"Meta POST failed ({response.status_code}):\n"
            f"{json.dumps(result, indent=2)}"
        )

    return result


# ============================================================
# FFMPEG HELPERS
# ============================================================

def run(cmd):
    print("\n$", " ".join(map(str, cmd)))

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        print(result.stderr)

        raise RuntimeError(
            f"Command failed with exit code "
            f"{result.returncode}"
        )

    return result


def get_video_duration(path: str) -> float:

    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        path,
    ]

    result = run(cmd)

    return float(result.stdout.strip())


# ============================================================
# NODE 1
# FIND TRENDING INSTAGRAM AUDIO
# ============================================================

def find_trending_audio(state: ReelState):

    print("\n=== FIND INSTAGRAM AUDIO ===")

    result = graph_get(
        "/ig_audio",
        {
            "ig_user_id": IG_USER_ID,
            "audio_type": AUDIO_TYPE,
            "limit": AUDIO_LIMIT,
        },
    )

    print(
        json.dumps(
            result,
            indent=2
        )
    )

    # Meta's /ig_audio response wraps tracks under the "audio" key.
    # Fall back to "data" or "items" for forward compatibility.
    raw = (
        result.get("audio")
        or result.get("data")
        or result.get("items")
    )

    # Flatten: if the response itself is a list, use it directly.
    if raw is None and isinstance(result, list):
        raw = result

    # Keep only valid track dicts that have an audio identifier.
    tracks = [
        item
        for item in (raw or [])
        if isinstance(item, dict)
        and (
            item.get("audio_id")
            or item.get("id")
        )
    ]

    if not tracks:
        raise RuntimeError(
            "Meta returned no Instagram catalog audio.\n"
            f"Raw response: {json.dumps(result, indent=2)}"
        )

    # Strategy: pick the most trending track that also has a download_url
    # so we can mix audio locally AND attach it to Instagram via audio_id.
    # Trending = highest position in the API response (index 0 = most trending).
    track_with_url = next(
        (t for t in tracks if t.get("download_url")),
        None
    )

    # Use the trending track with download_url; fall back to first track.
    track = track_with_url or tracks[0]

    if not track.get("download_url"):
        print(
            "WARNING: Selected track has no download_url. "
            "Music portion in local video will be silent, "
            "but audio_id will still be attached on Instagram."
        )

    # Log trending rank
    rank = tracks.index(track) + 1
    print(f"  Trending rank: #{rank} of {len(tracks)} returned")

    audio_id = (
        track.get("id")
        or track.get("audio_id")
    )

    if not audio_id:
        raise RuntimeError(
            "Could not find audio ID in Meta response."
        )

    title = (
        track.get("title")
        or track.get("name")
        or "Unknown"
    )

    print(
        f"\nSelected Instagram audio:"
        f"\n  ID: {audio_id}"
        f"\n  Title: {title}"
    )

    # FIX 1:
    # Meta's catalog response uses "duration_in_ms" not "duration_ms".
    # Check all known field names to be safe.
    duration_ms = (
        track.get("duration_in_ms")
        or track.get("duration_ms")
    )

    duration_s = (
        track.get("duration")
    )

    if duration_ms is not None:
        duration = float(duration_ms) / 1000.0
    elif duration_s is not None:
        duration = float(duration_s)
    else:
        raise RuntimeError(
            f"No duration found in track: {json.dumps(track, indent=2)}"
        )

    print(
        f"  Duration: {duration:.2f}s"
    )

    return {
        **state,
        "audio_id": str(audio_id),
        "audio_title": title,
        "audio_metadata": track,
        "audio_duration": duration,
    }


# ============================================================
# NODE 2
# CALCULATE FINAL VIDEO LENGTH
# ============================================================

# Instagram Reel maximum duration in seconds
REEL_MAX_DURATION = float(os.getenv("REEL_MAX_DURATION", "90"))


def calculate_video_duration(state: ReelState):

    print("\n=== CALCULATE VIDEO DURATION ===")

    asset = state["asset_video"]

    duration = get_video_duration(asset)

    audio_duration = state["audio_duration"]

    # Target: asset intro + catalog audio, capped at reel max.
    raw_duration = MUSIC_START_SECONDS + audio_duration

    # Instagram Reels max = 90 seconds.
    final_duration = min(raw_duration, REEL_MAX_DURATION)

    print(
        f"Asset duration:       {duration:.3f}s"
    )

    print(
        f"Music starts:         {MUSIC_START_SECONDS:.3f}s"
    )

    print(
        f"Catalog audio:        {audio_duration:.3f}s"
    )

    print(
        f"Raw duration:         {raw_duration:.3f}s"
    )

    print(
        f"Final video duration: {final_duration:.3f}s  (capped at {REEL_MAX_DURATION}s for Reels)"
    )

    if duration < MUSIC_START_SECONDS:
        raise RuntimeError(
            f"Asset is only {duration:.2f}s long, "
            f"but music starts at {MUSIC_START_SECONDS:.2f}s."
        )

    return {
        **state,
        "asset_duration": duration,
        "final_duration": final_duration,
    }


# ============================================================
# NODE 3
# BUILD VIDEO
# ============================================================

def download_catalog_audio(url: str, dest: Path) -> None:
    """Download the catalog audio MP4 from Meta CDN."""
    print(f"\nDownloading catalog audio...")

    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()

    with open(dest, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 256):
            f.write(chunk)

    print(f"Catalog audio saved: {dest}")


def build_video(state: ReelState):

    print("\n=== BUILD VIDEO ===")

    asset = state["asset_video"]
    output = state["output_video"]

    final_duration = state["final_duration"]
    loop_start = LOOP_START_SECONDS
    asset_duration = state["asset_duration"]

    if loop_start >= asset_duration:
        raise RuntimeError(
            f"LOOP_START_SECONDS={loop_start} is beyond "
            f"asset duration={asset_duration}"
        )

    if loop_start < MUSIC_START_SECONDS:
        raise RuntimeError(
            f"LOOP_START_SECONDS={loop_start} must be >= "
            f"MUSIC_START_SECONDS={MUSIC_START_SECONDS}. "
            f"The loop segment must start after the music crossover point."
        )

    temp_dir = Path("reel_tmp")
    temp_dir.mkdir(parents=True, exist_ok=True)

    normalized      = temp_dir / "normalized.mp4"
    loop_segment    = temp_dir / "loop_segment.mp4"
    catalog_audio   = temp_dir / "catalog_audio.mp4"
    asset_audio_seg = temp_dir / "asset_audio.aac"
    mixed_audio     = temp_dir / "mixed_audio.aac"

    try:

        # --------------------------------------------------------
        # Normalize video (keep asset audio track for first segment)
        # --------------------------------------------------------

        run([
            "ffmpeg", "-y",
            "-i", asset,
            "-vf",
            (
                f"scale={WIDTH}:{HEIGHT}:"
                "force_original_aspect_ratio=decrease,"
                f"pad={WIDTH}:{HEIGHT}:(ow-iw)/2:(oh-ih)/2"
            ),
            "-r", str(FPS),
            "-c:v", "libx264",
            "-preset", "medium",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "192k",
            "-movflags", "+faststart",
            str(normalized),
        ])

        # --------------------------------------------------------
        # Extract loop segment (video only — audio handled separately)
        # --------------------------------------------------------

        loop_duration = asset_duration - loop_start

        run([
            "ffmpeg", "-y",
            "-ss", str(loop_start),
            "-i", str(normalized),
            "-t", str(loop_duration),
            "-an",
            "-c:v", "libx264",
            "-preset", "medium",
            "-pix_fmt", "yuv420p",
            str(loop_segment),
        ])

        # --------------------------------------------------------
        # Repeat loop segment to cover final_duration
        # --------------------------------------------------------

        remaining = max(0.0, final_duration - asset_duration)
        repeat_count = (
            math.ceil(remaining / loop_duration)
            if remaining > 0
            else 0
        )

        concat_files = temp_dir / "concat.txt"
        with open(concat_files, "w", encoding="utf-8") as f:
            f.write(f"file '{normalized.resolve()}'\n")
            for _ in range(repeat_count):
                f.write(f"file '{loop_segment.resolve()}'\n")

        concatenated = temp_dir / "concatenated.mp4"
        run([
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_files),
            "-c", "copy",
            str(concatenated),
        ])

        # --------------------------------------------------------
        # Download catalog audio from Meta CDN
        # --------------------------------------------------------

        catalog_download_url = state["audio_metadata"].get("download_url")

        if catalog_download_url:
            download_catalog_audio(catalog_download_url, catalog_audio)
        else:
            # No download URL: use silence for the music portion
            print("WARNING: No download_url for catalog audio. Music portion will be silent.")
            catalog_audio = None

        # --------------------------------------------------------
        # Build audio track for the video file:
        #   0 -> MUSIC_START_SECONDS : original asset audio
        #   MUSIC_START_SECONDS -> final_duration : silence
        #
        # Why silence instead of catalog music?
        # On Instagram, audio_configuration plays the catalog track.
        # Setting video_volume=100 lets Instagram mix this embedded audio
        # (asset clip sound for 5s, then silence) with the catalog music.
        # Result on Instagram: 5s asset audio + catalog, then catalog only.
        # --------------------------------------------------------

        # Extract asset audio for the intro segment
        run([
            "ffmpeg", "-y",
            "-i", str(normalized),
            "-t", str(MUSIC_START_SECONDS),
            "-vn",
            "-acodec", "aac",
            "-b:a", "192k",
            str(asset_audio_seg),
        ])

        music_duration = final_duration - MUSIC_START_SECONDS

        # Generate silence for the catalog music portion
        silence_seg = temp_dir / "silence.aac"
        run([
            "ffmpeg", "-y",
            "-f", "lavfi",
            "-i", "anullsrc=r=44100:cl=stereo",
            "-t", str(music_duration),
            "-acodec", "aac",
            "-b:a", "192k",
            str(silence_seg),
        ])

        audio_concat = temp_dir / "audio_concat.txt"
        with open(audio_concat, "w", encoding="utf-8") as f:
            f.write(f"file '{asset_audio_seg.resolve()}'\n")
            f.write(f"file '{silence_seg.resolve()}'\n")

        run([
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(audio_concat),
            "-acodec", "aac",
            "-b:a", "192k",
            str(mixed_audio),
        ])

        # --------------------------------------------------------
        # Mux video + audio into final output, trimmed to final_duration
        # --------------------------------------------------------

        run([
            "ffmpeg", "-y",
            "-i", str(concatenated),
            "-i", str(mixed_audio),
            "-t", str(final_duration),
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-c:v", "libx264",
            "-preset", "medium",
            "-pix_fmt", "yuv420p",
            "-r", str(FPS),
            "-c:a", "aac",
            "-b:a", "192k",
            "-shortest",
            "-movflags", "+faststart",
            output,
        ])

        actual_duration = get_video_duration(output)

        print(
            f"\nFinal MP4:"
            f"\n  {output}"
            f"\n  duration={actual_duration:.3f}s"
        )

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
        print("\nCleaned up temp directory.")

    return {
        **state,
        "output_video": output,
    }


# ============================================================
# NODE 4
# CHECK PUBLIC VIDEO URL
# ============================================================

# ============================================================
# NODE 4A
# UPLOAD VIDEO TO GOOGLE DRIVE & GET PUBLIC URL
# ============================================================

def upload_to_drive(state: ReelState):

    print("\n=== UPLOAD TO GOOGLE DRIVE ===")

    output = state["output_video"]

    # Build credentials from .env tokens
    creds = Credentials(
        token=None,
        refresh_token=os.getenv("DRIVE_REFRESH_TOKEN"),
        client_id=os.getenv("GOOGLE_CLIENT_ID"),
        client_secret=os.getenv("GOOGLE_CLIENT_SECRET"),
        token_uri="https://oauth2.googleapis.com/token",
        scopes=["https://www.googleapis.com/auth/drive.file"],
    )

    # Auto-refresh the access token
    creds.refresh(GoogleRequest())

    service = gdrive_build(
        "drive",
        "v3",
        credentials=creds,
        cache_discovery=False,
    )

    file_metadata = {
        "name": Path(output).name,
    }

    print(f"Uploading {output} to Google Drive...")

    media = MediaFileUpload(
        output,
        mimetype="video/mp4",
        resumable=True,
    )

    uploaded = (
        service.files()
        .create(
            body=file_metadata,
            media_body=media,
            fields="id,name",
        )
        .execute()
    )

    file_id = uploaded.get("id")

    if not file_id:
        raise RuntimeError(
            "Google Drive did not return a file ID."
        )

    print(f"Drive file ID: {file_id}")

    # Make the file publicly readable (anyone with the link)
    service.permissions().create(
        fileId=file_id,
        body={
            "type": "anyone",
            "role": "reader",
        },
    ).execute()

    # Direct download URL — works for files under ~100MB without auth
    public_url = (
        f"https://drive.google.com/uc"
        f"?export=download&id={file_id}"
    )

    print(
        f"\nPublic Drive URL:"
        f"\n  {public_url}"
    )

    return {
        **state,
        "video_public_url": public_url,
    }


# ============================================================
# NODE 5
# CHECK PUBLIC VIDEO URL
# ============================================================

def check_public_video_url(state: ReelState):

    print("\n=== CHECK VIDEO URL ===")

    # Prefer URL already set by upload_to_drive node.
    # Fall back to VIDEO_PUBLIC_URL env var if Drive upload was skipped.
    url = (
        state.get("video_public_url")
        or VIDEO_PUBLIC_URL
    )

    if not url:
        raise RuntimeError(
            "\nVIDEO_PUBLIC_URL is missing.\n\n"
            "Either set VIDEO_PUBLIC_URL in .env\n"
            "or ensure DRIVE credentials are configured "
            "for auto-upload.\n"
        )

    print(
        "Video URL:"
        f"\n{url}"
    )

    return {
        **state,
        "video_public_url": url,
    }


# ============================================================
# NODE 5
# CREATE INSTAGRAM REEL CONTAINER
# ============================================================

def create_reel_container(state: ReelState):

    print("\n=== CREATE REEL CONTAINER ===")

    params = {
        "media_type": "REELS",
        "video_url": state["video_public_url"],
        "caption": CAPTION,
        "share_to_feed": str(
            SHARE_TO_FEED
        ).lower(),
    }

    # Attach trending audio via audio_configuration.
    # Per Meta Graph API docs, the correct field is audio_configuration
    # which accepts a JSON object with audio_id, audio_volume, video_volume.
    audio_id = state.get("audio_id")

    if audio_id:
        params["audio_configuration"] = json.dumps({
            "audio_id": audio_id,
            "audio_volume": 100,
            "video_volume": 100,  # keep asset clip audio audible in first 5s
        })
        print(
            f"Attaching audio_configuration:"
            f" audio_id={audio_id}"
        )
    else:
        print("WARNING: No audio_id available. Reel will use default audio.")
        if PUBLISH_REEL:
            raise RuntimeError(
                "PUBLISH_REEL=true but no audio_id found. "
                "Cannot publish without trending audio."
            )

    result = graph_post(
        f"/{IG_USER_ID}/media",
        params,
    )

    container_id = result.get("id")

    if not container_id:
        raise RuntimeError(
            "Meta did not return a container ID."
        )

    print(
        f"Container ID: {container_id}"
    )

    return {
        **state,
        "container_id": container_id,
    }


# ============================================================
# NODE 6
# WAIT FOR CONTAINER
# ============================================================

def wait_for_container(state: ReelState):

    print("\n=== WAIT FOR INSTAGRAM PROCESSING ===")

    container_id = state["container_id"]

    # FIX 4:
    # Increased from 60 -> 90 attempts (15 minutes total).
    # Added elapsed time logging so you can see how long Meta is taking.
    max_attempts = 90

    start_time = time.time()

    for attempt in range(max_attempts):

        result = graph_get(
            f"/{container_id}",
            {
                "fields":
                    "status_code,status"
            },
        )

        status_code = result.get(
            "status_code"
        )

        status = result.get(
            "status"
        )

        elapsed = time.time() - start_time

        print(
            f"[{attempt + 1}/{max_attempts}] "
            f"{status_code} - {status} "
            f"(elapsed: {elapsed:.0f}s)"
        )

        if status_code == "FINISHED":

            return {
                **state,
                "status": "FINISHED",
            }

        if status_code in [
            "ERROR",
            "EXPIRED",
        ]:

            raise RuntimeError(
                "Instagram container failed:\n"
                + json.dumps(
                    result,
                    indent=2
                )
            )

        time.sleep(10)

    raise TimeoutError(
        "Instagram container did not finish "
        "within 15 minutes (90 attempts)."
    )


# ============================================================
# NODE 7
# PUBLISH
# ============================================================

def publish_reel(state: ReelState):

    print("\n=== PUBLISH REEL ===")

    if not PUBLISH_REEL:

        print(
            "\nPUBLISH_REEL=false"
            "\n"
            "Dry run complete."
            "\n"
            "Nothing was published."
        )

        return {
            **state,
            "status": "DRY_RUN",
        }

    result = graph_post(
        f"/{IG_USER_ID}/media_publish",
        {
            "creation_id":
                state["container_id"],
        },
    )

    media_id = result.get("id")

    if not media_id:
        raise RuntimeError(
            "Instagram did not return media ID."
        )

    print(
        f"\nPublished Instagram Media ID:"
        f" {media_id}"
    )

    return {
        **state,
        "media_id": media_id,
        "status": "PUBLISHED",
    }


# ============================================================
# LANGGRAPH
# ============================================================

def build_graph():

    graph = StateGraph(ReelState)

    graph.add_node(
        "find_trending_audio",
        find_trending_audio,
    )

    graph.add_node(
        "calculate_video_duration",
        calculate_video_duration,
    )

    graph.add_node(
        "build_video",
        build_video,
    )

    graph.add_node(
        "check_public_video_url",
        check_public_video_url,
    )

    graph.add_node(
        "create_reel_container",
        create_reel_container,
    )

    graph.add_node(
        "wait_for_container",
        wait_for_container,
    )

    graph.add_node(
        "publish_reel",
        publish_reel,
    )

    graph.set_entry_point(
        "find_trending_audio"
    )

    graph.add_edge(
        "find_trending_audio",
        "calculate_video_duration",
    )

    graph.add_edge(
        "calculate_video_duration",
        "build_video",
    )

    graph.add_node(
        "upload_to_drive",
        upload_to_drive,
    )

    graph.add_edge(
        "build_video",
        "upload_to_drive",
    )

    graph.add_edge(
        "upload_to_drive",
        "check_public_video_url",
    )

    graph.add_edge(
        "check_public_video_url",
        "create_reel_container",
    )

    graph.add_edge(
        "create_reel_container",
        "wait_for_container",
    )

    graph.add_edge(
        "wait_for_container",
        "publish_reel",
    )

    graph.add_edge(
        "publish_reel",
        END,
    )

    return graph.compile()


# ============================================================
# MAIN
# ============================================================

def main():

    validate_environment()

    print(
        "\n=========================================="
        "\n AUTOMATED INSTAGRAM REEL BUILDER"
        "\n=========================================="
    )

    print(
        f"\nAsset: {ASSET_VIDEO}"
    )

    print(
        f"Music start: "
        f"{MUSIC_START_SECONDS}s"
    )

    print(
        f"Loop start: "
        f"{LOOP_START_SECONDS}s"
    )

    print(
        f"Publishing: "
        f"{PUBLISH_REEL}"
    )

    initial_state: ReelState = {
        "asset_video": ASSET_VIDEO,
        "output_video": OUTPUT_VIDEO,
    }

    app = build_graph()

    final_state = app.invoke(
        initial_state
    )

    print(
        "\n=========================================="
        "\n COMPLETE"
        "\n=========================================="
    )

    print(
        json.dumps(
            {
                k: v
                for k, v in final_state.items()
                if k not in [
                    "audio_metadata"
                ]
            },
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:

        print(
            "\nStopped by user."
        )

        sys.exit(130)

    except Exception as e:

        print(
            "\nERROR:"
        )

        print(e)

        sys.exit(1)