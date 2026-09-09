"""YouTube Shorts workflow.

Fetches trending royalty-free song via yt-dlp, assembles a vertical
1080x1920 video with karaoke captions, and uploads to YouTube as a Short.

IMPORTANT: Title always includes #Shorts so YouTube classifies it correctly.

Run:  python youtube_workflow.py
"""

import functools
import http.client
import json
import os
import random
import subprocess
import sys
import time
import pathlib
from contextlib import ExitStack
from pathlib import Path
from typing import TypedDict

import httplib2
import requests
from dotenv import load_dotenv
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload
from faster_whisper import WhisperModel
from langgraph.graph import END, START, StateGraph
from moviepy import (
    afx,
    AudioFileClip,
    ColorClip,
    CompositeAudioClip,
    CompositeVideoClip,
    TextClip,
    VideoFileClip,
    concatenate_audioclips,
)

load_dotenv()

# Fix Windows console encoding
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR       = Path(__file__).resolve().parent
ASSET_VIDEO    = BASE_DIR / "assets" / "the_transition_i_n_between_is.mp4"
AUDIO_DIR      = BASE_DIR / "output_audio"
FINAL_DIR      = BASE_DIR / "output_final_video"
FONT_PATH      = BASE_DIR / "assets" / "Roboto-Bold.ttf"
USED_SONGS_DB    = BASE_DIR / "assets" / "used_songs.json"
CAPTION_FONT_URL = "https://fonts.gstatic.com/s/anton/v25/1Ptgg87LROyAm0K08i4gS7lu.woff2"
CAPTION_FONT_PATH = BASE_DIR / "assets" / "Anton-Regular.ttf"
# Written by youtube_workflow after a successful download so instagram_workflow
# can reuse the same audio without a second yt-dlp call.
SHARED_AUDIO_MANIFEST = BASE_DIR / "output_audio" / ".shared_audio_manifest.json"

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
MUSIC_VOLUME        = 0.20
MUSIC_START_OFFSET  = 5.0            # song starts 5 seconds into the video
VIDEO_BITRATE       = "8000k"
VIDEO_PRESET        = "slow"
VIDEO_FFMPEG_PARAMS = ["-crf", "18", "-movflags", "+faststart"]

# YouTube Audio Library — copyright-free music for YouTube creators
# /videos tab ensures yt-dlp lists individual tracks
YT_AUDIO_LIBRARY_CHANNEL = "https://www.youtube.com/@YouTubeAudioLibrary/videos"

# ---------------------------------------------------------------------------
# Verified royalty-free music sources and their licensing policies.
# Only sources listed here pass validate_music_license().
# IMPORTANT: This registry confirms royalty-free status only.
#            It does NOT prevent YouTube Content ID claims for copyrighted music.
# ---------------------------------------------------------------------------
MUSIC_LICENSE_POLICIES: dict = {
    "YouTube Audio Library": {
        "license":              "YouTube Audio Library License",
        "license_url":          "https://www.youtube.com/audiolibrary/policies",
        "attribution_required": True,
    },
    "NoCopyrightSounds": {
        "license":              "NoCopyrightSounds License",
        "license_url":          "https://nocopyrightsounds.co.uk/licensing/",
        "attribution_required": True,
    },
    "Pixabay Music": {
        "license":              "Pixabay Content License",
        "license_url":          "https://pixabay.com/service/license-summary/",
        "attribution_required": False,
    },
}

REQUIRED_ENV_VARS = [
    "GOOGLE_CLIENT_ID",
    "GOOGLE_CLIENT_SECRET",
    "YOUTUBE_REFRESH_TOKEN",
]


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
class MusicMetadata(TypedDict):
    """Structured music licensing metadata — kept separate from VideoState.

    The renderer only receives this validated dict, never raw song titles
    from the fetcher.  validate_music_license() is the single gate that
    populates license/attribution fields and rejects unverified sources.
    """
    title:                str
    artist:               str
    source:               str    # e.g. "YouTube Audio Library"
    license:              str    # human-readable license name
    license_url:          str    # canonical license policy URL
    attribution_required: bool
    attribution_text:     str    # pre-formatted attribution string


class VideoState(TypedDict):
    trending_song_title:      str
    trending_song_artist:     str
    audio_library_video_id:   str   # YouTube Audio Library video ID
    music_path:               str
    music_attribution:        str
    music_metadata:           dict  # validated MusicMetadata — renderer reads this
    caption_words:            list  # timed word list from whisper
    final_video_path:         str
    youtube_url:              str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_caption_font() -> str:
    """Download Anton font (great for Shorts captions) if not already present."""
    if CAPTION_FONT_PATH.exists():
        return str(CAPTION_FONT_PATH)
    # Try Anton TTF from GitHub mirror (woff2 won't work with moviepy, use TTF)
    ttf_url = "https://github.com/google/fonts/raw/main/ofl/anton/Anton-Regular.ttf"
    try:
        print("[font] downloading Anton-Regular.ttf ...")
        r = requests.get(ttf_url, timeout=15)
        r.raise_for_status()
        CAPTION_FONT_PATH.parent.mkdir(parents=True, exist_ok=True)
        CAPTION_FONT_PATH.write_bytes(r.content)
        print(f"[font] saved -> {CAPTION_FONT_PATH}")
        return str(CAPTION_FONT_PATH)
    except Exception as exc:
        print(f"[font] download failed ({exc}), falling back to Roboto")
        return str(FONT_PATH) if FONT_PATH.exists() else None

def validate_prerequisites() -> None:
    missing = [v for v in REQUIRED_ENV_VARS if not os.getenv(v)]
    if missing:
        raise EnvironmentError("Missing env vars: " + ", ".join(missing))
    if not ASSET_VIDEO.is_file():
        raise FileNotFoundError(f"Asset video not found: {ASSET_VIDEO}")


# ---------------------------------------------------------------------------
# yt-dlp cookies (CI runner IPs are frequently bot-blocked by YouTube without
# these). Set the YOUTUBE_COOKIES secret to the contents of a Netscape-format
# cookies.txt exported from a signed-in browser session (e.g. via the
# "Get cookies.txt LOCALLY" extension). If unset, yt-dlp runs unauthenticated
# and may silently return zero results on hosted CI runners.
# ---------------------------------------------------------------------------
_COOKIE_FILE = BASE_DIR / ".ytdlp_cookies.txt"


def _ytdlp_cookie_args() -> list:
    cookies_content = os.getenv("YOUTUBE_COOKIES", "")
    if not cookies_content:
        return []
    try:
        if not _COOKIE_FILE.exists():
            _COOKIE_FILE.write_text(cookies_content, encoding="utf-8")
        return ["--cookies", str(_COOKIE_FILE)]
    except Exception as exc:
        print(f"[ytdlp] could not write cookies file: {exc}")
        return []


def _log_ytdlp_failure(label: str, result) -> None:
    """Print whatever yt-dlp actually said, instead of failing silently."""
    stderr = (result.stderr or "").strip()
    stdout = (result.stdout or "").strip()
    print(f"[{label}] yt-dlp exited {result.returncode}")
    if stderr:
        print(f"[{label}] stderr: {stderr[-1500:]}")
    elif stdout:
        print(f"[{label}] stdout: {stdout[-1500:]}")


def with_retry(max_attempts: int = 3, base_delay: int = 2):
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return fn(*args, **kwargs)
                except Exception:
                    if attempt + 1 == max_attempts:
                        raise
                    delay = base_delay * (2 ** attempt) + random.random()
                    print(f"[{fn.__name__}] retry in {delay:.1f}s ...")
                    time.sleep(delay)
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# No-repeat tracking
# ---------------------------------------------------------------------------
def _load_used_songs() -> set:
    if USED_SONGS_DB.is_file():
        try:
            return set(json.loads(USED_SONGS_DB.read_text(encoding="utf-8")))
        except Exception:
            pass
    return set()


def _save_used_song(title: str, artist: str) -> None:
    used = _load_used_songs()
    used.add(f"{title}|{artist}")
    USED_SONGS_DB.parent.mkdir(parents=True, exist_ok=True)
    USED_SONGS_DB.write_text(json.dumps(sorted(used), indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Chorus extraction (highest-energy segment via librosa RMS)
# ---------------------------------------------------------------------------
def _extract_chorus(src_path: Path, dest_path: Path, duration: float) -> None:
    """Extract the best chorus segment using FFmpeg only (no librosa required).

    Strategy: probe total duration, sample loudness at N points via
    ffmpeg volumedetect, then cut from the loudest window.
    Falls back to the 30%-mark of the song (typical first-chorus position)
    if probing fails.
    """
    try:
        # Get total duration via ffprobe
        probe = subprocess.run(
            ["ffprobe", "-v", "error",
             "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1",
             str(src_path)],
            capture_output=True, text=True, check=True, timeout=15,
        )
        total_dur = float(probe.stdout.strip())

        # Sample loudness at ~8 evenly-spaced windows to find the peak-energy segment
        step        = max(5.0, (total_dur - duration) / 8)
        best_start  = max(0.0, total_dur * 0.30 - duration / 2)  # default: 30% mark
        best_vol    = float("-inf")
        t           = 0.0

        while t + duration <= total_dur:
            res = subprocess.run(
                ["ffmpeg", "-y", "-ss", str(t), "-t", str(min(duration, 15.0)),
                 "-i", str(src_path), "-af", "volumedetect",
                 "-f", "null", "-"],
                capture_output=True, text=True, timeout=30,
            )
            for line in res.stderr.splitlines():
                if "mean_volume" in line:
                    try:
                        vol = float(line.split("mean_volume:")[1].split("dB")[0].strip())
                        if vol > best_vol:
                            best_vol   = vol
                            best_start = t
                    except ValueError:
                        pass
            t += step

        print(f"[chorus] loudest window at {best_start:.1f}s "
              f"(mean {best_vol:.1f} dB) — extracting {duration:.1f}s")
        subprocess.run(
            ["ffmpeg", "-y",
             "-ss", str(best_start), "-t", str(duration),
             "-i", str(src_path),
             "-b:a", "192k", str(dest_path)],
            check=True, capture_output=True, timeout=60,
        )
    except Exception as exc:
        print(f"[chorus] ffmpeg scan failed ({exc}), using 30%-mark fallback")
        try:
            probe = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", str(src_path)],
                capture_output=True, text=True, timeout=15,
            )
            total_dur  = float(probe.stdout.strip()) if probe.returncode == 0 else duration * 3
            start      = max(0.0, min(total_dur * 0.30, total_dur - duration))
        except Exception:
            start = 0.0
        subprocess.run(
            ["ffmpeg", "-y", "-ss", str(start), "-t", str(duration),
             "-i", str(src_path), "-b:a", "192k", str(dest_path)],
            check=True, capture_output=True, timeout=60,
        )




# ---------------------------------------------------------------------------
# Music License Validation + Attribution Badge
# ---------------------------------------------------------------------------
def validate_music_license(metadata: dict) -> dict:
    """Validate music licensing metadata before the video is rendered.

    Rules:
      • Source must be in MUSIC_LICENSE_POLICIES (verified royalty-free).
      • License/license_url fields are authoritative from the policy registry.
      • If attribution_required, attribution_text is auto-generated.
      • Unknown / unverified sources raise ValueError → video rendered without music.

    IMPORTANT: This validates royalty-free status only.
    It does NOT prevent Content ID claims for copyrighted music.
    Adding an attribution label does not make copyrighted music legal to use.
    """
    source = metadata.get("source", "")
    policy = MUSIC_LICENSE_POLICIES.get(source)

    if not policy:
        raise ValueError(
            f"Unverified music source {source!r}. "
            "Only tracks from verified royalty-free sources are allowed. "
            f"Accepted sources: {list(MUSIC_LICENSE_POLICIES)}"
        )

    # Merge authoritative policy fields into a copy of the metadata
    validated                        = dict(metadata)
    validated["license"]             = policy["license"]
    validated["license_url"]         = policy["license_url"]
    validated["attribution_required"]= policy["attribution_required"]

    if validated["attribution_required"] and not validated.get("attribution_text"):
        validated["attribution_text"] = (
            f'Music: "{validated["title"]}" by {validated["artist"]} '
            f'| {source} — {policy["license_url"]}'
        )

    print(
        f"[validate_music_license] ✓ source={source!r} "
        f"license={validated['license']!r} "
        f"attribution_required={validated['attribution_required']}"
    )
    return validated


def _probe_media(path: Path) -> dict:
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"Missing or empty media file: {path}")

    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration:stream=codec_type",
            "-of", "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    info = json.loads(result.stdout)
    duration = float(info.get("format", {}).get("duration", 0))
    if not 0 < duration < float("inf"):
        raise RuntimeError(f"Invalid media duration: {path}")

    return {
        "duration": duration,
        "stream_types": {
            stream.get("codec_type")
            for stream in info.get("streams", [])
        },
    }


def prepare_music_for_render(state: VideoState) -> dict:
    metadata = state.get("music_metadata")
    if not metadata:
        raise RuntimeError("Music metadata is missing. Upload aborted.")

    validated = validate_music_license(metadata)

    music_path = state.get("music_path")
    if not music_path:
        raise RuntimeError("Music download is missing. Upload aborted.")

    info = _probe_media(Path(music_path))
    if "audio" not in info["stream_types"]:
        raise RuntimeError("Downloaded music has no audio stream.")

    state["music_metadata"] = validated
    state["music_attribution"] = validated.get("attribution_text", "")
    return validated


def _render_attribution_badge(
    video,
    metadata: dict,
    badge_start: float,
    total_dur: float,
    font: str,
) -> list:
    """Create a YouTube Shorts-style music attribution badge overlay.

    Displays:  ♪  {title}  ·  {artist}

    The label identifies the royalty-free music source.
    It does NOT prevent Content ID claims for copyrighted music.

    Returns a list of moviepy clips to composite onto the video.
    """
    title  = metadata.get("title", "")
    artist = metadata.get("artist", "")
    if not title:
        return []

    badge_dur = max(0.0, total_dur - badge_start)
    if badge_dur <= 0:
        return []

    label_text = f"♪  {title}  ·  {artist}"
    badge_y    = int(1920 * 0.87)   # bottom area, like YouTube Shorts music label
    pad_x, pad_y = 28, 12

    try:
        badge_txt = TextClip(
            text=label_text, font=font, font_size=32,
            color="white", stroke_color="black", stroke_width=1,
            size=(int(1080 * 0.86), None), method="caption",
        )
        badge_bg = (
            ColorClip(
                size=(badge_txt.w + pad_x * 2, badge_txt.h + pad_y * 2),
                color=(10, 10, 10),
            )
            .with_opacity(0.72)
            .with_start(badge_start).with_duration(badge_dur)
            .with_position(("center", badge_y))
        )
        badge_txt = (
            badge_txt
            .with_start(badge_start).with_duration(badge_dur)
            .with_position(("center", badge_y + pad_y))
        )
        return [badge_bg, badge_txt]
    except Exception as exc:
        print(f"[attribution_badge] could not render badge: {exc}")
        return []


# ---------------------------------------------------------------------------
# Node 1 — fetch_trending_song
# ---------------------------------------------------------------------------
def fetch_trending_song(state: VideoState) -> VideoState:
    """Pick a royalty-free track from NCS or YouTube Audio Library.

    Strategy (in order):
      1. NoCopyrightSounds channel  — reliable, yt-dlp enumerates it well
      2. YouTube Audio Library channel  — may require extra yt-dlp flags
      3. yt-dlp search fallback  — fixed flags (no --flat-playlist for search)
      4. Hardcoded emergency tracks  — known-good video IDs, always present
    """
    print("[fetch_trending_song] fetching royalty-free tracks ...")
    used    = _load_used_songs()
    entries = []

    # ─── Source definitions ───────────────────────────────────────────────
    CHANNEL_SOURCES = [
        {
            "url":    "https://www.youtube.com/@NoCopyrightSounds/videos",
            "source": "NoCopyrightSounds",
            "artist": "NoCopyrightSounds",
        },
        {
            "url":    YT_AUDIO_LIBRARY_CHANNEL,
            "source": "YouTube Audio Library",
            "artist": "YouTube Audio Library",
        },
    ]

    # ─── 1 & 2: Try each channel with --flat-playlist ────────────────────
    active_source = CHANNEL_SOURCES[0]
    for src in CHANNEL_SOURCES:
        for cmd_prefix in (["yt-dlp"], [sys.executable, "-m", "yt_dlp"]):
            try:
                result = subprocess.run(
                    cmd_prefix + [
                        src["url"],
                        "--flat-playlist",
                        "--print", "%(id)s|||%(title)s",
                        "--playlist-items", "1-50",
                        "--no-warnings",
                    ] + _ytdlp_cookie_args(),
                    capture_output=True, text=True, timeout=60,
                )
                if result.returncode != 0 or not result.stdout.strip():
                    _log_ytdlp_failure(f"fetch_trending_song:{src['source']}", result)
                for line in result.stdout.strip().splitlines():
                    if "|||" in line:
                        vid_id, title = line.split("|||", 1)
                        vid_id = vid_id.strip()
                        title  = title.strip()
                        if vid_id and len(vid_id) >= 8:
                            entries.append({
                                "id":     vid_id,
                                "title":  title,
                                "source": src["source"],
                                "artist": src["artist"],
                            })
                if entries:
                    active_source = src
                    print(f"[fetch_trending_song] got {len(entries)} tracks from {src['source']}")
                    break
            except Exception as exc:
                print(f"[fetch_trending_song] {src['source']} fetch failed: {exc}")
        if entries:
            break

    # ─── 3: Search fallback (NOTE: use --no-playlist, NOT --flat-playlist) ─
    if not entries:
        print("[fetch_trending_song] channels failed — trying yt-dlp search ...")
        for query, src_name in [
            ("NoCopyrightSounds music free to use 2024", "NoCopyrightSounds"),
            ("youtube audio library free music no copyright", "YouTube Audio Library"),
        ]:
            for cmd_prefix in (["yt-dlp"], [sys.executable, "-m", "yt_dlp"]):
                try:
                    result = subprocess.run(
                        cmd_prefix + [
                            f"ytsearch30:{query}",
                            "--no-playlist",          # correct flag for search
                            "--print", "%(id)s|||%(title)s",
                            "--no-warnings",
                        ] + _ytdlp_cookie_args(),
                        capture_output=True, text=True, timeout=60,
                    )
                    if result.returncode != 0 or not result.stdout.strip():
                        _log_ytdlp_failure(f"fetch_trending_song:search:{src_name}", result)
                    for line in result.stdout.strip().splitlines():
                        if "|||" in line:
                            vid_id, title = line.split("|||", 1)
                            vid_id = vid_id.strip()
                            if vid_id and len(vid_id) >= 8:
                                entries.append({
                                    "id":     vid_id,
                                    "title":  title.strip(),
                                    "source": src_name,
                                    "artist": src_name,
                                })
                    if entries:
                        print(f"[fetch_trending_song] search got {len(entries)} results")
                        break
                except Exception as exc:
                    print(f"[fetch_trending_song] search failed: {exc}")
            if entries:
                break

    # ─── 4: Hardcoded emergency fallback  ────────────────────────────────
    # Known NCS tracks — public, no copyright, always available.
    if not entries:
        print("[fetch_trending_song] all sources failed — using emergency fallback tracks")
        entries = [
            {"id": "bM7SZ5SBzyY", "title": "Fade",               "source": "NoCopyrightSounds", "artist": "Alan Walker"},
            {"id": "y8XUlp4JhY4", "title": "Spectre",            "source": "NoCopyrightSounds", "artist": "Alan Walker"},
            {"id": "a-KKs05RMEM", "title": "Alone",              "source": "NoCopyrightSounds", "artist": "Alan Walker"},
            {"id": "TlBQH8M5mc4", "title": "Island",             "source": "NoCopyrightSounds", "artist": "Jarico"},
            {"id": "J2X5mJ3HDYE", "title": "Force",              "source": "NoCopyrightSounds", "artist": "Alan Walker"},
        ]

    # ─── Pick an unused track ─────────────────────────────────────────────
    random.shuffle(entries)
    chosen = None
    for entry in entries:
        # Key format: "{video_id}|{source}" — must match _save_used_song format
        key = f"{entry['id']}|{entry['source']}"
        if key not in used:
            chosen = entry
            break

    if not chosen:
        print("[fetch_trending_song] all tracks used — resetting tracker")
        USED_SONGS_DB.write_text("[]", encoding="utf-8")
        chosen = entries[0]

    state["trending_song_title"]    = chosen["title"]
    state["trending_song_artist"]   = chosen["artist"]
    state["audio_library_video_id"] = chosen["id"]

    state["music_metadata"] = {
        "title":                chosen["title"],
        "artist":               chosen["artist"],
        "source":               chosen["source"],
        "license":              "",   # filled by validate_music_license
        "license_url":          "",   # filled by validate_music_license
        "attribution_required": True, # filled by validate_music_license
        "attribution_text":     "",   # filled by validate_music_license
    }
    print(
        f"[fetch_trending_song] ✓ picked: {chosen['title']!r} "
        f"by {chosen['artist']} from {chosen['source']} (id={chosen['id']})"
    )
    return state


# ---------------------------------------------------------------------------
# Node 2 — download_trending_music (yt-dlp + chorus extraction)
# ---------------------------------------------------------------------------
def download_trending_music(state: VideoState) -> VideoState:
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    raw_mp3 = AUDIO_DIR / "asset_trending_music_full.mp3"
    chorus_mp3 = AUDIO_DIR / "asset_trending_music.mp3"

    state["music_path"] = ""
    state["music_attribution"] = ""

    video_id = state.get("audio_library_video_id", "")
    if not video_id:
        raise RuntimeError("No music video ID was selected.")

    command = [
        sys.executable, "-m", "yt_dlp",
        f"https://www.youtube.com/watch?v={video_id}",
        "--extract-audio",
        "--audio-format", "mp3",
        "--audio-quality", "0",
        "--output", str(AUDIO_DIR / "asset_trending_music_full.%(ext)s"),
        "--no-playlist",
        "--retries", "3",
        "--fragment-retries", "3",
        "--socket-timeout", "30",
        "--js-runtimes", "deno",        # use the deno runtime installed in CI
        "--remote-components", "ejs:github",  # download JS challenge solver from GitHub
    ] + _ytdlp_cookie_args()

    last_error = None

    for attempt in range(1, 4):
        # Never allow an earlier attempt's partial output to pass validation.
        for old in AUDIO_DIR.glob("asset_trending_music*"):
            if old.is_file():
                old.unlink()

        try:
            print(f"[download_trending_music] attempt {attempt}/3")
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=240,
            )
            if result.returncode != 0:
                _log_ytdlp_failure("download_trending_music", result)
                raise RuntimeError(
                    f"yt-dlp exited with code {result.returncode}"
                )

            # Require the actual postprocessed MP3, not a renamed .part file.
            info = _probe_media(raw_mp3)
            if "audio" not in info["stream_types"]:
                raise RuntimeError("Downloaded file has no audio stream.")

            _extract_chorus(
                raw_mp3,
                chorus_mp3,
                min(info["duration"], 60.0),
            )

            chorus_info = _probe_media(chorus_mp3)
            if "audio" not in chorus_info["stream_types"]:
                raise RuntimeError("Extracted chorus has no audio stream.")

            state["music_path"] = str(chorus_mp3)
            prepare_music_for_render(state)
            print(f"[download_trending_music] ready: {chorus_mp3}")
            return state

        except Exception as exc:
            last_error = exc
            print(f"[download_trending_music] attempt failed: {exc}")
            if attempt < 3:
                time.sleep(5 * attempt)

    raise RuntimeError(
        "Music download/extraction failed after three attempts. "
        "No video will be uploaded. Inspect the yt-dlp errors above."
    ) from last_error



# ---------------------------------------------------------------------------
# Node 3 — transcribe_music  (Whisper on the chorus audio -> timed lyrics)
# ---------------------------------------------------------------------------
def transcribe_music(state: VideoState) -> VideoState:
    prepare_music_for_render(state)
    state["caption_words"] = []

    print("[transcribe_music] transcribing chorus...")
    model = WhisperModel("small", device="cpu", compute_type="int8")

    segments, _ = model.transcribe(
        state["music_path"],
        word_timestamps=True,
        beam_size=5,
        condition_on_previous_text=False,
        vad_filter=False,
    )

    chorus_duration = _probe_media(Path(state["music_path"]))["duration"]
    words = []

    # segments is lazy: transcription errors can occur during iteration.
    for segment in segments:
        for word in segment.words or []:
            text = word.word.strip()
            start = max(0.0, float(word.start))
            end = min(chorus_duration, float(word.end))

            if text and start < end:
                words.append({
                    "text": text.upper(),
                    "start": start + MUSIC_START_OFFSET,
                    "end": end + MUSIC_START_OFFSET,
                })

    words.sort(key=lambda word: word["start"])

    if not words:
        # Instrumental track — no lyrics detected. Upload will proceed without
        # karaoke captions. The music badge still appears on the video.
        print("[transcribe_music] no lyrics detected (instrumental track) — skipping captions")


    state["caption_words"] = words
    print(f"[transcribe_music] {len(words)} caption words ready")
    return state

# ---------------------------------------------------------------------------
# Node 4 — assemble_video (asset video + song starting at 5s + Now Playing badge)
# ---------------------------------------------------------------------------
# Segment of the asset video to loop when the song is longer than the clip
LOOP_SEGMENT_START = 9.5   # seconds
LOOP_SEGMENT_END   = 10.0  # seconds


def _build_extended_video(asset_clip, total_duration: float):
    from moviepy import concatenate_videoclips

    asset_duration = float(asset_clip.duration)
    if asset_duration <= 0:
        raise RuntimeError("Asset video has an invalid duration.")

    if total_duration <= asset_duration:
        return asset_clip.subclipped(0, total_duration)

    loop_end = min(LOOP_SEGMENT_END, asset_duration)
    loop_start = min(
        LOOP_SEGMENT_START,
        max(0.0, loop_end - 0.5),
    )
    loop_segment = asset_clip.subclipped(loop_start, loop_end)

    if loop_segment.duration <= 0:
        raise RuntimeError("Cannot create a valid asset loop.")

    pieces = [asset_clip]
    remaining = total_duration - asset_duration

    while remaining > 0.000001:
        duration = min(loop_segment.duration, remaining)
        pieces.append(loop_segment.subclipped(0, duration))
        remaining -= duration

    return concatenate_videoclips(
        pieces, method="chain"
    ).with_duration(total_duration)


def assemble_video(state: VideoState) -> VideoState:
    state["final_video_path"] = ""

    # Mandatory checks: there is deliberately no raw-video fallback.
    metadata = prepare_music_for_render(state)
    words = state.get("caption_words", [])
    if not words:
        print("[assemble_video] no captions — rendering video without lyric overlay")

    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    destination = FINAL_DIR / "asset_final_video.mp4"
    temporary = FINAL_DIR / "asset_final_video.rendering.mp4"

    destination.unlink(missing_ok=True)
    temporary.unlink(missing_ok=True)

    caption_font = _ensure_caption_font()

    with ExitStack() as resources:
        asset = resources.enter_context(VideoFileClip(str(ASSET_VIDEO)))
        music = resources.enter_context(AudioFileClip(state["music_path"]))

        total_duration = MUSIC_START_OFFSET + music.duration
        if MUSIC_START_OFFSET < 0 or music.duration <= 0:
            raise RuntimeError("Invalid music duration or start offset.")

        scale = max(1080 / asset.w, 1920 / asset.h)
        visual = asset.without_audio().resized(scale)
        visual = visual.cropped(
            x_center=visual.w / 2,
            y_center=visual.h / 2,
            width=1080,
            height=1920,
        )
        visual = _build_extended_video(visual, total_duration)
        resources.callback(visual.close)

        audio_tracks = []

        # Keep original sound only during the intro so it cannot mask music.
        if asset.audio is not None:
            intro_duration = min(
                MUSIC_START_OFFSET,
                asset.duration,
                asset.audio.duration,
            )
            if intro_duration > 0:
                audio_tracks.append(
                    asset.audio.subclipped(0, intro_duration)
                )

        audio_tracks.append(
            music.with_effects([
                afx.MultiplyVolume(MUSIC_VOLUME),
            ]).with_start(MUSIC_START_OFFSET)
        )

        # Composition applies the five-second offset even without asset audio.
        mixed_audio = CompositeAudioClip(audio_tracks).with_duration(
            total_duration
        )
        resources.callback(mixed_audio.close)

        overlays = []
        caption_count = 0
        box_width = int(1080 * 0.82)
        box_x = (1080 - box_width) // 2
        box_y = int(1920 * 0.46)

        for index, word in enumerate(words):
            start = max(0.0, float(word["start"]))
            end = float(word["end"])

            if index + 1 < len(words):
                next_start = float(words[index + 1]["start"])
                if 0 < next_start - start <= 1.5:
                    end = next_start

            end = min(total_duration, max(end, start + 0.18))
            if start >= end:
                continue

            caption = (
                TextClip(
                    text=word["text"],
                    font=caption_font,
                    font_size=88,
                    color="white",
                    stroke_color="black",
                    stroke_width=6,
                    text_align="center",
                    size=(box_width, None),
                    method="caption",
                )
                .with_start(start)
                .with_duration(end - start)
                .with_position((box_x, box_y))
            )
            resources.callback(caption.close)
            overlays.append(caption)
            caption_count += 1

        if caption_count == 0:
            print("[assemble_video] no captions rendered — video will have no lyric overlay")


        badges = _render_attribution_badge(
            visual,
            metadata,
            MUSIC_START_OFFSET,
            total_duration,
            caption_font,
        )
        if not badges:
            raise RuntimeError("Music badge rendering failed.")

        for badge in badges:
            resources.callback(badge.close)
        overlays.extend(badges)

        final = (
            CompositeVideoClip(
                [visual, *overlays],
                size=(1080, 1920),
            )
            .with_duration(total_duration)
            .with_audio(mixed_audio)
        )
        resources.callback(final.close)

        print(
            f"[assemble_video] rendering {total_duration:.2f}s, "
            f"{caption_count} caption words, music starts at "
            f"{MUSIC_START_OFFSET:.2f}s"
        )

        final.write_videofile(
            str(temporary),
            fps=30,
            codec="libx264",
            audio_codec="aac",
            audio_bitrate="192k",
            bitrate=VIDEO_BITRATE,
            preset=VIDEO_PRESET,
            ffmpeg_params=VIDEO_FFMPEG_PARAMS,
        )

    # Only expose the final upload path after the export passes validation.
    info = _probe_media(temporary)
    if not {"video", "audio"}.issubset(info["stream_types"]):
        raise RuntimeError("Rendered output is missing video or audio.")

    if abs(info["duration"] - total_duration) > 0.5:
        raise RuntimeError(
            f"Rendered duration mismatch: expected {total_duration:.2f}s, "
            f"got {info['duration']:.2f}s."
        )

    subprocess.run(
        [
            "ffmpeg", "-v", "error", "-xerror",
            "-i", str(temporary),
            "-map", "0:a:0",
            "-f", "null", "-",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )

    temporary.replace(destination)
    state["final_video_path"] = str(destination)

    # Mark the track used only after a successful render.
    _save_used_song(
        state["audio_library_video_id"],
        metadata["source"],
    )

    # Write shared manifest so instagram_workflow can reuse this audio
    # without a second yt-dlp download (avoids bot-block on GitHub runners).
    try:
        manifest = {
            "music_path":             state["music_path"],
            "music_attribution":      state.get("music_attribution", ""),
            "music_metadata":         state.get("music_metadata", {}),
            "trending_song_title":    state.get("trending_song_title", ""),
            "trending_song_artist":   state.get("trending_song_artist", ""),
            "audio_library_video_id": state.get("audio_library_video_id", ""),
            "caption_words":          state.get("caption_words", []),
        }
        SHARED_AUDIO_MANIFEST.write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        print(f"[assemble_video] shared manifest written -> {SHARED_AUDIO_MANIFEST}")
    except Exception as exc:
        print(f"[assemble_video] could not write shared manifest: {exc}")

    print(f"[assemble_video] validated output: {destination}")
    return state


# ---------------------------------------------------------------------------
# Node 4 — upload_to_youtube
# ---------------------------------------------------------------------------
def _get_credentials(scope: str) -> Credentials:
    refresh = (
        os.getenv("YOUTUBE_REFRESH_TOKEN")
        if "youtube" in scope
        else os.getenv("DRIVE_REFRESH_TOKEN")
    )
    creds = Credentials(
        token=None, refresh_token=refresh,
        client_id=os.getenv("GOOGLE_CLIENT_ID"),
        client_secret=os.getenv("GOOGLE_CLIENT_SECRET"),
        token_uri="https://oauth2.googleapis.com/token",
        scopes=[scope],
    )
    creds.refresh(Request())
    return creds


def _run_resumable_upload(request, label: str):
    for retry in range(8):
        try:
            status, response = request.next_chunk()
            if status:
                print(f"[{label}] {int(status.progress() * 100)}%")
            if response is not None:
                return response
        except (HttpError, httplib2.HttpLib2Error, IOError,
                http.client.HTTPException, ConnectionError, TimeoutError):
            if retry == 7:
                raise
            time.sleep(min(2 ** (retry + 1) + random.random(), 60))
    raise RuntimeError(f"{label} upload did not return a response")


def upload_to_youtube(state: VideoState) -> VideoState:
    yt = build("youtube", "v3",
               credentials=_get_credentials(
                   "https://www.googleapis.com/auth/youtube.upload"))
    # Use a generic title — song name in title triggers Content ID copyright blocks
    # #Shorts MUST be in the title — YouTube uses this as a Short classification signal
    generic_titles = [
        "This transition hits different 🔥 #Shorts",
        "POV: you found this 👀 #Shorts #viral",
        "Wait for it... 🤯 #Shorts #viral",
        "The vibe is immaculate ✨ #Shorts",
        "Not me watching this 10 times 🔄 #Shorts",
        "This one goes hard 🎵 #Shorts",
        "The transition that broke the internet 🔥 #Shorts",
        "Stop scrolling ✋ #Shorts #viral",
    ]
    import random as _random
    title = _random.choice(generic_titles)
    description = "Follow for more! 🔔\n\n#shorts #viral #trending #fyp"
    attribution = state.get("music_attribution", "")
    if attribution:
        description += f"\n\n{attribution}"
    request = yt.videos().insert(
        part="snippet,status",
        body={
            "snippet": {
                "title": title,
                "description": description,
                "tags": ["shorts", "viral", "trending", "fyp", "foryou", "transition"],
            },
            "status": {"privacyStatus": "public"},
        },
        media_body=MediaFileUpload(
            state["final_video_path"],
            chunksize=4 * 1024 * 1024, resumable=True, mimetype="video/mp4",
        ),
    )
    response = _run_resumable_upload(request, "youtube")
    state["youtube_url"] = f"https://youtube.com/shorts/{response['id']}"
    print(f"[upload_to_youtube] {state['youtube_url']}")
    return state




# ---------------------------------------------------------------------------
# Build LangGraph
# ---------------------------------------------------------------------------
_NODES = [
    ("fetch_trending_song",     fetch_trending_song),
    ("download_trending_music", download_trending_music),
    ("transcribe_music",        transcribe_music),
    ("assemble_video",          assemble_video),
    ("upload_to_youtube",       upload_to_youtube),
]

graph = StateGraph(VideoState)
for name, fn in _NODES:
    graph.add_node(name, fn)

_node_names = [n for n, _ in _NODES]
graph.add_edge(START, _node_names[0])
for src, dst in zip(_node_names, _node_names[1:]):
    graph.add_edge(src, dst)
graph.add_edge(_node_names[-1], END)
workflow = graph.compile()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    validate_prerequisites()
    result = workflow.invoke({
        "trending_song_title":    "",
        "trending_song_artist":   "",
        "audio_library_video_id": "",
        "music_path":             "",
        "music_attribution":      "",
        "music_metadata":         {},
        "caption_words":          [],
        "final_video_path":       "",
        "youtube_url":            "",
    })
    print("\n--- DONE ---")
    for key in ("trending_song_title", "trending_song_artist",
                "final_video_path", "youtube_url"):
        print(f"  {key}: {result.get(key, '')}")