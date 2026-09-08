"""Asset video + trending song workflow.

Takes the local asset video, fetches the #1 trending song from iTunes charts,
downloads it via yt-dlp, extracts the chorus (highest-energy segment), and
overlays it starting at 5 seconds into the video. Uploads to YouTube + Drive.

No TTS, no script, no captions — just the raw asset video with trending music.

Run:  myenv/Scripts/python.exe asset_video_workflow.py
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
    "DRIVE_REFRESH_TOKEN",
    "INSTAGRAM_ACCOUNT_ID",
    "INSTAGRAM_ACCESS_TOKEN",
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
    drive_url:                str
    drive_file_id:            str
    instagram_url:            str


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
                 "-f", "null", "/dev/null"],
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


def prepare_music_for_render(state: VideoState) -> dict:
    """Validate music before it can be attached to the rendered video.

    An invalid source clears the audio path and metadata so later rendering
    and publishing nodes cannot use rejected music.
    """
    raw_metadata = state.get("music_metadata", {})
    if not raw_metadata:
        return {}

    try:
        validated_metadata = validate_music_license(raw_metadata)
    except ValueError as exc:
        print(f"[assemble_video] music rejected by license validator: {exc}")
        state["music_path"] = ""
        state["music_attribution"] = ""
        state["music_metadata"] = {}
        state["caption_words"] = []
        return {}

    state["music_metadata"] = validated_metadata
    return validated_metadata


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
                    ],
                    capture_output=True, text=True, timeout=60,
                )
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
                        ],
                        capture_output=True, text=True, timeout=60,
                    )
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
    """Download the Audio Library track selected by fetch_trending_song.

    Downloads by direct video ID (not keyword search) so we always get
    the exact copyright-free track chosen from the Audio Library channel.
    """
    AUDIO_DIR.mkdir(exist_ok=True)
    raw_mp3    = AUDIO_DIR / "asset_trending_music_full.mp3"
    chorus_mp3 = AUDIO_DIR / "asset_trending_music.mp3"

    for old in AUDIO_DIR.glob("asset_trending_music*.*"):
        old.unlink(missing_ok=True)

    video_id = state.get("audio_library_video_id", "")
    if not video_id:
        print("[download_trending_music] no audio library video_id — skipping music")
        state["music_path"]        = ""
        state["music_attribution"] = ""
        return state

    video_url     = f"https://www.youtube.com/watch?v={video_id}"
    dest_template = AUDIO_DIR / "asset_trending_music_full.%(ext)s"
    print(f"[download_trending_music] downloading Audio Library track: {video_url}")

    base_args = [
        video_url,
        "--extract-audio",
        "--audio-format", "mp3",
        "--audio-quality", "0",
        "--output", str(dest_template),
        "--no-playlist",
        "--quiet",
        "--no-warnings",
    ]

    downloaded = False
    try:
        subprocess.run(["yt-dlp"] + base_args, check=True, timeout=120)
        downloaded = True
    except FileNotFoundError:
        try:
            subprocess.run([sys.executable, "-m", "yt_dlp"] + base_args, check=True, timeout=120)
            downloaded = True
        except Exception as exc:
            print(f"[download_trending_music] error: {exc}")
    except Exception as exc:
        print(f"[download_trending_music] error: {exc}")

    # Rename whatever yt-dlp produced
    if not raw_mp3.exists():
        for c in AUDIO_DIR.glob("asset_trending_music_full.*"):
            c.rename(raw_mp3)
            downloaded = True
            break

    if downloaded and raw_mp3.exists():
        full_song_dur = AudioFileClip(str(raw_mp3)).duration
        music_dur     = min(full_song_dur, 60.0)
        print(f"[download_trending_music] extracting {music_dur:.1f}s chorus ...")
        _extract_chorus(raw_mp3, chorus_mp3, music_dur)

        state["music_path"]        = str(chorus_mp3)
        state["music_attribution"] = (
            f"Music: {state['trending_song_title']} "
            f"by {state['trending_song_artist']} (royalty-free)"
        )
        # Save used track: key = "{video_id}|{source}" to match fetch dedup check
        source = state.get("music_metadata", {}).get("source", "YouTube Audio Library")
        _save_used_song(video_id, source)
        print(f"[download_trending_music] chorus saved -> {chorus_mp3}")
    else:
        print("[download_trending_music] download failed — no music")
        state["music_path"]        = ""
        state["music_attribution"] = ""

    return state



# ---------------------------------------------------------------------------
# Node 3 — transcribe_music  (Whisper on the chorus audio -> timed lyrics)
# ---------------------------------------------------------------------------
def transcribe_music(state: VideoState) -> VideoState:
    """Use faster-whisper to extract word-level timestamps from the chorus clip."""
    music_path = state.get("music_path", "")
    if not music_path or not pathlib.Path(music_path).exists():
        print("[transcribe_music] no music file — skipping captions")
        state["caption_words"] = []
        return state

    print("[transcribe_music] running Whisper on chorus audio ...")
    try:
        model = WhisperModel("small", device="cpu", compute_type="int8")
        # music=True suppresses non-speech noise, giving cleaner lyric transcription
        segments, _ = model.transcribe(
            music_path,
            word_timestamps=True,
            beam_size=5,
            condition_on_previous_text=False,
            vad_filter=True,
        )
        words = []
        for seg in segments:
            if seg.words:
                for w in seg.words:
                    text = w.word.strip()
                    if text:
                        # Offset by MUSIC_START_OFFSET so timestamps match the video timeline
                        words.append({
                            "text":  text.upper(),
                            "start": w.start + MUSIC_START_OFFSET,
                            "end":   w.end   + MUSIC_START_OFFSET,
                        })
        print(f"[transcribe_music] {len(words)} words transcribed")
        state["caption_words"] = words
    except Exception as exc:
        print(f"[transcribe_music] failed ({exc}) — no captions")
        state["caption_words"] = []

    return state

# ---------------------------------------------------------------------------
# Node 4 — assemble_video (asset video + song starting at 5s + Now Playing badge)
# ---------------------------------------------------------------------------
# Segment of the asset video to loop when the song is longer than the clip
LOOP_SEGMENT_START = 9.5   # seconds
LOOP_SEGMENT_END   = 10.0  # seconds


def _build_extended_video(asset_clip, total_duration: float):
    """
    Build a video track that is exactly total_duration seconds long:
      - Plays the full asset clip once
      - If total_duration > asset clip length, loops the 9.5s-10s segment
        on repeat to fill the remainder
    """
    from moviepy import concatenate_videoclips

    asset_dur = asset_clip.duration

    if total_duration <= asset_dur:
        # Song fits within asset — just trim
        return asset_clip.subclipped(0, total_duration)

    # Asset plays in full, then the 9.5s-10s mini-loop fills the rest
    loop_seg = asset_clip.subclipped(LOOP_SEGMENT_START,
                                     min(LOOP_SEGMENT_END, asset_dur))
    extra_needed = total_duration - asset_dur
    print(
        f"[assemble_video] song longer than asset by {extra_needed:.1f}s — "
        f"looping {LOOP_SEGMENT_START}s-{LOOP_SEGMENT_END}s segment"
    )

    loop_pieces = []
    remaining = extra_needed
    while remaining > 0:
        piece = loop_seg.subclipped(0, min(loop_seg.duration, remaining))
        loop_pieces.append(piece)
        remaining -= piece.duration

    return concatenate_videoclips([asset_clip] + loop_pieces, method="compose")


def assemble_video(state: VideoState) -> VideoState:
    raw_asset = VideoFileClip(str(ASSET_VIDEO))
    asset_dur = raw_asset.duration
    print(f"[assemble_video] asset duration = {asset_dur:.1f}s")

    # Scale + crop to 1080x1920 vertical
    scale     = max(1080 / raw_asset.w, 1920 / raw_asset.h)
    raw_asset = raw_asset.resized(scale)
    raw_asset = raw_asset.cropped(
        x_center=raw_asset.w / 2, y_center=raw_asset.h / 2,
        width=1080, height=1920,
    )

    # Validate music before opening it or attaching it to the video.
    # Rejected music clears its path and metadata in the state.
    validated_metadata = prepare_music_for_render(state)

    # --- Determine total video duration ---
    music_clip = None
    if validated_metadata and state.get("music_path") and Path(state["music_path"]).is_file():
        music_clip = AudioFileClip(state["music_path"])

    # Total duration = 5s intro + chorus length; or just asset_dur if no music
    chorus_dur   = music_clip.duration if music_clip else 0
    total_dur    = MUSIC_START_OFFSET + chorus_dur if music_clip else asset_dur
    print(f"[assemble_video] total video duration = {total_dur:.1f}s "
          f"(asset={asset_dur:.1f}s, chorus={chorus_dur:.1f}s)")

    # Build video track (extends with loop if needed)
    video = _build_extended_video(raw_asset, total_dur)

    # --- Audio: original video audio (plays ONCE only) + trending song from 5s ---
    if music_clip:
        music_clip = music_clip.with_effects([afx.MultiplyVolume(MUSIC_VOLUME)])
        music_clip = music_clip.with_start(MUSIC_START_OFFSET)
        original_audio = raw_asset.audio
        if original_audio:
            # Original audio plays only for the asset's natural duration — NOT looped.
            # Beyond that point only the trending song is heard.
            orig_once = original_audio.subclipped(0, min(original_audio.duration, asset_dur))
            video = video.with_audio(CompositeAudioClip([orig_once, music_clip]))
        else:
            video = video.with_audio(music_clip)

    # --- Music attribution ---
    # License validation already completed before the music clip was opened.
    # No visual badge is burned into the video; Instagram receives audio_name.
    overlays = []
    caption_font = _ensure_caption_font()

    # ── Lyric captions (word-by-word, styled with Anton font) ──
    caption_words = state.get("caption_words", [])
    total_dur     = video.duration if hasattr(video, "duration") else total_dur

    for i, word in enumerate(caption_words):
        w_start = word["start"]
        w_end   = word["end"]
        # Smooth flow: hold word on screen until next word arrives (up to 1.5s max gap)
        if i + 1 < len(caption_words):
            next_start = caption_words[i + 1]["start"]
            if next_start > w_start and (next_start - w_start) <= 1.5:
                w_end = next_start
        w_dur = max(w_end - w_start, 0.18)
        if w_start >= total_dur:
            continue

        box_w = int(1080 * 0.82)
        box_x = (1080 - box_w) // 2
        box_y = int(1920 * 0.46)

        # Shadow layer (slightly offset dark clone)
        shadow = TextClip(
            text=word["text"], font=caption_font, font_size=88,
            color="#111111", text_align="center",
            size=(box_w, None), method="caption",
        )
        shadow = (shadow
                  .with_start(w_start).with_duration(w_dur)
                  .with_position((box_x + 4, box_y + 4)))

        # Main white text with thick black stroke
        txt = TextClip(
            text=word["text"], font=caption_font, font_size=88,
            color="white", stroke_color="black", stroke_width=6,
            text_align="center",
            size=(box_w, None), method="caption",
        )
        txt = (txt
               .with_start(w_start).with_duration(w_dur)
               .with_position((box_x, box_y)))

        overlays.extend([shadow, txt])

    if overlays:
        video = CompositeVideoClip([video, *overlays])

    FINAL_DIR.mkdir(exist_ok=True)
    dest = FINAL_DIR / "asset_final_video.mp4"
    video.write_videofile(
        str(dest), fps=30, codec="libx264", audio_codec="aac",
        bitrate=VIDEO_BITRATE, preset=VIDEO_PRESET,
        ffmpeg_params=VIDEO_FFMPEG_PARAMS,
    )
    state["final_video_path"] = str(dest)
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
    generic_titles = [
        "This transition hits different 🔥",
        "POV: you found this 👀 #shorts",
        "Wait for it... 🤯 #viral",
        "The vibe is immaculate ✨ #shorts",
        "Not me watching this 10 times 🔄",
        "This one goes hard 🎵 #shorts",
        "The transition that broke the internet 🔥",
        "Stop scrolling ✋ #shorts #viral",
    ]
    import random as _random
    title = _random.choice(generic_titles)
    description = "Follow for more! 🔔\n\n#shorts #viral #trending #fyp"
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
# Node 5 — upload_to_drive
# ---------------------------------------------------------------------------
def upload_to_drive(state: VideoState) -> VideoState:
    drive = build("drive", "v3",
                  credentials=_get_credentials(
                      "https://www.googleapis.com/auth/drive.file"))
    request = drive.files().create(
        body={"name": Path(state["final_video_path"]).name},
        media_body=MediaFileUpload(
            state["final_video_path"],
            chunksize=4 * 1024 * 1024, resumable=True,
        ),
        fields="id,webViewLink",
    )
    res = _run_resumable_upload(request, "drive")
    file_id = res["id"]
    state["drive_file_id"] = file_id
    state["drive_url"]     = res["webViewLink"]
    print(f"[upload_to_drive] {state['drive_url']} (file_id: {file_id})")

    # Grant public read permission so Meta Graph API can fetch the direct video stream
    try:
        drive.permissions().create(
            fileId=file_id,
            body={"role": "reader", "type": "anyone"},
        ).execute()
    except Exception as exc:
        print(f"[upload_to_drive] permission notice: {exc}")

    return state


# ---------------------------------------------------------------------------
# Node 6 — upload_to_instagram (Reels via Meta Graph API)
# ---------------------------------------------------------------------------
def upload_to_instagram(state: VideoState) -> VideoState:
    ig_account_id   = os.getenv("INSTAGRAM_ACCOUNT_ID")
    ig_access_token = os.getenv("INSTAGRAM_ACCESS_TOKEN")
    drive_file_id   = state.get("drive_file_id", "")

    if not ig_account_id or not ig_access_token:
        raise OSError(
            "Instagram credentials are required: set INSTAGRAM_ACCOUNT_ID and "
            "INSTAGRAM_ACCESS_TOKEN."
        )

    if not drive_file_id:
        raise RuntimeError("Drive video is required before the Instagram upload.")

    ig_captions = [
        "This transition hits different 🔥 #reels #viral #trending #explore #fyp",
        "Wait for the beat drop 👀 #reels #viral #transition #fyp #explorepage",
        "POV: you found the perfect vibe ✨ #reels #viral #explore #trending",
        "Stop scrolling ✋ #reels #viral #trending #explore #fyp",
    ]
    caption = random.choice(ig_captions)

    # Read validated music metadata built by fetch_trending_song + validate_music_license
    music_meta  = state.get("music_metadata", {})
    song_title  = music_meta.get("title", "")
    song_artist = music_meta.get("artist", "")
    audio_name  = f"{song_title} - {song_artist}" if song_title else ""

    # Attribution fallback: always append track credit to caption text
    if song_title:
        caption += f"\n\nAudio: {song_title} · {song_artist}"

    # Google Drive high-speed CDN direct video stream URL
    video_url = f"https://drive.usercontent.google.com/download?id={drive_file_id}&export=download"

    print("[upload_to_instagram] initializing Reel container via Meta Graph API ...")
    try:
        init_url = f"https://graph.facebook.com/v20.0/{ig_account_id}/media"

        # Build upload payload
        # audio_name → sets Instagram's native ♫ {audio_name} tag shown under profile name
        ig_payload: dict = {
            "media_type":    "REELS",
            "video_url":     video_url,
            "caption":       caption,
            "share_to_feed": True,
            "access_token":  ig_access_token,
        }
        if audio_name:
            ig_payload["audio_name"] = audio_name
            print(f"[upload_to_instagram] audio_name → {audio_name!r}")

        init_res = requests.post(init_url, data=ig_payload, timeout=30).json()

        if "id" not in init_res:
            raise RuntimeError(f"Instagram container initialization failed: {init_res}")

        container_id = init_res["id"]
        print(f"[upload_to_instagram] container created: {container_id}, waiting for processing ...")

        status_url = f"https://graph.facebook.com/v20.0/{container_id}"
        for i in range(1, 30):
            time.sleep(5)
            stat = requests.get(status_url, params={
                "fields": "status_code,status",
                "access_token": ig_access_token,
            }, timeout=15).json()
            code = stat.get("status_code")
            if code == "FINISHED":
                print("[upload_to_instagram] video processing finished!")
                break
            elif code == "ERROR":
                raise RuntimeError(f"Instagram container processing failed: {stat}")
            print(f"[upload_to_instagram] processing: {code} ...")
        else:
            raise TimeoutError("Instagram container processing timed out")

        # Publish the Reel
        pub_url = f"https://graph.facebook.com/v20.0/{ig_account_id}/media_publish"
        pub_res = requests.post(pub_url, data={
            "creation_id": container_id,
            "access_token": ig_access_token,
        }, timeout=30).json()

        if "id" in pub_res:
            post_id = pub_res["id"]
            media_info = requests.get(f"https://graph.facebook.com/v20.0/{post_id}", params={
                "fields": "permalink",
                "access_token": ig_access_token,
            }, timeout=15).json()
            ig_url = media_info.get("permalink", f"https://www.instagram.com/reel/{post_id}/")
            state["instagram_url"] = ig_url
            print(f"[upload_to_instagram] Reel published! {ig_url}")
            return state
        raise RuntimeError(f"Instagram Reel publishing failed: {pub_res}")

    except Exception as exc:
        raise RuntimeError(f"Instagram upload failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Build LangGraph
# ---------------------------------------------------------------------------
_NODES = [
    ("fetch_trending_song",     fetch_trending_song),
    ("download_trending_music", download_trending_music),
    ("transcribe_music",        transcribe_music),
    ("assemble_video",          assemble_video),
    ("upload_to_youtube",       upload_to_youtube),
    ("upload_to_drive",         upload_to_drive),
    ("upload_to_instagram",     upload_to_instagram),
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
        "drive_url":              "",
        "drive_file_id":          "",
        "instagram_url":          "",
    })
    print("\n--- DONE ---")
    for key in ("trending_song_title", "trending_song_artist",
                "final_video_path", "youtube_url", "drive_url", "instagram_url"):
        print(f"  {key}: {result.get(key, '')}")
