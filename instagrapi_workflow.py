"""instagrapi_workflow.py — Instagram Reels via Instagram's private mobile API.

Uses instagrapi instead of Meta Graph API so we can attach a REAL tappable
licensed song from Instagram's music catalog — exactly like posting from the
Instagram app. No manual steps required.

Pipeline:
  fetch_trending_song -> download_trending_music -> transcribe_music
  -> assemble_video -> search_instagram_music -> upload_via_instagrapi

Required GitHub secrets:
  INSTAGRAM_USERNAME      — Instagram account username (e.g. myaccount)
  INSTAGRAM_PASSWORD      — Instagram account password
  INSTAGRAM_SESSION_JSON  — base64-encoded instagrapi session (auto-created
                            on first successful run — see README)
  YOUTUBE_COOKIES         — yt-dlp Netscape cookie file contents

Run:  python instagrapi_workflow.py
"""

import base64
import functools
import json
import os
import random
import subprocess
import sys
import time
from contextlib import ExitStack
from pathlib import Path
from typing import TypedDict

import requests
from dotenv import load_dotenv
from faster_whisper import WhisperModel
from instagrapi import Client
from instagrapi.exceptions import (
    BadPassword,
    ChallengeRequired,
    LoginRequired,
    TwoFactorRequired,
)
from langgraph.graph import END, START, StateGraph
from moviepy import (
    afx,
    AudioFileClip,
    ColorClip,
    CompositeAudioClip,
    CompositeVideoClip,
    TextClip,
    VideoFileClip,
)

load_dotenv()

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR          = Path(__file__).resolve().parent
ASSET_VIDEO       = BASE_DIR / "assets" / "the_transition_i_n_between_is.mp4"
AUDIO_DIR         = BASE_DIR / "output_audio"
FINAL_DIR         = BASE_DIR / "output_final_video"
FONT_PATH         = BASE_DIR / "assets" / "Roboto-Bold.ttf"
USED_SONGS_DB     = BASE_DIR / "assets" / "used_songs.json"
CAPTION_FONT_PATH = BASE_DIR / "assets" / "Anton-Regular.ttf"
SESSION_FILE      = BASE_DIR / ".instagrapi_session.json"

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
MUSIC_VOLUME        = 0.20
MUSIC_START_OFFSET  = 5.0
VIDEO_BITRATE       = "8000k"
VIDEO_PRESET        = "slow"
VIDEO_FFMPEG_PARAMS = ["-crf", "18", "-movflags", "+faststart"]
YT_AUDIO_LIBRARY_CHANNEL = "https://www.youtube.com/@YouTubeAudioLibrary/videos"

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
}

REQUIRED_ENV_VARS = [
    "INSTAGRAM_USERNAME",
    "INSTAGRAM_PASSWORD",
]

# Trending music search queries — instagrapi searches Instagram's music catalog
TRENDING_MUSIC_QUERIES = [
    "trending 2024",
    "viral pop 2024",
    "viral hits",
    "trending music",
    "popular songs",
]


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
class VideoState(TypedDict):
    trending_song_title:    str
    trending_song_artist:   str
    audio_library_video_id: str
    music_path:             str
    music_attribution:      str
    music_metadata:         dict
    caption_words:          list
    final_video_path:       str
    ig_music_id:            str   # Instagram music asset ID (from catalog search)
    ig_music_title:         str   # display title of the attached IG music
    instagram_url:          str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _ensure_caption_font() -> str:
    if CAPTION_FONT_PATH.exists():
        return str(CAPTION_FONT_PATH)
    ttf_url = "https://github.com/google/fonts/raw/main/ofl/anton/Anton-Regular.ttf"
    try:
        print("[font] downloading Anton-Regular.ttf ...")
        r = requests.get(ttf_url, timeout=15)
        r.raise_for_status()
        CAPTION_FONT_PATH.parent.mkdir(parents=True, exist_ok=True)
        CAPTION_FONT_PATH.write_bytes(r.content)
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
    stderr = (result.stderr or "").strip()
    stdout = (result.stdout or "").strip()
    print(f"[{label}] yt-dlp exited {result.returncode}")
    if stderr:
        print(f"[{label}] stderr: {stderr[-1500:]}")
    elif stdout:
        print(f"[{label}] stdout: {stdout[-1500:]}")


def _load_used_songs() -> set:
    if USED_SONGS_DB.is_file():
        try:
            return set(json.loads(USED_SONGS_DB.read_text(encoding="utf-8")))
        except Exception:
            pass
    return set()


def _save_used_song(video_id: str, source: str) -> None:
    used = _load_used_songs()
    used.add(f"{video_id}|{source}")
    USED_SONGS_DB.parent.mkdir(parents=True, exist_ok=True)
    USED_SONGS_DB.write_text(json.dumps(sorted(used), indent=2), encoding="utf-8")


def _extract_chorus(src_path: Path, dest_path: Path, duration: float) -> None:
    try:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(src_path)],
            capture_output=True, text=True, check=True, timeout=15,
        )
        total_dur  = float(probe.stdout.strip())
        step       = max(5.0, (total_dur - duration) / 8)
        best_start = max(0.0, total_dur * 0.30 - duration / 2)
        best_vol   = float("-inf")
        t          = 0.0
        while t + duration <= total_dur:
            res = subprocess.run(
                ["ffmpeg", "-y", "-ss", str(t), "-t", str(min(duration, 15.0)),
                 "-i", str(src_path), "-af", "volumedetect", "-f", "null", "-"],
                capture_output=True, text=True, timeout=30,
            )
            for line in res.stderr.splitlines():
                if "mean_volume" in line:
                    try:
                        vol = float(line.split("mean_volume:")[1].split("dB")[0].strip())
                        if vol > best_vol:
                            best_vol, best_start = vol, t
                    except ValueError:
                        pass
            t += step
        print(f"[chorus] loudest window at {best_start:.1f}s ({best_vol:.1f} dB)")
        subprocess.run(
            ["ffmpeg", "-y", "-ss", str(best_start), "-t", str(duration),
             "-i", str(src_path), "-b:a", "192k", str(dest_path)],
            check=True, capture_output=True, timeout=60,
        )
    except Exception as exc:
        print(f"[chorus] scan failed ({exc}), using 30%%-mark fallback")
        try:
            probe = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", str(src_path)],
                capture_output=True, text=True, timeout=15,
            )
            total_dur = float(probe.stdout.strip()) if probe.returncode == 0 else duration * 3
            start     = max(0.0, min(total_dur * 0.30, total_dur - duration))
        except Exception:
            start = 0.0
        subprocess.run(
            ["ffmpeg", "-y", "-ss", str(start), "-t", str(duration),
             "-i", str(src_path), "-b:a", "192k", str(dest_path)],
            check=True, capture_output=True, timeout=60,
        )


def validate_music_license(metadata: dict) -> dict:
    source = metadata.get("source", "")
    policy = MUSIC_LICENSE_POLICIES.get(source)
    if not policy:
        raise ValueError(
            f"Unverified music source {source!r}. "
            f"Accepted: {list(MUSIC_LICENSE_POLICIES)}"
        )
    validated = dict(metadata)
    validated["license"]              = policy["license"]
    validated["license_url"]          = policy["license_url"]
    validated["attribution_required"] = policy["attribution_required"]
    if validated["attribution_required"] and not validated.get("attribution_text"):
        validated["attribution_text"] = (
            f'Music: "{validated["title"]}" by {validated["artist"]} '
            f'| {source} — {policy["license_url"]}'
        )
    return validated


def _probe_media(path: Path) -> dict:
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"Missing or empty media file: {path}")
    result = subprocess.run(
        ["ffprobe", "-v", "error",
         "-show_entries", "format=duration:stream=codec_type",
         "-of", "json", str(path)],
        capture_output=True, text=True, check=True, timeout=30,
    )
    info     = json.loads(result.stdout)
    duration = float(info.get("format", {}).get("duration", 0))
    if not 0 < duration < float("inf"):
        raise RuntimeError(f"Invalid media duration: {path}")
    return {
        "duration":     duration,
        "stream_types": {s.get("codec_type") for s in info.get("streams", [])},
    }


def prepare_music_for_render(state: VideoState) -> dict:
    metadata = state.get("music_metadata")
    if not metadata:
        raise RuntimeError("Music metadata is missing.")
    validated  = validate_music_license(metadata)
    music_path = state.get("music_path")
    if not music_path:
        raise RuntimeError("Music download is missing.")
    info = _probe_media(Path(music_path))
    if "audio" not in info["stream_types"]:
        raise RuntimeError("Downloaded music has no audio stream.")
    state["music_metadata"]    = validated
    state["music_attribution"] = validated.get("attribution_text", "")
    return validated


# ---------------------------------------------------------------------------
# Node 1 — fetch_trending_song
# ---------------------------------------------------------------------------
def fetch_trending_song(state: VideoState) -> VideoState:
    print("[fetch_trending_song] fetching YouTube Audio Library tracks ...")
    used, entries = _load_used_songs(), []

    CHANNEL_SOURCES = [
        {"url": YT_AUDIO_LIBRARY_CHANNEL,
         "source": "YouTube Audio Library", "artist": "YouTube Audio Library"},
    ]

    for src in CHANNEL_SOURCES:
        for cmd_prefix in (["yt-dlp"], [sys.executable, "-m", "yt_dlp"]):
            try:
                result = subprocess.run(
                    cmd_prefix + [src["url"], "--flat-playlist",
                                  "--print", "%(id)s|||%(title)s",
                                  "--playlist-items", "1-50", "--no-warnings"]
                    + _ytdlp_cookie_args(),
                    capture_output=True, text=True, timeout=60,
                )
                if result.returncode != 0 or not result.stdout.strip():
                    _log_ytdlp_failure(f"fetch:{src['source']}", result)
                for line in result.stdout.strip().splitlines():
                    if "|||" in line:
                        vid_id, title = line.split("|||", 1)
                        vid_id = vid_id.strip()
                        if vid_id and len(vid_id) >= 8:
                            entries.append({"id": vid_id, "title": title.strip(),
                                           "source": src["source"], "artist": src["artist"]})
                if entries:
                    print(f"[fetch_trending_song] got {len(entries)} tracks")
                    break
            except Exception as exc:
                print(f"[fetch_trending_song] failed: {exc}")
        if entries:
            break

    if not entries:
        entries = [
            {"id": "ZqX7X3kSFo4", "title": "Sky High",   "source": "YouTube Audio Library", "artist": "Elektronomia"},
            {"id": "YBUA5-_Sv14", "title": "Alive",       "source": "YouTube Audio Library", "artist": "Nekzlo"},
            {"id": "d-_PiaqJjbc", "title": "Hope",        "source": "YouTube Audio Library", "artist": "Tobu"},
            {"id": "EP625xQIGzs", "title": "Infectious",  "source": "YouTube Audio Library", "artist": "Tobu"},
            {"id": "gQngg8iQipk", "title": "Energy",      "source": "YouTube Audio Library", "artist": "Elektronomia"},
        ]

    random.shuffle(entries)
    chosen = next((e for e in entries if f"{e['id']}|{e['source']}" not in used), None)
    if not chosen:
        USED_SONGS_DB.write_text("[]", encoding="utf-8")
        chosen = entries[0]

    state.update({
        "trending_song_title":    chosen["title"],
        "trending_song_artist":   chosen["artist"],
        "audio_library_video_id": chosen["id"],
        "music_metadata": {
            "title": chosen["title"], "artist": chosen["artist"],
            "source": chosen["source"], "license": "",
            "license_url": "", "attribution_required": True, "attribution_text": "",
        },
    })
    print(f"[fetch_trending_song] picked {chosen['title']!r} (id={chosen['id']})")
    return state


# ---------------------------------------------------------------------------
# Node 2 — download_trending_music
# ---------------------------------------------------------------------------
def download_trending_music(state: VideoState) -> VideoState:
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    raw_mp3    = AUDIO_DIR / "asset_trending_music_full.mp3"
    chorus_mp3 = AUDIO_DIR / "asset_trending_music.mp3"
    state["music_path"] = state["music_attribution"] = ""

    video_id = state.get("audio_library_video_id", "")
    if not video_id:
        raise RuntimeError("No music video ID selected.")

    command = [
        sys.executable, "-m", "yt_dlp",
        f"https://www.youtube.com/watch?v={video_id}",
        "--extract-audio", "--audio-format", "mp3", "--audio-quality", "0",
        "--output", str(AUDIO_DIR / "asset_trending_music_full.%(ext)s"),
        "--no-playlist", "--retries", "3", "--fragment-retries", "3",
        "--socket-timeout", "30", "--js-runtimes", "deno",
        "--remote-components", "ejs:github",
    ] + _ytdlp_cookie_args()

    last_error = None
    for attempt in range(1, 4):
        for old in AUDIO_DIR.glob("asset_trending_music*"):
            if old.is_file():
                old.unlink()
        try:
            print(f"[download_trending_music] attempt {attempt}/3")
            result = subprocess.run(command, capture_output=True, text=True, timeout=240)
            if result.returncode != 0:
                _log_ytdlp_failure("download_trending_music", result)
                raise RuntimeError(f"yt-dlp exited {result.returncode}")
            info = _probe_media(raw_mp3)
            if "audio" not in info["stream_types"]:
                raise RuntimeError("No audio stream in download.")
            _extract_chorus(raw_mp3, chorus_mp3, min(info["duration"], 60.0))
            chorus_info = _probe_media(chorus_mp3)
            if "audio" not in chorus_info["stream_types"]:
                raise RuntimeError("No audio in chorus extract.")
            state["music_path"] = str(chorus_mp3)
            prepare_music_for_render(state)
            print(f"[download_trending_music] ready: {chorus_mp3}")
            return state
        except Exception as exc:
            last_error = exc
            print(f"[download_trending_music] attempt failed: {exc}")
            if attempt < 3:
                time.sleep(5 * attempt)

    raise RuntimeError("Music download failed after 3 attempts.") from last_error


# ---------------------------------------------------------------------------
# Node 3 — transcribe_music
# ---------------------------------------------------------------------------
def transcribe_music(state: VideoState) -> VideoState:
    prepare_music_for_render(state)
    state["caption_words"] = []
    print("[transcribe_music] transcribing ...")
    model    = WhisperModel("small", device="cpu", compute_type="int8")
    segments, _ = model.transcribe(
        state["music_path"], word_timestamps=True, beam_size=5,
        condition_on_previous_text=False, vad_filter=False,
    )
    chorus_duration = _probe_media(Path(state["music_path"]))["duration"]
    words = []
    for segment in segments:
        for word in segment.words or []:
            text  = word.word.strip()
            start = max(0.0, float(word.start))
            end   = min(chorus_duration, float(word.end))
            if text and start < end:
                words.append({"text": text.upper(),
                              "start": start + MUSIC_START_OFFSET,
                              "end":   end   + MUSIC_START_OFFSET})
    words.sort(key=lambda w: w["start"])
    if not words:
        raise RuntimeError("No captions detected. Upload aborted.")
    state["caption_words"] = words
    print(f"[transcribe_music] {len(words)} caption words")
    return state


# ---------------------------------------------------------------------------
# Node 4 — assemble_video
# ---------------------------------------------------------------------------
LOOP_SEGMENT_START = 9.5
LOOP_SEGMENT_END   = 10.0


def _build_extended_video(asset_clip, total_duration: float):
    from moviepy import concatenate_videoclips
    asset_duration = float(asset_clip.duration)
    if total_duration <= asset_duration:
        return asset_clip.subclipped(0, total_duration)
    loop_end     = min(LOOP_SEGMENT_END, asset_duration)
    loop_start   = min(LOOP_SEGMENT_START, max(0.0, loop_end - 0.5))
    loop_segment = asset_clip.subclipped(loop_start, loop_end)
    pieces       = [asset_clip]
    remaining    = total_duration - asset_duration
    while remaining > 0.000001:
        dur = min(loop_segment.duration, remaining)
        pieces.append(loop_segment.subclipped(0, dur))
        remaining -= dur
    return concatenate_videoclips(pieces, method="chain").with_duration(total_duration)


def assemble_video(state: VideoState) -> VideoState:
    state["final_video_path"] = ""
    metadata = prepare_music_for_render(state)
    words    = state.get("caption_words", [])
    if not words:
        raise RuntimeError("Captions missing. Upload aborted.")

    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    destination = FINAL_DIR / "asset_final_video.mp4"
    temporary   = FINAL_DIR / "asset_final_video.rendering.mp4"
    destination.unlink(missing_ok=True)
    temporary.unlink(missing_ok=True)
    caption_font = _ensure_caption_font()

    with ExitStack() as resources:
        asset = resources.enter_context(VideoFileClip(str(ASSET_VIDEO)))
        music = resources.enter_context(AudioFileClip(state["music_path"]))

        total_duration = MUSIC_START_OFFSET + music.duration
        scale  = max(1080 / asset.w, 1920 / asset.h)
        visual = asset.without_audio().resized(scale)
        visual = visual.cropped(x_center=visual.w / 2, y_center=visual.h / 2,
                                width=1080, height=1920)
        visual = _build_extended_video(visual, total_duration)
        resources.callback(visual.close)

        audio_tracks = []
        if asset.audio is not None:
            intro = min(MUSIC_START_OFFSET, asset.duration, asset.audio.duration)
            if intro > 0:
                audio_tracks.append(asset.audio.subclipped(0, intro))
        audio_tracks.append(
            music.with_effects([afx.MultiplyVolume(MUSIC_VOLUME)]).with_start(MUSIC_START_OFFSET)
        )
        mixed_audio = CompositeAudioClip(audio_tracks).with_duration(total_duration)
        resources.callback(mixed_audio.close)

        overlays, caption_count = [], 0
        box_width = int(1080 * 0.82)
        box_x, box_y = (1080 - box_width) // 2, int(1920 * 0.46)

        for index, word in enumerate(words):
            start = max(0.0, float(word["start"]))
            end   = float(word["end"])
            if index + 1 < len(words):
                nxt = float(words[index + 1]["start"])
                if 0 < nxt - start <= 1.5:
                    end = nxt
            end = min(total_duration, max(end, start + 0.18))
            if start >= end:
                continue
            caption = (
                TextClip(text=word["text"], font=caption_font, font_size=88,
                         color="white", stroke_color="black", stroke_width=6,
                         text_align="center", size=(box_width, None), method="caption")
                .with_start(start).with_duration(end - start)
                .with_position((box_x, box_y))
            )
            resources.callback(caption.close)
            overlays.append(caption)
            caption_count += 1

        if caption_count == 0:
            raise RuntimeError("No captions fit the timeline.")

        # No badge overlay — clean visuals for Instagram
        final = (
            CompositeVideoClip([visual, *overlays], size=(1080, 1920))
            .with_duration(total_duration).with_audio(mixed_audio)
        )
        resources.callback(final.close)
        print(f"[assemble_video] rendering {total_duration:.2f}s, {caption_count} captions")
        final.write_videofile(
            str(temporary), fps=30, codec="libx264",
            audio_codec="aac", audio_bitrate="192k",
            bitrate=VIDEO_BITRATE, preset=VIDEO_PRESET,
            ffmpeg_params=VIDEO_FFMPEG_PARAMS,
        )

    info = _probe_media(temporary)
    if not {"video", "audio"}.issubset(info["stream_types"]):
        raise RuntimeError("Rendered output missing video or audio.")
    if abs(info["duration"] - total_duration) > 0.5:
        raise RuntimeError(f"Duration mismatch: expected {total_duration:.2f}s")
    subprocess.run(
        ["ffmpeg", "-v", "error", "-xerror", "-i", str(temporary),
         "-map", "0:a:0", "-f", "null", "-"],
        check=True, capture_output=True, text=True, timeout=120,
    )
    temporary.replace(destination)
    state["final_video_path"] = str(destination)
    _save_used_song(state["audio_library_video_id"], state["music_metadata"]["source"])
    print(f"[assemble_video] validated: {destination}")
    return state


# ---------------------------------------------------------------------------
# Node 5 — search_instagram_music
# Returns the Instagram music asset ID for a trending song.
# This is what makes the Reel show the real tappable ♪ song link.
# ---------------------------------------------------------------------------
def _get_instagrapi_client() -> Client:
    """Return an authenticated instagrapi Client, reusing saved session."""
    cl = Client()
    cl.delay_range = [1, 3]   # polite delay between API calls

    username = os.getenv("INSTAGRAM_USERNAME", "")
    password = os.getenv("INSTAGRAM_PASSWORD", "")
    session_b64 = os.getenv("INSTAGRAM_SESSION_JSON", "")

    # Try loading saved session first (avoids triggering 2FA on every run)
    if session_b64:
        try:
            session_json = base64.b64decode(session_b64).decode("utf-8")
            SESSION_FILE.write_text(session_json, encoding="utf-8")
            cl.load_settings(str(SESSION_FILE))
            cl.login(username, password)
            print("[instagrapi] session restored ✓")
            return cl
        except Exception as exc:
            print(f"[instagrapi] session restore failed ({exc}), trying fresh login ...")

    # Fresh login
    try:
        cl.login(username, password)
        print("[instagrapi] fresh login successful ✓")
    except TwoFactorRequired:
        # In CI the 2FA code must be passed via env var INSTAGRAM_2FA_CODE
        code = os.getenv("INSTAGRAM_2FA_CODE", "")
        if not code:
            raise RuntimeError(
                "Instagram 2FA required. Add INSTAGRAM_2FA_CODE as a GitHub secret "
                "with the current TOTP code, then re-run the workflow."
            )
        cl.login(username, password, verification_code=code)
        print("[instagrapi] 2FA login successful ✓")
    except (BadPassword, LoginRequired) as exc:
        raise RuntimeError(f"Instagram login failed: {exc}") from exc

    # Persist session so next run is silent
    cl.dump_settings(str(SESSION_FILE))
    encoded = base64.b64encode(SESSION_FILE.read_bytes()).decode("utf-8")
    print(
        f"\n[instagrapi] SESSION SAVED — update your INSTAGRAM_SESSION_JSON secret with:\n{encoded}\n"
    )
    return cl


def search_instagram_music(state: VideoState) -> VideoState:
    """Search Instagram's licensed music catalog for a trending track.

    Tries the downloaded song title first (best match with captions),
    then falls back to generic trending searches.
    Sets state['ig_music_id'] which upload_via_instagrapi attaches to the Reel.
    """
    state["ig_music_id"]    = ""
    state["ig_music_title"] = ""

    cl = _get_instagrapi_client()

    # Build query list: specific song first, then generic trending fallbacks
    song_title  = state.get("trending_song_title", "")
    song_artist = state.get("trending_song_artist", "")
    queries = []
    if song_title:
        queries.append(song_title)
    if song_artist and song_artist not in ("YouTube Audio Library",):
        queries.append(song_artist)
    queries.extend(TRENDING_MUSIC_QUERIES)

    for query in queries:
        try:
            print(f"[search_instagram_music] searching: {query!r} ...")
            tracks = cl.music_search(query)
            if tracks:
                track = tracks[0]
                state["ig_music_id"]    = str(track.id)
                state["ig_music_title"] = getattr(track, "title", query)
                print(
                    f"[search_instagram_music] found: {state['ig_music_title']!r} "
                    f"(id={state['ig_music_id']})"
                )
                return state
        except Exception as exc:
            print(f"[search_instagram_music] query {query!r} failed: {exc}")
            time.sleep(2)

    # Not fatal — Reel will be uploaded without a tappable music link
    print("[search_instagram_music] no music found — Reel will upload without music tag")
    return state


# ---------------------------------------------------------------------------
# Node 6 — upload_via_instagrapi
# ---------------------------------------------------------------------------
def upload_via_instagrapi(state: VideoState) -> VideoState:
    """Upload the rendered video as an Instagram Reel using instagrapi.

    If ig_music_id was found, attaches it so the Reel shows a real
    tappable ♪ song link from Instagram's licensed music catalog.
    """
    state["instagram_url"] = ""

    ig_captions = [
        "This transition hits different 🔥 #reels #viral #trending #explore #fyp",
        "Wait for the beat drop 👀 #reels #viral #transition #fyp #explorepage",
        "POV: you found the perfect vibe ✨ #reels #viral #explore #trending",
        "Stop scrolling ✋ #reels #viral #trending #explore #fyp",
        "The vibe is unmatched 🎶 #reels #viral #fyp #trending #explore",
    ]
    caption = random.choice(ig_captions)
    attribution = state.get("music_attribution", "")
    if attribution:
        caption += f"\n\n{attribution}"

    cl = _get_instagrapi_client()

    extra_data: dict = {}
    ig_music_id = state.get("ig_music_id", "")
    if ig_music_id:
        # Attach the licensed track — this creates the real tappable ♪ link
        extra_data["clips_audio_metadata"] = json.dumps([{
            "audio_asset_id": ig_music_id,
            "start_time_ms":  "0",
        }])
        print(f"[upload_via_instagrapi] attaching music: {state.get('ig_music_title')!r}")
    else:
        print("[upload_via_instagrapi] uploading without music tag")

    print("[upload_via_instagrapi] uploading Reel ...")
    media = cl.clip_upload(
        path=Path(state["final_video_path"]),
        caption=caption,
        extra_data=extra_data,
    )

    post_id = str(media.pk)
    ig_url  = f"https://www.instagram.com/reel/{media.code}/"
    state["instagram_url"] = ig_url
    print(f"[upload_via_instagrapi] published! {ig_url}")
    return state


# ---------------------------------------------------------------------------
# Build LangGraph
# ---------------------------------------------------------------------------
_NODES = [
    ("fetch_trending_song",     fetch_trending_song),
    ("download_trending_music", download_trending_music),
    ("transcribe_music",        transcribe_music),
    ("assemble_video",          assemble_video),
    ("search_instagram_music",  search_instagram_music),
    ("upload_via_instagrapi",   upload_via_instagrapi),
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
        "ig_music_id":            "",
        "ig_music_title":         "",
        "instagram_url":          "",
    })
    print("\n--- DONE ---")
    for key in ("trending_song_title", "ig_music_title", "final_video_path", "instagram_url"):
        print(f"  {key}: {result.get(key, '')}")
