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
import tempfile
import time
import pathlib
from pathlib import Path
from typing import TypedDict

import httplib2
import librosa
import numpy as np
import requests
import soundfile as sf
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
    vfx,
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
YT_AUDIO_LIBRARY_CHANNEL = "https://www.youtube.com/@YouTubeAudioLibrary"

REQUIRED_ENV_VARS = [
    "GOOGLE_CLIENT_ID",
    "GOOGLE_CLIENT_SECRET",
    "YOUTUBE_REFRESH_TOKEN",
    "DRIVE_REFRESH_TOKEN",
]


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
class VideoState(TypedDict):
    trending_song_title:      str
    trending_song_artist:     str
    audio_library_video_id:   str   # YouTube Audio Library video ID
    music_path:               str
    music_attribution:        str
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
    try:
        y, sr = librosa.load(str(src_path), sr=None, mono=True)
        frame_length = int(sr * 0.5)
        hop_length   = frame_length // 2
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]

        frames_needed = max(1, int(duration / (hop_length / sr)))

        if len(rms) <= frames_needed:
            best_start_frame = 0
        else:
            cumsum = np.cumsum(np.concatenate(([0], rms)))
            window_sums = cumsum[frames_needed:] - cumsum[:-frames_needed]
            best_start_frame = int(np.argmax(window_sums))

        start_sample = best_start_frame * hop_length
        end_sample   = min(start_sample + int(duration * sr), len(y))
        chorus_y     = y[start_sample:end_sample]

        start_sec = start_sample / sr
        print(f"[chorus] peak-energy at {start_sec:.1f}s ({len(chorus_y)/sr:.1f}s extracted)")

        # WAV -> ffmpeg -> MP3 (works on both Windows and Ubuntu)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_wav = tmp.name
        sf.write(tmp_wav, chorus_y, sr)
        subprocess.run(
            ["ffmpeg", "-y", "-i", tmp_wav, "-b:a", "192k", str(dest_path)],
            check=True, capture_output=True,
        )
        os.unlink(tmp_wav)
    except Exception as exc:
        print(f"[chorus] failed ({exc}), using first {duration:.0f}s instead")
        clip = AudioFileClip(str(src_path))
        clip.subclipped(0, min(duration, clip.duration)).write_audiofile(str(dest_path))


def _loop_audio(clip: AudioFileClip, duration: float) -> AudioFileClip:
    pieces, remaining = [], duration
    while remaining > 0:
        piece = clip.subclipped(0, min(clip.duration, remaining))
        pieces.append(piece)
        remaining -= piece.duration
    return concatenate_audioclips(pieces)


# ---------------------------------------------------------------------------
# Node 1 — fetch_trending_song (iTunes Top-10, skip already-used)
# ---------------------------------------------------------------------------
def fetch_trending_song(state: VideoState) -> VideoState:
    """Pick a random copyright-free track from YouTube Audio Library.

    YouTube Audio Library tracks are explicitly licensed for reuse on
    YouTube — no Content ID claims, no copyright strikes.
    """
    print("[fetch_trending_song] fetching YouTube Audio Library tracks ...")
    used = _load_used_songs()
    entries = []

    # ── Primary: pull track list directly from the Audio Library channel ──
    for cmd_prefix in (["yt-dlp"], [sys.executable, "-m", "yt_dlp"]):
        try:
            result = subprocess.run(
                cmd_prefix + [
                    YT_AUDIO_LIBRARY_CHANNEL,
                    "--flat-playlist",
                    "--print", "%(id)s|||%(title)s",
                    "--playlist-items", "1-80",
                    "--no-warnings", "--quiet",
                ],
                capture_output=True, text=True, timeout=45,
            )
            for line in result.stdout.strip().splitlines():
                if "|||" in line:
                    vid_id, title = line.split("|||", 1)
                    entries.append({"id": vid_id.strip(), "title": title.strip()})
            if entries:
                break
        except Exception as exc:
            print(f"[fetch_trending_song] channel fetch attempt failed: {exc}")

    # ── Fallback: search YouTube for Audio Library music ──
    if not entries:
        print("[fetch_trending_song] falling back to search ...")
        for cmd_prefix in (["yt-dlp"], [sys.executable, "-m", "yt_dlp"]):
            try:
                result = subprocess.run(
                    cmd_prefix + [
                        "ytsearch30:youtube audio library no copyright background music",
                        "--flat-playlist",
                        "--print", "%(id)s|||%(title)s",
                        "--no-warnings", "--quiet",
                    ],
                    capture_output=True, text=True, timeout=45,
                )
                for line in result.stdout.strip().splitlines():
                    if "|||" in line:
                        vid_id, title = line.split("|||", 1)
                        entries.append({"id": vid_id.strip(), "title": title.strip()})
                if entries:
                    break
            except Exception as exc:
                print(f"[fetch_trending_song] search fallback failed: {exc}")

    random.shuffle(entries)
    chosen = None
    for entry in entries:
        if entry["id"] not in used:
            chosen = entry
            break

    if not chosen:
        print("[fetch_trending_song] all tracks used — resetting tracker")
        USED_SONGS_DB.write_text("[]", encoding="utf-8")
        chosen = entries[0] if entries else {"id": "", "title": "No Copyright Music"}

    state["trending_song_title"]    = chosen["title"]
    state["trending_song_artist"]   = "YouTube Audio Library"
    state["audio_library_video_id"] = chosen["id"]
    print(f"[fetch_trending_song] picked: {chosen['title']!r} (id={chosen['id']})")
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
            f"(YouTube Audio Library — free to use)"
        )
        # Track by video_id so the same track isn't picked again
        _save_used_song(video_id, "YouTube Audio Library")
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

    # --- Determine total video duration ---
    music_clip = None
    if state.get("music_path") and Path(state["music_path"]).is_file():
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

    # --- "Now Playing" badge (appears at 5s when music kicks in, shown for 5s) ---
    overlays = []
    song_title  = state.get("trending_song_title", "")
    song_artist = state.get("trending_song_artist", "")
    font = str(FONT_PATH) if FONT_PATH.exists() else None
    if song_title and music_clip:
        badge_text  = f"Now Playing: {song_title} - {song_artist}"
        badge_start = MUSIC_START_OFFSET
        badge_dur   = min(5.0, total_dur - badge_start)
        if badge_dur > 0:
            badge_txt = TextClip(
                text=badge_text, font=font, font_size=36,
                color="white", stroke_color="black", stroke_width=2,
                size=(int(1080 * 0.90), None), method="caption",
            )
            badge_y = int(1920 * 0.88)
            badge_bg = (
                ColorClip(size=(badge_txt.w + 40, badge_txt.h + 20), color=(20, 20, 20))
                .with_opacity(0.70)
                .with_start(badge_start).with_duration(badge_dur)
                .with_position(("center", badge_y))
            )
            badge_txt = (
                badge_txt.with_start(badge_start).with_duration(badge_dur)
                .with_position(("center", badge_y))
            )
            overlays.extend([badge_bg, badge_txt])

    # ── Lyric captions (word-by-word, styled with Anton font) ──
    caption_words = state.get("caption_words", [])
    caption_font  = _ensure_caption_font()
    total_dur     = video.duration if hasattr(video, 'duration') else total_dur

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
        print("[upload_to_instagram] INSTAGRAM_ACCOUNT_ID or INSTAGRAM_ACCESS_TOKEN missing — skipping upload")
        state["instagram_url"] = ""
        return state

    if not drive_file_id:
        print("[upload_to_instagram] no drive_file_id available — skipping upload")
        state["instagram_url"] = ""
        return state

    ig_captions = [
        "This transition hits different 🔥 #reels #viral #trending #explore #fyp",
        "Wait for the beat drop 👀 #reels #viral #transition #fyp #explorepage",
        "POV: you found the perfect vibe ✨ #reels #viral #explore #trending",
        "Stop scrolling ✋ #reels #viral #trending #explore #fyp",
    ]
    caption = random.choice(ig_captions)

    # Google Drive high-speed CDN direct video stream URL
    video_url = f"https://drive.usercontent.google.com/download?id={drive_file_id}&export=download"

    print("[upload_to_instagram] initializing Reel container via Meta Graph API ...")
    try:
        init_url = f"https://graph.facebook.com/v20.0/{ig_account_id}/media"
        init_res = requests.post(init_url, data={
            "media_type": "REELS",
            "video_url": video_url,
            "caption": caption,
            "share_to_feed": True,
            "access_token": ig_access_token,
        }, timeout=30).json()

        if "id" not in init_res:
            print(f"[upload_to_instagram] container init failed: {init_res}")
            state["instagram_url"] = ""
            return state

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
                print(f"[upload_to_instagram] container failed: {stat}")
                state["instagram_url"] = ""
                return state
            print(f"[upload_to_instagram] processing: {code} ...")
        else:
            print("[upload_to_instagram] processing timed out")
            state["instagram_url"] = ""
            return state

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
        else:
            print(f"[upload_to_instagram] publish failed: {pub_res}")

    except Exception as exc:
        print(f"[upload_to_instagram] Meta API error: {exc}")

    state["instagram_url"] = ""
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
