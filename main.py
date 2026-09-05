import functools
import glob
import http.client
import os
import random
import subprocess
import time
from typing import TypedDict

import asyncio
import edge_tts
import httplib2
import requests
from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from moviepy import (
    afx,
    AudioFileClip,
    ColorClip,
    CompositeAudioClip,
    CompositeVideoClip,
    TextClip,
    VideoFileClip,
    concatenate_videoclips,
    vfx,
)
from faster_whisper import WhisperModel

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TARGET_DURATION = 22  # seconds — aim for the middle of the 20-25s range
MAX_DURATION = 25     # hard cap enforced in assemble_video regardless of script length
CROSSFADE_DURATION = 0.3

AUDIO_DIR = "output_audio"
RAW_VIDEO_DIR = "output_raw_video"
FINAL_VIDEO_DIR = "output_final_video"
FONT_PATH = "assets/Roboto-Bold.ttf"     # commit a real .ttf to this path in your repo
MUSIC_DIR = "assets/music"               # optional local fallback — leave empty, not required
MUSIC_VOLUME = 0.12                      # kept low so it sits under the voice

# Mood-matched music via the free Jamendo API (https://developer.jamendo.com) — no
# files to commit. Get a free client_id at https://devportal.jamendo.com/ and set
# it as the JAMENDO_CLIENT_ID secret. If unset, music mixing is skipped entirely.
MOOD_OPTIONS = ["upbeat", "calm", "dramatic", "mysterious", "motivational", "playful"]
JAMENDO_MOOD_TAGS = {
    "upbeat": "happy",
    "calm": "relaxing",
    "dramatic": "dramatic",
    "mysterious": "dark",
    "motivational": "epic",
    "playful": "fun",
}
JAMENDO_FALLBACK_TAG = "instrumental"
REQUIRE_COMMERCIAL_SAFE_MUSIC = True  # skip tracks whose license forbids commercial use

VIDEO_FFMPEG_PARAMS = ["-crf", "18"]     # lower = higher quality; 18 is visually near-lossless
VIDEO_BITRATE = "8000k"
VIDEO_PRESET = "slow"                    # slower encode, better quality-per-bitrate

REQUIRED_ENV_VARS = [
    "GROQ_API_KEY",
    "PEXELS_API_KEY",
    "GOOGLE_CLIENT_ID",
    "GOOGLE_CLIENT_SECRET",
    "YOUTUBE_REFRESH_TOKEN",
    "DRIVE_REFRESH_TOKEN",
]


def validate_env():
    missing = [v for v in REQUIRED_ENV_VARS if not os.getenv(v)]
    if missing:
        raise EnvironmentError(
            f"Missing required environment variable(s): {', '.join(missing)}. "
            f"Set them as repo/organization secrets before running."
        )


PEXELS_HEADERS = {}  # populated after validate_env() runs in __main__


# ---------------------------------------------------------------------------
# Small retry helper for flaky network calls (Groq, Pexels)
# ---------------------------------------------------------------------------
def with_retry(max_attempts=3, base_delay=2):
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            attempt = 0
            while True:
                try:
                    return fn(*args, **kwargs)
                except Exception as e:
                    attempt += 1
                    if attempt >= max_attempts:
                        print(f"[{fn.__name__}] failed after {attempt} attempts: {e}")
                        raise
                    delay = base_delay * (2 ** (attempt - 1)) + random.random()
                    print(f"[{fn.__name__}] attempt {attempt} failed ({e}); retrying in {delay:.1f}s")
                    time.sleep(delay)
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# Google credential helpers
# ---------------------------------------------------------------------------
def get_youtube_credentials():
    creds = Credentials(
        token=None,
        refresh_token=os.getenv("YOUTUBE_REFRESH_TOKEN"),
        client_id=os.getenv("GOOGLE_CLIENT_ID"),
        client_secret=os.getenv("GOOGLE_CLIENT_SECRET"),
        token_uri="https://oauth2.googleapis.com/token",
        scopes=["https://www.googleapis.com/auth/youtube.upload"],
    )
    creds.refresh(Request())
    return creds


def get_drive_credentials():
    creds = Credentials(
        token=None,
        refresh_token=os.getenv("DRIVE_REFRESH_TOKEN"),
        client_id=os.getenv("GOOGLE_CLIENT_ID"),
        client_secret=os.getenv("GOOGLE_CLIENT_SECRET"),
        token_uri="https://oauth2.googleapis.com/token",
        scopes=["https://www.googleapis.com/auth/drive.file"],
    )
    creds.refresh(Request())
    return creds


model = ChatGroq(
    model="openai/gpt-oss-120b",
    groq_api_key=os.getenv("GROQ_API_KEY"),
)


class VideoState(TypedDict):
    topic: str
    script: str
    visual_query: str
    music_mood: str
    music_attribution: str
    audio_path: str
    video_clip_paths: list
    raw_video_path: str
    caption_words: list
    final_video_path: str
    youtube_url: str
    drive_url: str


# ---------------------------------------------------------------------------
# Pipeline nodes
# ---------------------------------------------------------------------------
@with_retry()
def _invoke_model(prompt: str) -> str:
    return model.invoke(prompt).content.strip()


def choose_topic(state: VideoState) -> VideoState:
    prompt = (
        f"Suggest one specific, engaging topic for a {TARGET_DURATION}-second YouTube Short / "
        "Instagram Reel. It should be something with broad appeal — productivity, "
        "psychology, science facts, life hacks, etc. "
        "Reply with ONLY the topic itself, no extra text, no quotes."
    )
    topic = _invoke_model(prompt)
    state["topic"] = topic
    print(f"[choose_topic] chosen topic: {topic}")
    return state


def generate_script(state: VideoState) -> VideoState:
    prompt = (
        f"Write a punchy ~{TARGET_DURATION} second short-form video script about: {state['topic']}\n"
        "Follow this structure strictly:\n"
        "1. Hook (1 line): a curiosity gap or bold claim, no throat-clearing.\n"
        "2. Body (2-3 lines): concrete, specific facts or steps — not generic advice.\n"
        "3. Payoff (1 line): a twist, takeaway, or call to action.\n"
        f"Keep it tight — speakable in {TARGET_DURATION} seconds or less, so roughly "
        f"{TARGET_DURATION * 3} words maximum. Conversational tone. "
        "No stage directions, no labels like 'Hook:' — just the spoken lines."
    )
    state["script"] = _invoke_model(prompt)
    print(f"[generate_script] script generated ({len(state['script'])} chars)")
    return state


def generate_visual_query(state: VideoState) -> VideoState:
    prompt = (
        f"Topic: {state['topic']}\n"
        "Give a short, concrete stock-footage search query (3-6 words) that describes "
        "a VISUAL scene matching this topic — something a camera could literally film "
        "(e.g. a person, action, or setting), not an abstract concept. "
        "Reply with ONLY the search query, no extra text, no quotes."
    )
    state["visual_query"] = _invoke_model(prompt)
    print(f"[generate_visual_query] query: {state['visual_query']}")
    return state


def generate_music_mood(state: VideoState) -> VideoState:
    prompt = (
        f"Script:\n{state['script']}\n\n"
        f"Which single mood best fits background music for this video? "
        f"Choose EXACTLY ONE word from this list: {', '.join(MOOD_OPTIONS)}. "
        "Reply with ONLY that one word, lowercase, no extra text, no quotes."
    )
    mood = _invoke_model(prompt).lower().strip()
    if mood not in MOOD_OPTIONS:
        print(f"[generate_music_mood] model returned unrecognized mood '{mood}', defaulting to 'calm'")
        mood = "calm"
    state["music_mood"] = mood
    print(f"[generate_music_mood] mood: {mood}")
    return state


async def _generate_speech_async(script, voice, output_file):
    communicate = edge_tts.Communicate(script, voice)
    await communicate.save(output_file)


def _normalize_loudness(path: str):
    try:
        subprocess.run(
            ["ffmpeg-normalize", path, "-o", path, "-f", "-t", "-16"],
            check=True, capture_output=True,
        )
        print(f"[generate_speech] loudness normalized {path}")
    except FileNotFoundError:
        print("[generate_speech] ffmpeg-normalize not installed, skipping (pip install ffmpeg-normalize)")
    except subprocess.CalledProcessError as e:
        print(f"[generate_speech] normalization failed, keeping original audio: {e.stderr.decode(errors='ignore')[:200]}")


def generate_speech(state: VideoState) -> VideoState:
    os.makedirs(AUDIO_DIR, exist_ok=True)
    voice = "en-US-ChristopherNeural"
    output_file = os.path.join(AUDIO_DIR, "output.mp3")
    asyncio.run(_generate_speech_async(state["script"], voice, output_file))
    _normalize_loudness(output_file)
    state["audio_path"] = output_file
    print(f"[generate_speech] audio saved to {output_file}")
    return state


@with_retry()
def _pexels_search(url, headers, params):
    resp = requests.get(url, headers=headers, params=params, timeout=15)
    resp.raise_for_status()
    return resp.json().get("videos", [])


def fetch_stock_clips(state: VideoState, clips_needed: int = 4) -> VideoState:
    query = state.get("visual_query") or state["topic"]
    url = "https://api.pexels.com/videos/search"
    params = {"query": query, "per_page": clips_needed, "orientation": "portrait"}

    results = _pexels_search(url, PEXELS_HEADERS, params)

    if not results:
        params["query"] = "lifestyle abstract background"
        results = _pexels_search(url, PEXELS_HEADERS, params)

    os.makedirs("stock_clips", exist_ok=True)
    clip_paths = []

    for i, video in enumerate(results[:clips_needed]):
        video_files = sorted(video["video_files"], key=lambda f: f.get("width", 0))
        chosen = next((f for f in video_files if f.get("width", 0) >= 720), video_files[-1])

        clip_path = f"stock_clips/clip_{i}.mp4"
        with requests.get(chosen["link"], stream=True, timeout=30) as r:
            r.raise_for_status()
            with open(clip_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

        clip_paths.append(clip_path)
        print(f"[fetch_stock_clips] downloaded {clip_path}")

    state["video_clip_paths"] = clip_paths
    return state


@with_retry()
def _jamendo_search(tag: str, client_id: str):
    resp = requests.get(
        "https://api.jamendo.com/v3.0/tracks/",
        params={
            "client_id": client_id,
            "format": "json",
            "limit": 10,
            "tags": tag,
            "audioformat": "mp32",
            "include": "musicinfo",
            "order": "popularity_total",
        },
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json().get("results", [])


def _is_commercial_safe(license_ccurl: str) -> bool:
    # e.g. "http://creativecommons.org/licenses/by-nc-sa/3.0/" -> "by-nc-sa"
    try:
        segment = license_ccurl.rstrip("/").split("/")[-2]
    except IndexError:
        return False
    return "nc" not in segment.split("-")


def _fetch_jamendo_track(mood: str):
    """Search Jamendo for a mood-matched track and download it. Returns
    (local_path, attribution_string) or (None, None) if unavailable."""
    client_id = os.getenv("JAMENDO_CLIENT_ID")
    if not client_id:
        print("[assemble_video] JAMENDO_CLIENT_ID not set, skipping background music")
        return None, None

    tags_to_try = [JAMENDO_MOOD_TAGS.get(mood, JAMENDO_FALLBACK_TAG), JAMENDO_FALLBACK_TAG]
    for tag in tags_to_try:
        try:
            results = _jamendo_search(tag, client_id)
        except Exception as e:
            print(f"[assemble_video] Jamendo search failed for tag '{tag}': {e}")
            continue

        for track in results:
            if not track.get("audiodownload_allowed", False):
                continue
            if REQUIRE_COMMERCIAL_SAFE_MUSIC and not _is_commercial_safe(track.get("license_ccurl", "")):
                continue

            audio_url = track.get("audiodownload") or track.get("audio")
            if not audio_url:
                continue

            local_path = os.path.join(AUDIO_DIR, "bg_track.mp3")
            try:
                with requests.get(audio_url, stream=True, timeout=30) as r:
                    r.raise_for_status()
                    with open(local_path, "wb") as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
            except Exception as e:
                print(f"[assemble_video] failed to download track {track.get('id')}: {e}")
                continue

            attribution = (
                f"Music: \"{track.get('name', 'Untitled')}\" by {track.get('artist_name', 'Unknown Artist')} "
                f"(Jamendo, {track.get('license_ccurl', '')})"
            )
            print(f"[assemble_video] using Jamendo track: {attribution}")
            return local_path, attribution

    print(f"[assemble_video] no usable Jamendo track found for mood '{mood}', skipping music")
    return None, None


def _add_background_music(voice_path: str, duration: float, mood: str = None) -> tuple[str, str]:
    """Mix a mood-matched track (fetched live from Jamendo) under the voice
    track. Returns (audio_path_to_use, attribution_string_or_empty)."""
    music_path, attribution = _fetch_jamendo_track(mood or "calm")

    if not music_path:
        # Optional local fallback, only used if you happen to have files there
        local_tracks = glob.glob(os.path.join(MUSIC_DIR, "*.mp3")) if os.path.isdir(MUSIC_DIR) else []
        if local_tracks:
            music_path = random.choice(local_tracks)
            attribution = ""
            print(f"[assemble_video] falling back to local track: {os.path.basename(music_path)}")
        else:
            return voice_path, ""

    voice = AudioFileClip(voice_path)
    music = AudioFileClip(music_path)

    if music.duration < duration:
        loops = int(duration // music.duration) + 1
        music = concatenate_videoclips([music] * loops) if hasattr(music, "size") else music
    music = music.subclipped(0, duration).with_effects([afx.MultiplyVolume(MUSIC_VOLUME)])

    mixed = CompositeAudioClip([music, voice])
    mixed_path = os.path.join(AUDIO_DIR, "mixed.mp3")
    mixed.write_audiofile(mixed_path)
    return mixed_path, attribution


def assemble_video(state: VideoState) -> VideoState:
    audio = AudioFileClip(state["audio_path"])
    target_duration = min(audio.duration, MAX_DURATION)
    if audio.duration > MAX_DURATION:
        audio = audio.subclipped(0, MAX_DURATION)
        print(f"[assemble_video] audio trimmed to {MAX_DURATION}s cap")

    mixed_audio_path, attribution = _add_background_music(state["audio_path"], target_duration, mood=state.get("music_mood"))
    state["music_attribution"] = attribution
    audio = AudioFileClip(mixed_audio_path).subclipped(0, target_duration)

    clips = []
    running_total = 0.0
    for path in state["video_clip_paths"]:
        if running_total >= target_duration:
            break
        clip = VideoFileClip(path)
        clip = clip.resized(height=1920)
        if clip.w > 1080:
            x_center = clip.w / 2
            clip = clip.cropped(x_center=x_center, width=1080)

        remaining = target_duration - running_total
        if clip.duration > remaining:
            clip = clip.subclipped(0, remaining)

        clips.append(clip)
        running_total += clip.duration

    # Crossfade between clips instead of hard cuts
    faded_clips = []
    for i, c in enumerate(clips):
        if i > 0:
            c = c.with_effects([vfx.CrossFadeIn(CROSSFADE_DURATION)])
        faded_clips.append(c)

    raw_video = concatenate_videoclips(
        faded_clips, method="compose",
        padding=-CROSSFADE_DURATION if len(faded_clips) > 1 else 0,
    )
    raw_video = raw_video.with_audio(audio)

    os.makedirs(RAW_VIDEO_DIR, exist_ok=True)
    raw_path = os.path.join(RAW_VIDEO_DIR, "raw_video.mp4")
    raw_video.write_videofile(
        raw_path, fps=30, codec="libx264", audio_codec="aac",
        bitrate=VIDEO_BITRATE, preset=VIDEO_PRESET, ffmpeg_params=VIDEO_FFMPEG_PARAMS,
    )

    state["raw_video_path"] = raw_path
    print(f"[assemble_video] wrote {raw_path} ({raw_video.duration:.1f}s)")
    return state


def generate_captions(state: VideoState) -> VideoState:
    whisper_model = WhisperModel("tiny", device="cpu", compute_type="int8")
    segments, _ = whisper_model.transcribe(state["audio_path"], word_timestamps=True)

    words = []
    for segment in segments:
        for word in segment.words:
            words.append({"text": word.word.strip(), "start": word.start, "end": word.end})

    state["caption_words"] = words
    print(f"[generate_captions] transcribed {len(words)} words")
    return state


def burn_captions(state: VideoState) -> VideoState:
    video = VideoFileClip(state["raw_video_path"])
    words = state["caption_words"]

    font = FONT_PATH if os.path.exists(FONT_PATH) else None
    if font is None:
        print(f"[burn_captions] WARNING: {FONT_PATH} not found, falling back to default font")

    # Single-word "pop" captions — one word on screen at a time, closer to
    # the TikTok/Shorts caption style than static multi-word chunks.
    overlay_clips = []
    for w in words:
        if not w["text"]:
            continue
        start, end = w["start"], w["end"]
        duration = max(end - start, 0.15)

        txt_clip = (
            TextClip(
                text=w["text"].upper(),
                font=font,
                font_size=90,
                color="white",
                stroke_color="black",
                stroke_width=4,
                size=(int(video.w * 0.85), None),
                method="caption",
            )
            .with_start(start)
            .with_duration(duration)
            .with_position(("center", "center"))
        )

        # Semi-transparent backing box so captions stay readable over any footage
        bg = (
            ColorClip(size=(txt_clip.w + 50, txt_clip.h + 30), color=(0, 0, 0))
            .with_opacity(0.35)
            .with_start(start)
            .with_duration(duration)
            .with_position(("center", "center"))
        )

        overlay_clips.append(bg)
        overlay_clips.append(txt_clip)

    final = CompositeVideoClip([video, *overlay_clips])

    os.makedirs(FINAL_VIDEO_DIR, exist_ok=True)
    final_path = os.path.join(FINAL_VIDEO_DIR, "final_video.mp4")
    final.write_videofile(
        final_path, fps=30, codec="libx264", audio_codec="aac",
        bitrate=VIDEO_BITRATE, preset=VIDEO_PRESET, ffmpeg_params=VIDEO_FFMPEG_PARAMS,
    )

    state["final_video_path"] = final_path
    print(f"[burn_captions] wrote {final_path}")
    return state


# ---------------------------------------------------------------------------
# Resumable, retrying uploads
# ---------------------------------------------------------------------------
RETRIABLE_STATUS_CODES = (500, 502, 503, 504)
RETRIABLE_EXCEPTIONS = (
    httplib2.HttpLib2Error,
    IOError,
    http.client.NotConnected,
    http.client.IncompleteRead,
    http.client.ImproperConnectionState,
    http.client.CannotSendRequest,
    http.client.CannotSendHeader,
    http.client.ResponseNotReady,
    http.client.BadStatusLine,
    ConnectionError,
    TimeoutError,
)


def _run_resumable_upload(request, max_retries=8, label="upload"):
    response = None
    retry = 0
    while response is None:
        try:
            status, response = request.next_chunk()
            if status:
                print(f"[{label}] progress {int(status.progress() * 100)}%")
        except HttpError as e:
            if e.resp.status in RETRIABLE_STATUS_CODES:
                retry += 1
                if retry > max_retries:
                    raise RuntimeError(f"{label} failed after max retries") from e
                sleep_time = min(2 ** retry + random.random(), 60)
                print(f"[{label}] HTTP {e.resp.status}, retrying in {sleep_time:.1f}s ({retry}/{max_retries})")
                time.sleep(sleep_time)
            else:
                raise
        except RETRIABLE_EXCEPTIONS as e:
            retry += 1
            if retry > max_retries:
                raise RuntimeError(f"{label} failed after max retries") from e
            sleep_time = min(2 ** retry + random.random(), 60)
            print(f"[{label}] {type(e).__name__}: {e}, retrying in {sleep_time:.1f}s ({retry}/{max_retries})")
            time.sleep(sleep_time)
    return response


def upload_to_youtube(state: VideoState) -> VideoState:
    creds = get_youtube_credentials()
    youtube = build("youtube", "v3", credentials=creds)

    description = state["script"]
    if state.get("music_attribution"):
        description += f"\n\n{state['music_attribution']}"

    media = MediaFileUpload(
        state["final_video_path"],
        chunksize=4 * 1024 * 1024,
        resumable=True,
        mimetype="video/mp4",
    )
    request = youtube.videos().insert(
        part="snippet,status",
        body={
            "snippet": {
                "title": state["topic"][:100],
                "description": description,
                "tags": ["shorts"],
            },
            "status": {"privacyStatus": "public"},
        },
        media_body=media,
    )

    response = _run_resumable_upload(request, label="upload_to_youtube")

    video_url = f"https://youtube.com/shorts/{response['id']}"
    state["youtube_url"] = video_url
    print(f"[upload_to_youtube] uploaded -> {video_url}")
    return state


def upload_to_drive(state: VideoState) -> VideoState:
    creds = get_drive_credentials()
    drive = build("drive", "v3", credentials=creds)

    file_metadata = {"name": os.path.basename(state["final_video_path"])}
    media = MediaFileUpload(state["final_video_path"], chunksize=4 * 1024 * 1024, resumable=True)

    request = drive.files().create(body=file_metadata, media_body=media, fields="id, webViewLink")
    response = _run_resumable_upload(request, label="upload_to_drive")

    state["drive_url"] = response["webViewLink"]
    print(f"[upload_to_drive] uploaded -> {response['webViewLink']}")
    return state


# ---------------------------------------------------------------------------
# Graph wiring
# ---------------------------------------------------------------------------
graph = StateGraph(VideoState)
graph.add_node("choose_topic", choose_topic)
graph.add_node("generate_script", generate_script)
graph.add_node("generate_visual_query", generate_visual_query)
graph.add_node("generate_music_mood", generate_music_mood)
graph.add_node("generate_speech", generate_speech)
graph.add_node("fetch_stock_clips", fetch_stock_clips)
graph.add_node("assemble_video", assemble_video)
graph.add_node("generate_captions", generate_captions)
graph.add_node("burn_captions", burn_captions)
graph.add_node("upload_to_youtube", upload_to_youtube)
graph.add_node("upload_to_drive", upload_to_drive)

graph.add_edge(START, "choose_topic")
graph.add_edge("choose_topic", "generate_script")
graph.add_edge("generate_script", "generate_visual_query")
graph.add_edge("generate_visual_query", "generate_music_mood")
graph.add_edge("generate_music_mood", "generate_speech")
graph.add_edge("generate_speech", "fetch_stock_clips")
graph.add_edge("fetch_stock_clips", "assemble_video")
graph.add_edge("assemble_video", "generate_captions")
graph.add_edge("generate_captions", "burn_captions")
graph.add_edge("burn_captions", "upload_to_youtube")
graph.add_edge("upload_to_youtube", "upload_to_drive")
graph.add_edge("upload_to_drive", END)
workflow = graph.compile()


if __name__ == "__main__":
    validate_env()
    PEXELS_HEADERS["Authorization"] = os.getenv("PEXELS_API_KEY")

    result = workflow.invoke({
        "topic": "",
        "script": "",
        "visual_query": "",
        "music_mood": "",
        "music_attribution": "",
        "audio_path": "",
        "video_clip_paths": [],
        "raw_video_path": "",
        "caption_words": [],
        "final_video_path": "",
        "youtube_url": "",
        "drive_url": "",
    })
    print("\n--- FINAL STATE ---")
    print(f"Topic:  {result['topic']}")
    print(f"Script:\n{result['script']}")
    print(f"Audio:  {result['audio_path']}")
    print(f"Final video: {result['final_video_path']}")
    print(f"YouTube: {result['youtube_url']}")
    print(f"Drive:   {result['drive_url']}")