import os
import re
import sys
import time
import json
import math
import random
import shutil
import subprocess
from datetime import datetime, timedelta
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

# Volume boost for the asset audio in the first MUSIC_START_SECONDS.
# 1.0 = original level, 1.5 = 50% louder, 2.0 = twice as loud.
ASSET_AUDIO_BOOST = float(os.getenv("ASSET_AUDIO_BOOST", "1.8"))

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

# Last.fm API key (optional — set as GitHub secret LASTFM_API_KEY)
LASTFM_API_KEY = os.getenv("LASTFM_API_KEY", "").strip()

# ── Scoring weights (must sum to 1.0) ─────────────────────
# Override via env vars in generate-video.yml if needed.
W_TREND_RANK        = float(os.getenv("W_TREND_RANK",        "0.30"))
W_RANK_MOVEMENT     = float(os.getenv("W_RANK_MOVEMENT",     "0.20"))
W_TREND_PERSISTENCE = float(os.getenv("W_TREND_PERSISTENCE", "0.15"))
W_ACCOUNT_PERF      = float(os.getenv("W_ACCOUNT_PERF",      "0.25"))
W_EXTERNAL_POP      = float(os.getenv("W_EXTERNAL_POP",      "0.10"))
EXPLORATION_RATE    = float(os.getenv("EXPLORATION_RATE",    "0.10"))

# Music history persistence
MUSIC_HISTORY_PATH       = Path("assets/music_history.json")
_USED_SONGS_LEGACY       = Path("assets/used_songs.json")
LASTFM_CACHE_TTL_DAYS    = int(os.getenv("LASTFM_CACHE_TTL_DAYS", "7"))
CANDIDATE_HISTORY_WINDOW = int(os.getenv("CANDIDATE_HISTORY_WINDOW", "10"))

# Selection mode
# "auto"   — intelligent scoring pipeline (default / scheduled runs)
# "manual" — caller supplies an explicit audio_id
SELECTION_MODE  = os.getenv("SELECTION_MODE",  "auto").strip().lower()
MANUAL_AUDIO_ID = os.getenv("MANUAL_AUDIO_ID", "").strip()


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

    caption: str

    status: str
    error: str

    # ── New keys added by music-selection system ───────────
    selected_score:      float   # final score (None for manual)
    selection_reason:    str     # scoring breakdown or "manual"
    selection_mode:      str     # "auto" | "manual"
    last_candidate_keys: list    # all candidate keys from this run


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
# MUSIC HISTORY HELPERS
# ============================================================

def _normalise(s: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", s.lower())).strip()


def _song_key(track: dict, lastfm: Optional[dict] = None) -> str:
    """
    Build a stable deduplication key for a track.
    Hierarchy: Meta audio ID > MusicBrainz ID (from Last.fm) > normalised text.
    """
    meta_id = track.get("audio_id") or track.get("id")
    if meta_id:
        return f"meta:{meta_id}"
    if lastfm and lastfm.get("mbid"):
        return f"mbid:{lastfm['mbid']}"
    title  = _normalise(track.get("title") or track.get("name") or "")
    artist = _normalise(track.get("artist") or "")
    return f"text:{title}|{artist}"


def load_history() -> dict:
    """Load music_history.json; auto-migrates from used_songs.json on first run."""
    if MUSIC_HISTORY_PATH.exists():
        with open(MUSIC_HISTORY_PATH, encoding="utf-8") as f:
            return json.load(f)

    # First run — create empty history.
    history: Dict[str, Any] = {
        "version":           3,
        "used_song_keys":    [],
        "candidate_history": [],
        "lastfm_cache":      {},
        "reels":             [],
    }

    # Migrate legacy used_songs.json if present.
    if _USED_SONGS_LEGACY.exists():
        with open(_USED_SONGS_LEGACY, encoding="utf-8") as f:
            legacy = json.load(f)
        if isinstance(legacy, list):
            for entry in legacy:
                key = f"text:{_normalise(str(entry))}"
                if key not in history["used_song_keys"]:
                    history["used_song_keys"].append(key)
        print(
            f"[MIGRATE] Imported {len(history['used_song_keys'])} entries "
            f"from used_songs.json → music_history.json"
        )

    save_history(history)
    return history


def save_history(history: dict) -> None:
    """Write music_history.json atomically via a temp file."""
    MUSIC_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = MUSIC_HISTORY_PATH.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    tmp.replace(MUSIC_HISTORY_PATH)


def _needs_snapshot(record: dict, now: datetime) -> bool:
    """True iff this Reel is ≥24 h old and has no frozen 24 h snapshot yet."""
    if record.get("snapshot_24h") is not None:
        return False
    published = record.get("published_at", "")
    if not published:
        return False
    try:
        pub_dt = datetime.fromisoformat(
            published.replace("Z", "+00:00")
        ).replace(tzinfo=None)
        return (now - pub_dt) >= timedelta(hours=24)
    except ValueError:
        return False


def _fetch_insights(reel_id: str) -> dict:
    """Fetch Reel-level insights from the Meta Insights API."""
    result = graph_get(
        f"/{reel_id}/insights",
        {
            "metric": "plays,reach,likes,comments,shares,saved",
            "period": "lifetime",
        },
    )

    # Response: {"data": [{"name": "plays", "values": [{"value": N}]}, ...]}
    data: Dict[str, Any] = {}
    raw = result.get("data", [])
    if isinstance(raw, list):
        for item in raw:
            name   = item.get("name") or item.get("id", "")
            values = item.get("values", [])
            if values:
                data[name] = values[-1].get("value")
            elif "value" in item:
                data[name] = item["value"]
    else:
        data = result

    return {
        "plays":    data.get("plays"),
        "reach":    data.get("reach"),
        "likes":    data.get("likes"),
        "comments": data.get("comments"),
        "shares":   data.get("shares"),
        "saved":    data.get("saved") or data.get("saved_count"),
    }


def _enrich_lastfm(
    title: str,
    artist: str,
    api_key: str,
) -> Optional[dict]:
    """
    Fetch track metadata from Last.fm track.getInfo.
    Returns None on any failure or no match — never raises.
    Does NOT download audio. API key is never logged.
    """
    if not api_key or not title:
        return None

    params = {
        "method":      "track.getInfo",
        "api_key":     api_key,          # never printed
        "artist":      artist,
        "track":       title,
        "format":      "json",
        "autocorrect": 1,
    }

    def _call() -> Optional[dict]:
        try:
            resp = requests.get(
                "https://ws.audioscrobbler.com/2.0/",
                params=params,
                timeout=10,
            )
            if resp.status_code == 429:
                time.sleep(2)
                resp = requests.get(
                    "https://ws.audioscrobbler.com/2.0/",
                    params=params,
                    timeout=10,
                )
            return resp.json()
        except Exception as exc:
            print(f"  [LASTFM] HTTP error for '{title}': {exc}")
            return None

    body = _call()
    if body is None or "error" in body or "track" not in body:
        return None

    t = body["track"]

    playcount = int(t.get("playcount") or 0) or None
    listeners = int(t.get("listeners") or 0) or None
    tags      = [
        tag["name"]
        for tag in t.get("toptags", {}).get("tag", [])[:5]
        if tag.get("name")
    ]
    release_date = (
        (t.get("album") or {}).get("releasedate", "")
    ).strip() or None

    return {
        "cached_at":    datetime.utcnow().isoformat() + "Z",
        "provider":     "lastfm",
        "mbid":         t.get("mbid") or None,
        "playcount":    playcount,
        "listeners":    listeners,
        "tags":         tags,
        "release_date": release_date,
    }


def _rank_in_previous_run(
    key: str,
    candidate_history: list,
) -> Optional[int]:
    """
    Return the 1-based rank of this song in the most recent stored run,
    or None if it was not present.
    NOTE: call this BEFORE appending the current run to candidate_history.
    """
    if not candidate_history:
        return None
    for c in candidate_history[-1].get("candidates", []):
        if c["key"] == key:
            return c["rank"]   # 1-based
    return None


def _persistence_score(
    key: str,
    candidate_history: list,
    n: int,
) -> float:
    """
    0–1 score: how consistently + highly ranked this song has been
    across the last CANDIDATE_HISTORY_WINDOW runs.
    """
    if not candidate_history:
        return 0.0

    window = candidate_history[-CANDIDATE_HISTORY_WINDOW:]
    ranks  = []

    for run in window:
        for c in run.get("candidates", []):
            if c["key"] == key:
                ranks.append(c["rank"])   # 1-based
                break

    if not ranks:
        return 0.0

    freq           = len(ranks) / len(window)                      # 0–1
    avg_rank_score = 1.0 - ((sum(ranks) / len(ranks) - 1) / max(n - 1, 1))
    return (freq + avg_rank_score) / 2.0


def _score_candidate(
    track: dict,
    key: str,
    rank_0based: int,
    n: int,
    candidate_history: list,
    account_avg_eng: float,
    confidence: float,
) -> dict:
    """
    Calculate all five scoring signals for one candidate.
    Returns a dict of component scores and the total.
    """
    # A — Instagram Trend Rank (30%)
    rank_score = 1.0 - (rank_0based / max(n - 1, 1))
    A = W_TREND_RANK * rank_score

    # B — Instagram Rank Movement (20%)
    prev_rank = _rank_in_previous_run(key, candidate_history)
    if prev_rank is None:
        movement_score = 0.5   # new this run — neutral
    else:
        current_rank = rank_0based + 1   # 1-based
        delta        = prev_rank - current_rank   # positive = moved up
        max_delta    = n - 1
        movement_score = max(0.0, min(1.0, (delta + max_delta) / (2 * max_delta)))
    B = W_RANK_MOVEMENT * movement_score

    # C — Instagram Trend Persistence (15%)
    persist = _persistence_score(key, candidate_history, n)
    C = W_TREND_PERSISTENCE * persist

    # D — My-account Performance (25%)
    # Scored at account-average level (no artist/duration proxies).
    # Exact song-match is impossible for new songs — baseline = account avg.
    if account_avg_eng > 0 and confidence > 0:
        D = W_ACCOUNT_PERF * confidence * min(account_avg_eng / 0.20, 1.0)
    else:
        D = 0.0

    # E — External Popularity via Last.fm (10%)
    lastfm = track.get("_lastfm")
    if lastfm:
        listeners = lastfm.get("listeners") or 0
        playcount  = lastfm.get("playcount") or 0
        if listeners > 0:
            pop_raw = math.log10(listeners + 1) / math.log10(5_000_000)
        elif playcount > 0:
            pop_raw = math.log10(playcount + 1) / math.log10(500_000_000)
        else:
            pop_raw = 0.0
        E = W_EXTERNAL_POP * min(pop_raw, 1.0)
    else:
        E = 0.0

    return {
        "A":             A,
        "B":             B,
        "C":             C,
        "D":             D,
        "E":             E,
        "total":         A + B + C + D + E,
        "rank_score":    rank_score,
        "move_score":    movement_score,
        "persist_score": persist,
        "prev_rank":     prev_rank,
    }


# ============================================================
# NODE 0
# SNAPSHOT MATURE REELS (≥24 h, one-time, never overwrite)
# ============================================================

def snapshot_mature_reels(state: ReelState) -> ReelState:
    """
    For every Reel that has reached ≥24 h old and has no frozen snapshot yet,
    fetch its Meta Insights and lock the snapshot permanently.
    Non-fatal: failures are logged and retried on the next run.
    """
    print("\n=== SNAPSHOT MATURE REELS ===")

    history       = load_history()
    now           = datetime.utcnow()
    snapped       = 0
    too_young     = 0
    already_done  = 0

    for record in history.get("reels", []):
        if record.get("snapshot_24h") is not None:
            already_done += 1
            continue

        if not _needs_snapshot(record, now):
            too_young += 1
            continue

        reel_id = record.get("reel_id", "")
        title   = record.get("title", "")

        try:
            insights = _fetch_insights(reel_id)
            record["snapshot_24h"] = {
                "fetched_at": now.isoformat() + "Z",
                "plays":      insights.get("plays"),
                "reach":      insights.get("reach"),
                "likes":      insights.get("likes"),
                "comments":   insights.get("comments"),
                "shares":     insights.get("shares"),
                "saved":      insights.get("saved"),
            }
            snapped += 1
            print(
                f"  [SNAPPED] '{title}' reel={reel_id}"
                f"  plays={insights.get('plays')}"
                f"  reach={insights.get('reach')}"
                f"  likes={insights.get('likes')}"
                f"  [LOCKED FOREVER]"
            )
        except Exception as exc:
            # Non-fatal — try again next run (snapshot_24h stays None)
            print(f"  [WARNING] Could not snapshot reel {reel_id}: {exc}")

    save_history(history)
    print(
        f"  Snapshotted: {snapped}"
        f"  | Too young: {too_young}"
        f"  | Already done: {already_done}"
    )
    return state   # ReelState is unchanged — pure side-effect node


# ============================================================
# NODE 1
# SELECT MUSIC  (replaces find_trending_audio)
# ============================================================

def select_music(state: ReelState) -> ReelState:
    """
    1. Fetch ~20 Meta trending candidates.
    2. Enrich each with Last.fm metadata (cached, non-fatal).
    3. Hard-exclude every previously used song.
    4. Score remaining candidates on 5 signals.
    5. Select the top scorer (with small exploration randomness).
    Returns the same state keys as the old find_trending_audio node.
    """
    print("\n=== SELECT MUSIC ===")

    # ── Step A: Fetch Meta candidates ────────────────────────
    result = graph_get(
        "/ig_audio",
        {
            "ig_user_id": IG_USER_ID,
            "audio_type": AUDIO_TYPE,
            "limit":      AUDIO_LIMIT,
        },
    )

    print(json.dumps(result, indent=2))

    raw = (
        result.get("audio")
        or result.get("data")
        or result.get("items")
    )
    if raw is None and isinstance(result, list):
        raw = result

    all_tracks = [
        item
        for item in (raw or [])
        if isinstance(item, dict)
        and (item.get("audio_id") or item.get("id"))
    ]

    if not all_tracks:
        raise RuntimeError(
            "Meta returned no Instagram catalog audio.\n"
            f"Raw response: {json.dumps(result, indent=2)}"
        )

    print(f"\nMeta returned {len(all_tracks)} candidate(s).")

    # ── Step B: Record candidate ranks for this run ───────────
    # (used for rank-movement and persistence scoring in future runs)
    run_at = datetime.utcnow().isoformat() + "Z"
    current_run_candidates = [
        {"key": _song_key(t), "rank": idx + 1}   # 1-based
        for idx, t in enumerate(all_tracks)
    ]

    # ── Step C: Enrich with Last.fm ───────────────────────────
    history = load_history()
    print("\n--- Last.fm Enrichment ---")

    if not LASTFM_API_KEY:
        print("  LASTFM_API_KEY not set — skipping enrichment (non-fatal).")

    for track in all_tracks:
        key    = _song_key(track)
        title  = track.get("title") or track.get("name") or ""
        artist = track.get("artist") or ""

        # Check cache first
        cached = history.get("lastfm_cache", {}).get(key)
        if cached:
            cached_at  = datetime.fromisoformat(
                cached["cached_at"].replace("Z", "+00:00")
            ).replace(tzinfo=None)
            age_days   = (datetime.utcnow() - cached_at).days
            if age_days <= LASTFM_CACHE_TTL_DAYS:
                track["_lastfm"] = cached
                print(
                    f"  [CACHE] '{title}'"
                    f"  listeners={cached.get('listeners')}"
                )
                continue

        # Fetch from Last.fm
        if LASTFM_API_KEY:
            track["_lastfm"] = _enrich_lastfm(title, artist, LASTFM_API_KEY)
            if track["_lastfm"]:
                history.setdefault("lastfm_cache", {})[key] = track["_lastfm"]
                print(
                    f"  [LASTFM] '{title}'"
                    f"  listeners={track['_lastfm'].get('listeners')}"
                    f"  playcount={track['_lastfm'].get('playcount')}"
                    f"  tags={track['_lastfm'].get('tags')}"
                )
            else:
                track["_lastfm"] = None
                print(f"  [LASTFM] '{title}': no match")
        else:
            track["_lastfm"] = None

    # ── Step D: Hard-exclude used songs ──────────────────────
    used_keys = set(history.get("used_song_keys", []))
    fresh     = []

    print("\n--- Used-song Filter ---")
    for track in all_tracks:
        lastfm = track.get("_lastfm")
        key    = _song_key(track, lastfm)
        title  = track.get("title") or track.get("name") or "Unknown"
        artist = track.get("artist") or ""

        # Check all three key tiers
        meta_id = track.get("audio_id") or track.get("id")
        mbid    = (lastfm or {}).get("mbid")
        rejected = (
            key in used_keys
            or (meta_id and f"meta:{meta_id}" in used_keys)
            or (mbid    and f"mbid:{mbid}"    in used_keys)
        )

        if rejected:
            print(f"  REJECTED (already used): '{title}' by {artist} [key={key}]")
        else:
            track["_key"] = key
            fresh.append(track)

    if not fresh:
        raise RuntimeError(
            "All Meta candidates have already been used. "
            "No new song available this run. "
            "Wait for Meta to refresh its catalog."
        )

    print(f"  {len(fresh)} candidate(s) remain after exclusions.")

    # ── Step E: Score each remaining candidate ────────────────
    matured = [
        r for r in history.get("reels", [])
        if r.get("snapshot_24h") is not None
    ]

    # Account-wide engagement average (song-level learning — no artist/duration proxy)
    if matured:
        eng_rates = []
        for r in matured:
            snap   = r["snapshot_24h"]
            reach  = snap.get("reach") or 1
            total_eng = (
                (snap.get("likes")    or 0)
                + (snap.get("comments") or 0)
                + (snap.get("shares")   or 0)
                + (snap.get("saved")    or 0)
            )
            eng_rates.append(total_eng / reach)
        account_avg_eng = sum(eng_rates) / len(eng_rates)
    else:
        account_avg_eng = 0.0

    confidence       = min(1.0, len(matured) / 10.0)
    n                = len(all_tracks)
    cand_history     = history.get("candidate_history", [])

    for track in fresh:
        key       = track["_key"]
        rank_idx  = next(
            (i for i, t in enumerate(all_tracks)
             if _song_key(t) == key),
            0,
        )
        track["_scores"] = _score_candidate(
            track          = track,
            key            = key,
            rank_0based    = rank_idx,
            n              = n,
            candidate_history = cand_history,
            account_avg_eng   = account_avg_eng,
            confidence        = confidence,
        )
        track["_final_score"] = track["_scores"]["total"]
        track["_rank_1based"] = rank_idx + 1

    ranked = sorted(fresh, key=lambda t: t["_final_score"], reverse=True)

    # Log all scores
    print("\n=== CANDIDATE SCORING RESULTS ===")
    for i, t in enumerate(ranked):
        s         = t["_scores"]
        listeners = (t.get("_lastfm") or {}).get("listeners")
        prev_r    = s["prev_rank"]
        move_str  = (
            f"(was #{prev_r})" if prev_r else "(new this run)"
        )
        print(
            f"  #{i+1:02d} '{t.get('title') or t.get('name')}'"
            f" | score={t['_final_score']:.3f}"
            f" | A={s['A']:.3f} B={s['B']:.3f}"
            f" C={s['C']:.3f} D={s['D']:.3f} E={s['E']:.3f}"
            f" | rank=#{t['_rank_1based']} {move_str}"
            f" | listeners={listeners}"
        )

    # ── Step F: Exploration ───────────────────────────────────
    if random.random() < EXPLORATION_RATE and len(ranked) >= 2:
        pool     = ranked[:min(5, len(ranked))]
        selected = random.choice(pool)
        expl_str = f"EXPLORATION (from top-{len(pool)} pool)"
    else:
        selected = ranked[0]
        expl_str = "deterministic (top scorer)"

    sel_title  = selected.get("title") or selected.get("name") or "Unknown"
    sel_artist = selected.get("artist") or ""
    sel_s      = selected["_scores"]
    sel_key    = selected["_key"]
    sel_rank   = selected["_rank_1based"]
    listeners  = (selected.get("_lastfm") or {}).get("listeners")

    reason = (
        f"rank=#{sel_rank}"
        f" prev_rank={sel_s['prev_rank']}"
        f" A={sel_s['A']:.3f}"
        f" B={sel_s['B']:.3f}"
        f" C={sel_s['C']:.3f}"
        f" D={sel_s['D']:.3f}"
        f" E={sel_s['E']:.3f}"
        f" listeners={listeners}"
        f" mode={expl_str}"
    )

    print(f"\nSELECTED: '{sel_title}' by {sel_artist}")
    print(f"  key={sel_key}")
    print(f"  score={selected['_final_score']:.3f}")
    print(f"  {reason}")

    if not selected.get("download_url"):
        print(
            "WARNING: Selected track has no download_url. "
            "Music portion in local video will be silent, "
            "but audio_id will still be attached on Instagram."
        )

    # ── Save candidate history for this run ───────────────────
    history.setdefault("candidate_history", []).append({
        "run_at":     run_at,
        "candidates": current_run_candidates,
    })
    history["candidate_history"] = (
        history["candidate_history"][-CANDIDATE_HISTORY_WINDOW:]
    )
    save_history(history)

    # ── Extract audio fields (same contract as old node) ──────
    audio_id = selected.get("id") or selected.get("audio_id")
    if not audio_id:
        raise RuntimeError("Could not find audio ID in selected track.")

    duration_ms = (
        selected.get("duration_in_ms")
        or selected.get("duration_ms")
    )
    duration_s_field = selected.get("duration")

    if duration_ms is not None:
        duration = float(duration_ms) / 1000.0
    elif duration_s_field is not None:
        duration = float(duration_s_field)
    else:
        raise RuntimeError(
            f"No duration found in selected track: "
            f"{json.dumps({k: v for k, v in selected.items() if not k.startswith('_')}, indent=2)}"
        )

    print(f"  Audio duration: {duration:.2f}s")

    # Tag the selected track with its rank so record_new_reel can store it
    selected["_meta_rank"] = sel_rank

    return {
        **state,
        "audio_id":           str(audio_id),
        "audio_title":        sel_title,
        "audio_metadata":     selected,
        "audio_duration":     duration,
        "selected_score":     selected["_final_score"],
        "selection_reason":   reason,
        "last_candidate_keys": [c["key"] for c in current_run_candidates],
    }



# ============================================================
# NODE 2
# GENERATE SHORT TRENDING CAPTION
# ============================================================

# Seed hashtag queries to search on Meta (music / song niche)
_HASHTAG_SEEDS = [
    "songs", "music", "viral", "reels", "trending",
    "fyp", "explorepage", "newmusic",
]

# Always append these regardless of API result
_FIXED_TAGS = ["#reels", "#songs", "#viral"]


def _fetch_hashtag_id(tag: str) -> Optional[str]:
    """Return Meta hashtag object ID for a given tag name, or None."""
    try:
        result = graph_get(
            "/ig_hashtag_search",
            {
                "user_id": IG_USER_ID,
                "q": tag,
            },
        )
        data = result.get("data", [])
        if data and isinstance(data, list):
            return data[0].get("id")
    except Exception as exc:
        print(f"  hashtag search failed for '{tag}': {exc}")
    return None


def _fetch_hashtag_media_count(hashtag_id: str) -> int:
    """Return media_count for a hashtag ID, or 0 on error."""
    try:
        result = graph_get(
            f"/{hashtag_id}",
            {"fields": "name,media_count"},
        )
        return int(result.get("media_count") or 0)
    except Exception:
        return 0


def _build_short_hook(audio_title: str) -> str:
    """
    Build a ≤7-word hook from the audio title.
    e.g. 'Blinding Lights' -> '🎵 Blinding Lights on repeat ✨'
    """
    # Strip featured-artist suffixes like '(feat. ...)'  / '[prod. ...]'
    import re
    clean = re.sub(
        r"[\(\[][^)\]]*[\)\]]",
        "",
        audio_title,
    ).strip()

    words = clean.split()

    # Keep at most 4 title words so hook stays within 6-7 total
    title_part = " ".join(words[:4])

    hook = f"🎵 {title_part} on repeat ✨"

    return hook


def generate_caption(state: ReelState):

    print("\n=== GENERATE CAPTION ===")

    audio_title = state.get("audio_title", "")

    # --- short hook (≤7 words) ---
    hook = _build_short_hook(audio_title)
    print(f"  Hook: {hook}")

    # --- fetch trending hashtags from Meta ---
    tag_data: list[tuple[int, str]] = []  # (media_count, "#name")

    for seed in _HASHTAG_SEEDS:
        hid = _fetch_hashtag_id(seed)
        if not hid:
            continue
        count = _fetch_hashtag_media_count(hid)
        tag_data.append((count, f"#{seed}"))
        print(f"  #{seed}: {count:,} posts")

    # Sort by media_count descending, keep top 4
    tag_data.sort(key=lambda x: x[0], reverse=True)
    top_tags = [t for _, t in tag_data[:4]]

    # Merge with fixed tags, deduplicate, keep order
    all_tags: list[str] = []
    seen: set[str] = set()
    for t in top_tags + _FIXED_TAGS:
        if t not in seen:
            all_tags.append(t)
            seen.add(t)

    hashtags = " ".join(all_tags)

    # Final caption: hook + blank line + hashtags
    caption = f"{hook}\n\n{hashtags}"

    print(f"\n  Caption:\n{caption}")

    return {
        **state,
        "caption": caption,
    }


# ============================================================
# NODE 3
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

        # Scale to exact portrait (9:16) dimensions for Instagram Reels.
        # 'force_original_aspect_ratio=increase' scales up so both dimensions
        # meet the target, then 'crop' center-crops the overshoot.
        # Result: frame is always 100% filled — no black bars.
        run([
            "ffmpeg", "-y",
            "-i", asset,
            "-vf",
            (
                f"scale={WIDTH}:{HEIGHT}:"
                "force_original_aspect_ratio=increase,"
                f"crop={WIDTH}:{HEIGHT},"
                "setsar=1"
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

        # Extract asset audio for the intro segment and boost its volume
        # so the original clip audio is clearly audible in the first 5s.
        run([
            "ffmpeg", "-y",
            "-i", str(normalized),
            "-t", str(MUSIC_START_SECONDS),
            "-vn",
            "-af", f"volume={ASSET_AUDIO_BOOST}",
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

        # Final mux: enforce exact 1080x1920 portrait output for Instagram Reels.
        # scale+crop fills the frame completely — no black bars.
        run([
            "ffmpeg", "-y",
            "-i", str(concatenated),
            "-i", str(mixed_audio),
            "-t", str(final_duration),
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-vf",
            (
                f"scale={WIDTH}:{HEIGHT}:"
                "force_original_aspect_ratio=increase,"
                f"crop={WIDTH}:{HEIGHT},"
                "setsar=1"
            ),
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
        # Use dynamically generated caption; fall back to env var CAPTION
        "caption": state.get("caption") or CAPTION,
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
# NODE — RECORD NEW REEL
# (runs after publish_reel, before END)
# ============================================================

def record_new_reel(state: ReelState) -> ReelState:
    """
    Save the initial history record for the just-published Reel.
    Makes ZERO API calls — insights are fetched later by snapshot_mature_reels
    on the first run that executes ≥24 hours after published_at.
    """
    print("\n=== RECORD NEW REEL ===")

    media_id = state.get("media_id")

    if not media_id:
        print("Dry run — PUBLISH_REEL=false. Skipping record.")
        return state

    now        = datetime.utcnow().isoformat() + "Z"
    audio_meta = state.get("audio_metadata") or {}
    lastfm     = audio_meta.get("_lastfm")
    key        = _song_key(audio_meta, lastfm)

    record: Dict[str, Any] = {
        "song_id":          state.get("audio_id", ""),
        "song_key":         key,
        "title":            state.get("audio_title", ""),
        "artist":           audio_meta.get("artist", ""),
        "duration_s":       state.get("audio_duration", 0.0),
        "published_at":     now,
        "reel_id":          media_id,
        "selection_mode":   state.get("selection_mode", SELECTION_MODE),
        "selected_score":   state.get("selected_score"),
        "selection_reason": state.get("selection_reason", ""),
        "meta_rank":        audio_meta.get("_meta_rank"),
        "external_metadata": (
            {k: v for k, v in lastfm.items() if k != "cached_at"}
            if lastfm else None
        ),
        "snapshot_24h":     None,   # filled by snapshot_mature_reels on future run
    }

    history = load_history()
    history.setdefault("reels", []).append(record)

    # Register all three key tiers so the song can never be selected again
    used = set(history.get("used_song_keys", []))
    used.add(key)
    meta_id = audio_meta.get("audio_id") or audio_meta.get("id")
    if meta_id:
        used.add(f"meta:{meta_id}")
    if lastfm and lastfm.get("mbid"):
        used.add(f"mbid:{lastfm['mbid']}")
    history["used_song_keys"] = list(used)

    # Update last_candidate_keys in history
    history["last_candidate_keys"] = state.get("last_candidate_keys", [])

    save_history(history)

    print(f"  Recorded: reel_id={media_id}")
    print(f"  Mode:     {record['selection_mode']}")
    print(f"  Song:     '{record['title']}' by {record['artist']}")
    print(f"  Key:      {key}")
    print(
        f"  24 h snapshot will be taken on the first run "
        f"after {now} +24 h"
    )
    return state


# ============================================================
# NODE 0B
# RESOLVE MANUAL AUDIO  (MANUAL mode only)
# ============================================================

def resolve_manual_audio(state: ReelState) -> ReelState:
    """
    MANUAL mode entry point.
    Uses the audio_id supplied by the operator.
    1. Validates the audio_id is present.
    2. Checks it has not already been used (hard exclusion).
    3. Attempts to resolve metadata from Meta Graph API.
    4. Falls back to null/unknown values if Meta cannot resolve it.
    5. Returns the same state keys as select_music() so the rest
       of the pipeline is completely unchanged.
    """
    print("\n=== RESOLVE MANUAL AUDIO ===")
    print(f"  Mode:     MANUAL")
    print(f"  audio_id: {MANUAL_AUDIO_ID}")

    if not MANUAL_AUDIO_ID:
        raise RuntimeError(
            "MANUAL mode: MANUAL_AUDIO_ID env var is empty. "
            "Supply an Instagram audio ID when triggering the workflow."
        )

    # ── Check used-song exclusion ────────────────────────────
    history   = load_history()
    used_keys = set(history.get("used_song_keys", []))

    candidate_key = f"meta:{MANUAL_AUDIO_ID}"
    if candidate_key in used_keys or MANUAL_AUDIO_ID in used_keys:
        raise RuntimeError(
            f"MANUAL mode: audio_id '{MANUAL_AUDIO_ID}' has already been "
            f"published and is permanently excluded from reuse. "
            f"Choose a different audio ID."
        )

    # ── Try to resolve metadata from Meta Graph API ─────────────
    # We call the existing graph_get() on the audio node.
    # This may or may not return rich metadata depending on permissions.
    # We NEVER invent data; unknown fields stay None.
    title    = None
    artist   = None
    duration = None
    metadata: Dict[str, Any] = {"id": MANUAL_AUDIO_ID}

    try:
        result = graph_get(
            f"/{MANUAL_AUDIO_ID}",
            {
                "fields": (
                    "id,title,name,artist,"
                    "duration_in_ms,duration_ms,duration,"
                    "download_url"
                )
            },
        )
        metadata = result

        title = (
            result.get("title")
            or result.get("name")
        )
        artist = result.get("artist")

        duration_ms = (
            result.get("duration_in_ms")
            or result.get("duration_ms")
        )
        if duration_ms is not None:
            duration = float(duration_ms) / 1000.0
        elif result.get("duration") is not None:
            duration = float(result["duration"])

        print(f"  Meta resolved: title='{title}' artist='{artist}' duration={duration}s")

    except Exception as exc:
        # Non-fatal — we still have the audio_id which is enough to publish
        print(
            f"  WARNING: Could not resolve metadata for audio_id "
            f"'{MANUAL_AUDIO_ID}': {exc}\n"
            f"  Proceeding with null metadata."
        )

    if duration is None:
        raise RuntimeError(
            f"MANUAL mode: could not determine audio duration for "
            f"audio_id '{MANUAL_AUDIO_ID}'. "
            f"Meta did not return duration_in_ms, duration_ms, or duration. "
            f"Cannot calculate video length without duration."
        )

    display_title  = title  or "(unknown title)"
    display_artist = artist or "(unknown artist)"

    print(f"  MANUAL SELECTED: '{display_title}' by {display_artist}")
    print(f"  Duration: {duration:.2f}s")

    if not metadata.get("download_url"):
        print(
            "  WARNING: No download_url available for this audio_id. "
            "Music portion in local video will be silent, "
            "but audio_id will still be attached on Instagram."
        )

    # Tag metadata so record_new_reel knows this is manual
    metadata["_meta_rank"]      = None
    metadata["_lastfm"]         = None
    metadata["_selection_mode"] = "manual"

    return {
        **state,
        "audio_id":           MANUAL_AUDIO_ID,
        "audio_title":        display_title,
        "audio_metadata":     metadata,
        "audio_duration":     duration,
        "selected_score":     None,
        "selection_reason":   "manual",
        "selection_mode":     "manual",
        "last_candidate_keys": [],
    }


# ============================================================
# LANGGRAPH
# ============================================================


def _route_selection(state: ReelState) -> str:
    """
    Conditional router:
      SELECTION_MODE == "manual" → resolve_manual_audio
      anything else             → select_music  (AUTO)
    """
    if SELECTION_MODE == "manual":
        print("[ROUTER] mode=manual → resolve_manual_audio")
        return "resolve_manual_audio"
    print("[ROUTER] mode=auto → select_music")
    return "select_music"


def build_graph():

    graph = StateGraph(ReelState)

    # ── Selection nodes (one or the other runs, never both) ──
    graph.add_node("snapshot_mature_reels", snapshot_mature_reels)
    graph.add_node("select_music",          select_music)          # AUTO
    graph.add_node("resolve_manual_audio",  resolve_manual_audio)  # MANUAL
    graph.add_node("record_new_reel",       record_new_reel)

    # ── Shared pipeline nodes (logic unchanged) ──────────────
    graph.add_node("generate_caption",        generate_caption)
    graph.add_node("calculate_video_duration", calculate_video_duration)
    graph.add_node("build_video",             build_video)
    graph.add_node("upload_to_drive",         upload_to_drive)
    graph.add_node("check_public_video_url",  check_public_video_url)
    graph.add_node("create_reel_container",   create_reel_container)
    graph.add_node("wait_for_container",      wait_for_container)
    graph.add_node("publish_reel",            publish_reel)

    # ── Entry point ───────────────────────────────────────────
    graph.set_entry_point("snapshot_mature_reels")

    # ── Conditional routing: AUTO vs MANUAL ──────────────────
    #    Both paths converge at generate_caption.
    graph.add_conditional_edges(
        "snapshot_mature_reels",
        _route_selection,
        {
            "select_music":         "select_music",
            "resolve_manual_audio": "resolve_manual_audio",
        },
    )

    # ── Convergence: both selection nodes → same pipeline ───
    graph.add_edge("select_music",         "generate_caption")
    graph.add_edge("resolve_manual_audio", "generate_caption")

    # ── Shared pipeline edges (unchanged) ───────────────────
    graph.add_edge("generate_caption",         "calculate_video_duration")
    graph.add_edge("calculate_video_duration", "build_video")
    graph.add_edge("build_video",              "upload_to_drive")
    graph.add_edge("upload_to_drive",          "check_public_video_url")
    graph.add_edge("check_public_video_url",   "create_reel_container")
    graph.add_edge("create_reel_container",    "wait_for_container")
    graph.add_edge("wait_for_container",       "publish_reel")
    graph.add_edge("publish_reel",             "record_new_reel")
    graph.add_edge("record_new_reel",          END)

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