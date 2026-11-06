# Handoff: Instagram reel pipeline

Last updated: **2026-11-06**. Pushed that day: the licensed-songs-only picker, 1 post/day, and the auto first comment.
Read this first when picking the project back up. `IMPLEMENTATION.md` and `TECHSTACK.md` are older
than several changes (timings, mood clips); trust the code, `assets/clips/clips.json` and this file.

## What runs

- **Workflow:** `.github/workflows/generate-video.yml`, cron `17 13 * * *` UTC (18:47 IST, 1 post/day since 2026-11-06),
  plus manual `workflow_dispatch` (mode `auto` or `manual` with an audio id or song title).
- **Script:** `reelbuilder.py` (LangGraph). Nothing else in the repo is used by the workflow;
  `asset_video_workflow.py`, `instagram_workflow.py`, `instagrapi_workflow.py`, `youtube_workflow.py`
  and `main.py` are older experiments.
- **Account:** Instagram `@themansquotes`, 21 followers and 133 posts on 2026-10-26.
- **Repo:** https://github.com/krish01info/work-flow (public). Runs:
  https://github.com/krish01info/work-flow/actions/workflows/generate-video.yml

Pipeline order:

1. `snapshot_mature_reels`: 24 h insights for older reels (**currently always fails**, see Open issues 1)
2. `select_music` (auto) or `resolve_manual_audio` (manual): Meta `/ig_audio` candidates scored on signals A–G
3. `choose_clip`: the song's mood picks a clip from `assets/clips/clips.json`
4. `generate_caption`: the hook matches the clip on screen, credits the artist, 5 hashtags
5. `calculate_video_duration` → `build_video` (ffmpeg, 15 s, 1080×1920) → `upload_to_drive`
6. `create_reel_container` (Instagram attaches the song through `audio_configuration`) → `wait_for_container`
   → `publish_reel` → `post_first_comment` → `cleanup_drive` → `record_new_reel` (written to `assets/music_history.json`;
   the bot commits it back after every run)

## Mood clips (added 2026-10-27)

| Song mood | Clip | Source | Framing / notes |
|---|---|---|---|
| sad | `assets/the_transition_i_n_between_is.mp4` | original clip | `split` framing; output byte-identical to before |
| love | `assets/clips/love_a.mp4` | `romantic.mp4` | 16:9 with a square scene in the middle; crop `center_x` 0.516 |
| hype | `assets/clips/hype_a.mp4` | `energetic.mp4` | 9:16; `zoom` 1.18 removes white rounded corners baked into the video |
| party | `assets/clips/party_a.mp4` | `party.mp4` | 16:9; crop `center_x` 0.41 so the son's arms aren't cut |
| unknown (chill) | none yet, **uses the sad clip** | — | the user's decision: don't skip these songs |
| devotional | none, uses the sad clip | — | scored 0 so these songs stay out, as before |

- The new clips (Gemini/Veo, 10 s, 24 fps) use `music_start` 2.5 (the dad's line ends about 2.3–2.4 s),
  `loop_start` 5.0, a 0.5 s `loop_crossfade` into the loop, and their own `audio_boost`
  (1.5 / 1.48 / 1.28, so the line peaks near -1 dB instead of clipping).
- Mood no longer ranks songs: every mood scores 1.0 in signal G (party used to be 0.05 and unknown 0.4).
- With several clips for one mood, `choose_clip` takes the least recently used, read from the `clip`
  field that each reel record now stores.
- The source files came from `C:\Users\rs224\Downloads\New folder`.

### Adding a clip

1. Generate it in Gemini with the prompt kit, which has 5 moods × 4 versions and the analysis of the
   original clip: https://claude.ai/code/artifact/db8c0def-48e7-48d4-937f-80025fcff399
2. Measure it: when the dad's line ends (audio loudness), when the son is revealed, the steady section
   for the loop, any vignette at the edges, and the intro peak after the boost (keep it at -1 dB or lower).
3. Save it as `assets/clips/<mood>_<letter>.mp4` (moods: `sad`, `love`, `hype`, `party`, `unknown`)
   and add an entry to `clips.json`. Fields: `name`, `mood`, `file`, `trim_start`, `music_start`,
   `loop_start`, `loop_crossfade`, `audio_boost`, `thumb_offset_ms`, and `framing`, which is either
   `{"mode": "cover", "center_x", "center_y", "zoom"}` or `{"mode": "split", "split_at", "slice_x0", "slice_x1"}`.
4. Build it offline before pushing. This makes no network calls and posts nothing:

   ```python
   import reelbuilder as rb                      # run from the repo root
   rb.load_history = lambda: {"reels": []}
   st = rb.choose_clip({"song_mood": "love"})
   st.update(output_video="test_love.mp4", audio_duration=60.0)
   rb.build_video(rb.calculate_video_duration(st))
   ```

   Check it with `ffprobe` (15.0 s, 1080x1920, 30 fps) and a contact sheet, e.g.
   `ffmpeg -i test_love.mp4 -vf "fps=1,scale=126:224,tile=15x1" -frames:v 1 sheet.png`.

## Open issues, most urgent first

0. **Status on 2026-11-06:** the last 20 reels got 2 likes and 0 comments in total; 151 posts, 21 followers;
   name still "THE MAN", bio still "songs". Likely flagged as unoriginal. If Account status shows
   "can't be recommended", a fresh account is the realistic route.

1. **Meta token expires 2026-11-07.** The token in the local `.env` expires then; if the
   `META_ACCESS_TOKEN` GitHub secret is the same token, publishing stops that day. It also lacks
   `instagram_manage_insights`, so every insights call fails with `(#10) Application does not have
   permission`, no reel has a 24 h snapshot, and scoring signal D is a constant. Fix: generate a new
   long-lived token with `instagram_manage_insights` and `instagram_manage_comments` (the second is
   for the first-comment idea below) plus the current scopes (`pages_show_list`, `instagram_basic`,
   `instagram_content_publish`, `pages_read_engagement`) and update the secret.
   Check it with `GET /debug_token`.
2. **Chill clip** (mood `unknown`) isn't made yet, and each mood has only one version. Add versions B to D
   so the same clip doesn't come back every few posts.
3. **Low reach.** The diagnosis from 2026-10-26: about 133 reels with near-identical visuals (the same
   clip every time until the mood clips), and the last 27 reels got 8 likes and 1 comment in total, which
   points to Instagram limiting recommendations. The user should check Instagram → Account status →
   "What can't be recommended". More varied visuals (now under way) is the main fix.
4. **Late cron.** GitHub's cron starts 3–7 h late, so posts land around 17:00–19:00 and 22:30–00:10 IST,
   and it dropped a run on 2026-10-13 and 2026-10-19. Fix: trigger `workflow_dispatch` from an external
   scheduler such as cron-job.org.
5. **Song choice** (fixed 2026-10-28, pushed 2026-11-06; until then the bot kept picking Bhojpuri/puja tracks). Meta's `/ig_audio` list starts
   with about 12 licensed songs (they have `on_platform_audio_preview_link`) and pads the rest (~190) with
   royalty-free Sound Collection tracks (they have `download_url`). The picker kept landing on library tracks,
   e.g. "Bindaas (Instrumental)" at rank #49 on 2026-10-27, which won through a false title-only chart match
   with a Tamil "Bindaas" by Anirudh. Now `select_music` keeps only licensed songs (library tracks are a
   fallback), skips tracks longer than `MAX_SONG_SECONDS` (480; kathas and jukeboxes), and chart matches
   need the artist too and never count instrumentals or "Reels Sound" edits. More devotional words
   (katha, chhath, laxmi) keep festival-puja songs out.
6. **Last.fm.** As of 2026-10-26 the cache was empty because the artist was always blank. That's fixed
   in `ad2cb7c`; check the next run's log for `[LASTFM]` hits. It's still unknown whether the
   `LASTFM_API_KEY` secret is set.
7. Minor: captions keep at most 5 hashtags, so the artist tag is dropped whenever there's a mood tag.

## Growth ideas from 2026-10-27 (not built yet)

- **Searchable profile (the user changes this in the app; the API can't edit the profile).**
  On 2026-10-27 the name was "THE MAN" and the bio just "songs", neither of which matches anything
  people search for. Instagram search matches the username, the name field and the bio. Recommended:
  - Name: `Trending Songs | Are Ya Winning Son` (the name can only be changed twice in 14 days)
  - Bio (135 of 150 characters):
    ```
    🎧 a trending song every day
    sad songs • love songs • party songs
    Hindi • Punjabi • English
    dad: are ya winning, son? 👇 request a song
    ```
  - Optional, the user's call: the username `@themansquotes` (left over from the quotes era) is the
    strongest search match but says nothing about songs. Changing it breaks old links and doesn't
    match the YouTube handle.
- **Auto first comment (idea 4): BUILT 2026-11-06** (`post_first_comment`, non-fatal; works once the token has `instagram_manage_comments`). Original plan: `POST /{media-id}/comments?message=...` right after
  `publish_reel`. It needs `instagram_manage_comments` on the token (missing as of 2026-10-27). Plan: a
  non-fatal `post_first_comment` node between `publish_reel` and `cleanup_drive`, rotating questions
  ("rate this song 1–10 👇", "be honest… are YOU winning? 👇") that don't repeat the caption's call to
  action, with the result stored in the reel record. The account's own comment doesn't count as
  engagement; its job is to get viewers replying.
- **Auto-share to Story (idea 5): not possible with the official API.** It can only post the video as a
  separate Story. That Story has no link back to the reel (no reshare, no stickers), so its views don't
  count for the reel, and it can't carry the song, which leaves 12 s of silence after the dad's line.
  Resharing by hand in the app (paper plane → Add to story) takes 5 seconds and does link to the reel.
  The unofficial `instagrapi` route (see `instagrapi_workflow.py`) could automate it but risks the
  account; not recommended.

## Checking the first runs after `ad2cb7c`

- In the run log, `=== CHOOSE CLIP ===` shows the song mood and the clip used.
- `assets/music_history.json` → each reel record now has `clip` and `song_mood`.
- `git pull` first: the bot commits `music_history.json` after every run.

## Local environment notes

- ffmpeg 9 is installed locally (no `-vsync`; use `-fps_mode`); GitHub's runner uses Ubuntu's ffmpeg.
  `drawtext` needs `fontfile=assets/Anton-Regular.ttf` here, because there's no fontconfig.
- Python 3.13 (Anaconda) has the `reelbuilder` dependencies. `gh` isn't installed, and calling `bash`
  from Python reaches a broken WSL, so run shell steps through Git Bash directly.
- `tests/test_instagram_required.py` and `tests/test_music_license_gate.py` fail locally (`faster_whisper`
  isn't installed); this predates the mood clips and is unrelated to `reelbuilder`.
- The GitHub API works without auth (60 requests/hour), but run logs need repo admin auth.
