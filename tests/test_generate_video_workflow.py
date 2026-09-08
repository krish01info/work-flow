from pathlib import Path


def test_used_song_tracker_push_is_not_silenced():
    workflow = Path(".github/workflows/generate-video.yml").read_text(encoding="utf-8")

    assert "git push ||" not in workflow
