from pathlib import Path

import asset_video_workflow as workflow
from asset_video_workflow import prepare_music_for_render


def rejected_music_state(path: str = "output_audio/untrusted.mp3") -> dict:
    return {
        "music_path": path,
        "music_attribution": "Music: Untrusted Song",
        "music_metadata": {
            "title": "Untrusted Song",
            "artist": "Unknown Artist",
            "source": "Unverified Source",
            "license": "",
            "license_url": "",
            "attribution_required": False,
            "attribution_text": "",
        },
        "caption_words": [{"text": "UNTRUSTED", "start": 0.0, "end": 1.0}],
    }


def test_rejected_music_is_removed_from_state_before_rendering():
    state = rejected_music_state()

    metadata = prepare_music_for_render(state)

    assert metadata == {}
    assert state["music_path"] == ""
    assert state["music_attribution"] == ""
    assert state["music_metadata"] == {}
    assert state["caption_words"] == []


def test_assemble_validates_music_before_opening_the_audio_file(tmp_path, monkeypatch):
    audio_path = tmp_path / "untrusted.mp3"
    audio_path.write_bytes(b"not real audio")
    state = rejected_music_state(str(audio_path))

    class FakeVideo:
        duration = 10.0
        w = 1080
        h = 1920
        audio = None

        def resized(self, _scale):
            return self

        def cropped(self, **_kwargs):
            return self

        def write_videofile(self, path, **_kwargs):
            Path(path).write_bytes(b"video")

    fake_video = FakeVideo()
    monkeypatch.setattr(workflow, "VideoFileClip", lambda _path: fake_video)
    monkeypatch.setattr(workflow, "_build_extended_video", lambda _clip, _duration: fake_video)
    monkeypatch.setattr(workflow, "_ensure_caption_font", lambda: None)
    monkeypatch.setattr(workflow, "FINAL_DIR", tmp_path / "output")

    def audio_file_must_not_be_opened(_path):
        raise AssertionError("rejected music was opened before validation")

    monkeypatch.setattr(workflow, "AudioFileClip", audio_file_must_not_be_opened)

    workflow.assemble_video(state)

    assert state["music_path"] == ""
    assert state["music_metadata"] == {}
