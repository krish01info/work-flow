import pytest

import asset_video_workflow as workflow


def test_instagram_upload_requires_credentials(monkeypatch):
    monkeypatch.delenv("INSTAGRAM_ACCOUNT_ID", raising=False)
    monkeypatch.delenv("INSTAGRAM_ACCESS_TOKEN", raising=False)

    with pytest.raises(OSError, match="Instagram credentials are required"):
        workflow.upload_to_instagram({"drive_file_id": "drive-file-id"})


def test_instagram_upload_requires_a_drive_video(monkeypatch):
    monkeypatch.setenv("INSTAGRAM_ACCOUNT_ID", "account-id")
    monkeypatch.setenv("INSTAGRAM_ACCESS_TOKEN", "token")

    with pytest.raises(RuntimeError, match="Drive video is required"):
        workflow.upload_to_instagram({"drive_file_id": ""})


def test_instagram_upload_failure_fails_the_pipeline(monkeypatch):
    monkeypatch.setenv("INSTAGRAM_ACCOUNT_ID", "account-id")
    monkeypatch.setenv("INSTAGRAM_ACCESS_TOKEN", "token")

    class FailedResponse:
        def json(self):
            return {"error": {"message": "video URL could not be fetched"}}

    monkeypatch.setattr(workflow.requests, "post", lambda *_args, **_kwargs: FailedResponse())

    with pytest.raises(RuntimeError, match="container initialization failed"):
        workflow.upload_to_instagram({"drive_file_id": "drive-file-id"})
