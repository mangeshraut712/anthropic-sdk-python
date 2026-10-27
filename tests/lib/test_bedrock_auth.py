from __future__ import annotations

import httpx2
import pytest

from anthropic import CredentialsError
from anthropic.lib.bedrock._auth import get_auth_headers


def test_raises_credentials_error_on_missing_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    class NoCredentialsSession:
        region_name = "us-east-1"

        def get_credentials(self) -> None:
            return None

    monkeypatch.setattr(
        "anthropic.lib.bedrock._auth._get_session",
        lambda **_kwargs: NoCredentialsSession(),
    )

    with pytest.raises(CredentialsError, match="could not resolve credentials from session"):
        get_auth_headers(
            method="POST",
            url="https://bedrock-runtime.us-east-1.amazonaws.com/model/x/invoke",
            headers=httpx2.Headers({"connection": "keep-alive"}),
            aws_access_key=None,
            aws_secret_key=None,
            aws_session_token=None,
            region="us-east-1",
            profile=None,
            data="{}",
        )
