# Showcase copy of services/api/src/mahjong_api/speech.py from the private mahjong-companion repo. Here to be read, not run.

"""Short-lived Azure Speech tokens for the phone (Phase 6).

The phone streams microphone audio straight to Azure Speech and gets text back; only the text
reaches this app, as heard lines. The Speech key stays on the server: the phone gets a token
that expires after 10 minutes and asks for a new one while listening. Audio is never stored.
"""

from __future__ import annotations

import os
import urllib.request

TOKEN_MINUTES = 10


class SpeechNotSetUp(RuntimeError):
    """No Speech key or region configured."""


def issue_token() -> dict[str, str | int]:
    key = os.environ.get("AZURE_SPEECH_KEY")
    region = os.environ.get("AZURE_SPEECH_REGION")
    if not key or not region:
        raise SpeechNotSetUp("Listening isn't set up: no AZURE_SPEECH_KEY or AZURE_SPEECH_REGION")
    request = urllib.request.Request(
        f"https://{region}.api.cognitive.microsoft.com/sts/v1.0/issueToken",
        data=b"",
        headers={"Ocp-Apim-Subscription-Key": key},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        token = response.read().decode()
    return {"token": token, "region": region, "expires_in_minutes": TOKEN_MINUTES}
