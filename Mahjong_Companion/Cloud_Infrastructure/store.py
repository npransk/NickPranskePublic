# Showcase copy of services/api/src/mahjong_api/store.py from the private mahjong-companion repo. Here to be read, not run.

"""Where games and the card live: a local folder, or the private blob storage in Azure.

Names look like ``games/<id>.json`` and ``cards/2024.toml``; the first part is the blob container
(or the subfolder locally). Locally the root is ``data/private``, which is gitignored, so your
card file is already in place at ``data/private/cards/2024.toml``.

Set ``STORAGE_ACCOUNT_URL`` to use blob storage. In Azure the Container App's managed identity
signs in; locally ``az login`` does.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Protocol


class Store(Protocol):
    def get(self, name: str) -> str | None: ...
    def put(self, name: str, text: str) -> None: ...
    def put_bytes(self, name: str, data: bytes, content_type: str) -> None: ...
    def list(self, folder: str) -> list[str]: ...


class LocalStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def get(self, name: str) -> str | None:
        path = self.root / name
        return path.read_text(encoding="utf-8") if path.is_file() else None

    def put(self, name: str, text: str) -> None:
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def put_bytes(self, name: str, data: bytes, content_type: str) -> None:
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def list(self, folder: str) -> list[str]:
        path = self.root / folder
        return sorted(f"{folder}/{p.name}" for p in path.iterdir()) if path.is_dir() else []


class BlobStore:
    def __init__(self, account_url: str):
        from azure.identity import DefaultAzureCredential
        from azure.storage.blob import BlobServiceClient

        self.client = BlobServiceClient(account_url, credential=DefaultAzureCredential())

    def _split(self, name: str) -> tuple[str, str]:
        container, _, blob = name.partition("/")
        return container, blob

    def get(self, name: str) -> str | None:
        from azure.core.exceptions import ResourceNotFoundError

        container, blob = self._split(name)
        try:
            return self.client.get_blob_client(container, blob).download_blob().readall().decode()
        except ResourceNotFoundError:
            return None

    def put(self, name: str, text: str) -> None:
        container, blob = self._split(name)
        self.client.get_blob_client(container, blob).upload_blob(text, overwrite=True)

    def put_bytes(self, name: str, data: bytes, content_type: str) -> None:
        from azure.storage.blob import ContentSettings

        container, blob = self._split(name)
        self.client.get_blob_client(container, blob).upload_blob(
            data, overwrite=True, content_settings=ContentSettings(content_type=content_type)
        )

    def list(self, folder: str) -> list[str]:
        container = self.client.get_container_client(folder)
        return sorted(f"{folder}/{b.name}" for b in container.list_blobs())


def store_from_env() -> Store:
    if url := os.environ.get("STORAGE_ACCOUNT_URL"):
        return BlobStore(url)
    return LocalStore(os.environ.get("DATA_DIR", "data/private"))
