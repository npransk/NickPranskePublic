# Showcase copy of services/api/tests/conftest.py from the private mahjong-companion repo. Here to be read, not run.

"""Shared test plumbing: no real Langfuse, and a stand-in that records what would be sent."""

from contextlib import contextmanager

import pytest
from mahjong_api import card_draft, games, listen, observability, playbyplay, strategist, vision

LANGFUSE_SETTINGS = (
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_SECRET_KEY",
    "LANGFUSE_PUBLIC_KEY_MJ",
    "LANGFUSE_SECRET_KEY_MJ",
    "LANGFUSE_BASE_URL",
    "LANGFUSE_HOST",
    "LANGFUSE_TRACING_ENVIRONMENT",
    "LANGFUSE_RELEASE",
    "CONTAINER_APP_NAME",
)
# Every module that looks the client up by name.
MODULES = (observability, strategist, vision, listen, playbyplay, card_draft, games)
# The real cached functions, kept so their caches can be cleared while a test has them replaced.
CACHED = (observability.langfuse_client, observability.prompt_link)


@pytest.fixture(autouse=True)
def no_langfuse(monkeypatch):
    """No keys, so the real client is switched off: nothing leaves the machine."""
    for name in LANGFUSE_SETTINGS:
        monkeypatch.delenv(name, raising=False)
    for cached in CACHED:
        cached.cache_clear()
    yield
    for cached in CACHED:
        cached.cache_clear()


class FakeObservation:
    """One agent, generation, tool or span, with everything it was given."""

    def __init__(self, **fields):
        self.fields = fields

    def update(self, **fields):
        metadata = {**(self.fields.get("metadata") or {}), **(fields.pop("metadata", None) or {})}
        self.fields.update({key: value for key, value in fields.items() if value is not None})
        if metadata:
            self.fields["metadata"] = metadata

    def __getitem__(self, key):
        return self.fields[key]

    def get(self, key, default=None):
        return self.fields.get(key, default)


class FakeLangfuse:
    """Records observations, events and scores instead of sending them."""

    trace_id = "f" * 32

    def __init__(self):
        self.observations: list[FakeObservation] = []
        self.events: list[dict] = []
        self.scores: list[dict] = []
        self.flushed = 0

    @contextmanager
    def start_as_current_observation(self, **fields):
        observation = FakeObservation(**fields)
        self.observations.append(observation)
        yield observation

    def create_event(self, **fields):
        self.events.append(fields)

    def create_score(self, **fields):
        self.scores.append(fields)

    def get_current_trace_id(self):
        return self.trace_id

    def get_prompt(self, name, **kwargs):
        raise LookupError(f"No prompt {name} registered")

    def flush(self):
        self.flushed += 1

    def named(self, name):
        """The observations with this name, in order."""
        return [o for o in self.observations if o["name"] == name]

    def score(self, name):
        """The latest value written for this score name."""
        return [s["value"] for s in self.scores if s["name"] == name][-1]

    def score_names(self):
        return {s["name"] for s in self.scores}


@pytest.fixture
def langfuse(monkeypatch):
    """A recording stand-in for the Langfuse client, in every module that uses one."""
    fake = FakeLangfuse()
    for module in MODULES:
        if hasattr(module, "langfuse_client"):
            monkeypatch.setattr(module, "langfuse_client", lambda: fake)
    return fake
