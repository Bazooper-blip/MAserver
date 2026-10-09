"""Unit tests for the Apple Music setup flow helpers."""

from typing import TYPE_CHECKING, Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import ClientError

from music_assistant.providers.apple_music import setup_flow
from music_assistant.providers.apple_music.constants import (
    CONF_CUSTOM_APP_TOKEN,
    CONF_MUSIC_APP_TOKEN,
    CONF_MUSIC_USER_TOKEN,
)
from music_assistant.providers.apple_music.setup_flow import _app_token_accepted

if TYPE_CHECKING:
    from music_assistant.models.setup_flow import SetupSession


class _FakeRequestCtx:
    """Minimal async context manager mimicking aiohttp's request context."""

    def __init__(self, status: int) -> None:
        self._response = MagicMock(status=status)

    async def __aenter__(self) -> MagicMock:
        return self._response

    async def __aexit__(self, *_exc: object) -> bool:
        return False


def _make_mass(status: int) -> MagicMock:
    mass = MagicMock()
    mass.http_session.get = MagicMock(return_value=_FakeRequestCtx(status))
    return mass


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (200, True),
        (401, False),
        (403, False),
        # a throttled or broken /v1/test says nothing about the token itself
        (429, None),
        (500, None),
        (503, None),
    ],
)
@pytest.mark.asyncio
async def test_app_token_accepted_per_status(status: int, expected: bool | None) -> None:
    """Only an explicit rejection by Apple marks the developer token as invalid."""
    assert await _app_token_accepted(_make_mass(status), "app-token") is expected


@pytest.mark.asyncio
async def test_app_token_accepted_empty_token_is_rejected() -> None:
    """An empty (e.g. not provisioned) token is rejected without calling the API."""
    mass = _make_mass(200)
    assert await _app_token_accepted(mass, "") is False
    mass.http_session.get.assert_not_called()


@pytest.mark.asyncio
async def test_app_token_accepted_network_error_is_inconclusive() -> None:
    """An unreachable API does not make a valid token look invalid."""
    mass = MagicMock()
    mass.http_session.get = MagicMock(side_effect=ClientError("boom"))
    assert await _app_token_accepted(mass, "app-token") is None


class _FakeSession:
    """Setup session stand-in that answers each form from a queue of submissions."""

    def __init__(self, submissions: list[dict[str, Any]], setup_data: dict[str, Any]) -> None:
        self.mass = MagicMock()
        self.context = MagicMock(setup_data=setup_data)
        self.forms: list[tuple[str, list[Any], Any]] = []
        self.finished_with: dict[str, Any] | None = None
        self._submissions = submissions

    async def form(
        self, entries: list[Any], step_id: str = "user", errors: Any = None
    ) -> dict[str, Any]:
        self.forms.append((step_id, entries, errors))
        return self._submissions.pop(0)

    async def finish(self, values: dict[str, Any]) -> dict[str, str]:
        self.finished_with = values
        return {}


async def _run_flow(
    submissions: list[dict[str, Any]],
    accepted: dict[str, bool | None],
    setup_data: dict[str, Any] | None = None,
) -> tuple[_FakeSession, AsyncMock]:
    """Run the setup flow with stubbed token validation and MusicKit sign-in."""
    session = _FakeSession(submissions, setup_data or {})
    musickit = AsyncMock(return_value={"music-user-token": "user-token"})

    async def _accepted(_mass: Any, token: str) -> bool | None:
        return accepted.get(token, False)

    with (
        patch.object(setup_flow, "MUSIC_APP_TOKEN", "bundled"),
        patch.object(setup_flow, "_app_token_accepted", _accepted),
        patch.object(setup_flow, "_musickit_authenticate", musickit),
    ):
        await setup_flow.run_setup(cast("SetupSession", session))
    return session, musickit


def _entry_keys(entries: list[Any]) -> list[str]:
    return [entry.key for entry in entries]


@pytest.mark.asyncio
async def test_setup_uses_bundled_token_when_no_own_token_given() -> None:
    """Leaving the own token empty signs in with, and stores a fallback to, the bundled one."""
    session, musickit = await _run_flow([{}], {"bundled": True})
    assert CONF_CUSTOM_APP_TOKEN in _entry_keys(session.forms[0][1])
    assert musickit.await_args is not None
    assert musickit.await_args.args[1] == "bundled"
    assert session.finished_with is not None
    assert session.finished_with[CONF_MUSIC_APP_TOKEN] == ""
    assert session.finished_with[CONF_MUSIC_USER_TOKEN] == "user-token"


@pytest.mark.asyncio
async def test_setup_signs_in_with_own_token() -> None:
    """An accepted own developer token is used for the sign-in and stored."""
    session, musickit = await _run_flow(
        [{CONF_CUSTOM_APP_TOKEN: " own "}], {"bundled": True, "own": True}
    )
    assert musickit.await_args is not None
    assert musickit.await_args.args[1] == "own"
    assert session.finished_with is not None
    assert session.finished_with[CONF_MUSIC_APP_TOKEN] == "own"


@pytest.mark.asyncio
async def test_setup_rejects_own_token_that_is_not_accepted() -> None:
    """An own token Apple does not positively accept is flagged and asked for again."""
    session, musickit = await _run_flow(
        [{CONF_CUSTOM_APP_TOKEN: "throttled"}, {}], {"bundled": True, "throttled": None}
    )
    assert len(session.forms) == 2
    assert session.forms[1][2] == {CONF_CUSTOM_APP_TOKEN: "invalid_value"}
    assert musickit.await_count == 1
    assert session.finished_with is not None
    assert session.finished_with[CONF_MUSIC_APP_TOKEN] == ""


@pytest.mark.asyncio
async def test_setup_prefills_stored_own_token_on_reconfigure() -> None:
    """A reconfigure offers the stored own token, so submitting unchanged keeps it."""
    session, _ = await _run_flow(
        [{CONF_CUSTOM_APP_TOKEN: "own"}],
        {"bundled": True, "own": True},
        setup_data={CONF_MUSIC_APP_TOKEN: "own"},
    )
    entry = next(e for e in session.forms[0][1] if e.key == CONF_CUSTOM_APP_TOKEN)
    assert entry.default_value == "own"
    assert session.finished_with is not None
    assert session.finished_with[CONF_MUSIC_APP_TOKEN] == "own"


@pytest.mark.asyncio
async def test_setup_does_not_offer_own_token_when_bundled_is_rejected() -> None:
    """A rejected bundled token is replaced in its own step, not offered again later."""
    session, musickit = await _run_flow(
        [{CONF_MUSIC_APP_TOKEN: "typed"}, {}], {"bundled": False, "typed": True}
    )
    assert session.forms[0][0] == "app_token"
    assert CONF_CUSTOM_APP_TOKEN not in _entry_keys(session.forms[1][1])
    assert musickit.await_args is not None
    assert musickit.await_args.args[1] == "typed"
    assert session.finished_with is not None
    assert session.finished_with[CONF_MUSIC_APP_TOKEN] == "typed"
