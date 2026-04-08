from __future__ import annotations

import os
from typing import Any, AsyncIterator, Iterator, cast

import pytest
import pytest_asyncio
from openai import AsyncOpenAI, OpenAI
from openai import APIStatusError


API_KEY = os.environ.get("TEST_API_KEY", "compat-test-key")
BASE_URL = os.environ.get("TEST_API_BASE_URL", "http://127.0.0.1:4010/v1")
TEST_MODEL = os.environ.get("TEST_MODEL", "gpt-4o")
TEST_RESPONSES_MODEL = os.environ.get("TEST_RESPONSES_MODEL", "gpt-5.4")


@pytest.fixture(scope="session")
def client() -> Iterator[OpenAI]:
    with OpenAI(api_key=API_KEY, base_url=BASE_URL) as c:
        yield c


@pytest_asyncio.fixture
async def async_client() -> AsyncIterator[AsyncOpenAI]:
    async with AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL) as c:
        yield c


def _error_code(exc: APIStatusError) -> str | None:
    if not isinstance(exc.body, dict):
        return None
    body = cast(dict[str, Any], exc.body)
    code = body.get("code")
    return code if isinstance(code, str) else None


def _chat_available(client: OpenAI) -> bool:
    try:
        client.chat.completions.create(
            model=TEST_MODEL,
            messages=[{"role": "user", "content": "ping"}],
        )
    except APIStatusError as exc:
        if exc.status_code == 401:
            return False
        if _error_code(exc) in {"model_not_supported", "unsupported_api_for_model"}:
            return False
        return False
    return True


@pytest.fixture(scope="session")
def chat_available(client: OpenAI) -> bool:
    return _chat_available(client)


@pytest.fixture(scope="session")
def compat_model() -> str:
    return TEST_MODEL


@pytest.fixture(scope="session")
def responses_model() -> str:
    return TEST_RESPONSES_MODEL


@pytest.fixture(scope="session")
def responses_available(client: OpenAI, responses_model: str) -> bool:
    try:
        client.responses.create(model=responses_model, input="ping")
    except APIStatusError as exc:
        if exc.status_code == 401:
            return False
        if _error_code(exc) in {"model_not_supported", "unsupported_api_for_model"}:
            return False
        return False
    return True
