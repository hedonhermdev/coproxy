from __future__ import annotations

import os
from typing import Any, AsyncIterator, Iterator, cast

import pytest
import pytest_asyncio
from anthropic import APIStatusError, Anthropic, AsyncAnthropic


API_KEY = os.environ.get("TEST_API_KEY", "compat-test-key")
BASE_URL = os.environ.get("TEST_ANTHROPIC_BASE_URL", "http://127.0.0.1:4011")
TEST_MODEL = os.environ.get("TEST_ANTHROPIC_MODEL", "claude-haiku-4.5")
TEST_IMAGE_MODEL = os.environ.get("TEST_ANTHROPIC_IMAGE_MODEL", "claude-opus-4.7")

# 1x1 transparent PNG — minimal payload to verify image content blocks flow
# through the proxy without changing upstream model behavior.
TEST_IMAGE_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAA"
    "C0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
)


@pytest.fixture(scope="session")
def anthropic_client() -> Iterator[Anthropic]:
    with Anthropic(api_key=API_KEY, base_url=BASE_URL) as c:
        yield c


@pytest_asyncio.fixture
async def async_anthropic_client() -> AsyncIterator[AsyncAnthropic]:
    async with AsyncAnthropic(api_key=API_KEY, base_url=BASE_URL) as c:
        yield c


@pytest.fixture(scope="session")
def anthropic_model() -> str:
    return TEST_MODEL


def _error_code(exc: APIStatusError) -> str | None:
    if not isinstance(exc.body, dict):
        return None
    body = cast(dict[str, Any], exc.body)
    code = body.get("code")
    return code if isinstance(code, str) else None


def _messages_available(client: Anthropic, model: str) -> bool:
    try:
        client.messages.create(
            model=model,
            max_tokens=8,
            messages=[{"role": "user", "content": "ping"}],
        )
    except APIStatusError as exc:
        if exc.status_code == 401:
            return False
        if _error_code(exc) in {"model_not_supported", "unsupported_api_for_model"}:
            return False
        return False
    except Exception:
        return False
    return True


@pytest.fixture(scope="session")
def messages_available(anthropic_client: Anthropic, anthropic_model: str) -> bool:
    return _messages_available(anthropic_client, anthropic_model)


@pytest.fixture(scope="session")
def anthropic_image_model() -> str:
    return TEST_IMAGE_MODEL


@pytest.fixture(scope="session")
def anthropic_image_png_b64() -> str:
    return TEST_IMAGE_PNG_B64


def _image_messages_available(client: Anthropic, model: str) -> bool:
    try:
        client.messages.create(
            model=model,
            max_tokens=8,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "ping"},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": TEST_IMAGE_PNG_B64,
                            },
                        },
                    ],
                }
            ],
        )
    except APIStatusError as exc:
        if exc.status_code == 401:
            return False
        if _error_code(exc) in {"model_not_supported", "unsupported_api_for_model"}:
            return False
        return False
    except Exception:
        return False
    return True


@pytest.fixture(scope="session")
def anthropic_image_model_available(
    anthropic_client: Anthropic, anthropic_image_model: str
) -> bool:
    return _image_messages_available(anthropic_client, anthropic_image_model)
