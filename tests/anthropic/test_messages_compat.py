from __future__ import annotations

from typing import Any, cast

import pytest
from anthropic import APIStatusError, Anthropic, AsyncAnthropic
from anthropic.types import (
    Message,
    RawContentBlockDeltaEvent,
    RawContentBlockStartEvent,
    RawContentBlockStopEvent,
    RawMessageDeltaEvent,
    RawMessageStartEvent,
    RawMessageStopEvent,
)


# The set of event types the raw Messages SSE stream is allowed to emit.
# Mirrors the Anthropic Messages spec; if the proxy emits anything else,
# something is wrong.
_RAW_STREAM_EVENT_TYPES = {
    "message_start",
    "message_delta",
    "message_stop",
    "content_block_start",
    "content_block_delta",
    "content_block_stop",
    "ping",
}

_RAW_STREAM_EVENT_CLASSES = (
    RawMessageStartEvent,
    RawMessageDeltaEvent,
    RawMessageStopEvent,
    RawContentBlockStartEvent,
    RawContentBlockDeltaEvent,
    RawContentBlockStopEvent,
)


def _basic_messages() -> list[dict[str, str]]:
    return [{"role": "user", "content": "Reply with exactly: pong"}]


def _assert_message_shape(message: Message) -> None:
    assert message.type == "message"
    assert message.role == "assistant"
    assert isinstance(message.id, str) and message.id
    assert isinstance(message.model, str)
    assert isinstance(message.content, list)
    # Anthropic's required usage object should always be present.
    assert message.usage is not None
    assert isinstance(message.usage.input_tokens, int)
    assert isinstance(message.usage.output_tokens, int)


def test_messages_create_basic(
    anthropic_client: Anthropic, messages_available: bool, anthropic_model: str
) -> None:
    if not messages_available:
        pytest.skip("Skipping messages success tests: GHCP auth is unavailable")

    message = anthropic_client.messages.create(
        model=anthropic_model,
        max_tokens=64,
        messages=_basic_messages(),
    )
    _assert_message_shape(message)


def test_messages_with_system_string(
    anthropic_client: Anthropic, messages_available: bool, anthropic_model: str
) -> None:
    if not messages_available:
        pytest.skip("Skipping messages success tests: GHCP auth is unavailable")

    message = anthropic_client.messages.create(
        model=anthropic_model,
        max_tokens=64,
        system="You are concise.",
        messages=_basic_messages(),
    )
    _assert_message_shape(message)


def test_messages_with_system_blocks(
    anthropic_client: Anthropic, messages_available: bool, anthropic_model: str
) -> None:
    if not messages_available:
        pytest.skip("Skipping messages success tests: GHCP auth is unavailable")

    message = anthropic_client.messages.create(
        model=anthropic_model,
        max_tokens=64,
        system=[{"type": "text", "text": "You are concise."}],
        messages=_basic_messages(),
    )
    _assert_message_shape(message)


def test_messages_raw_response(
    anthropic_client: Anthropic, messages_available: bool, anthropic_model: str
) -> None:
    if not messages_available:
        pytest.skip("Skipping messages success tests: GHCP auth is unavailable")

    response = anthropic_client.messages.with_raw_response.create(
        model=anthropic_model,
        max_tokens=64,
        messages=_basic_messages(),
    )
    assert response.is_closed is True
    assert response.http_request.headers.get("X-Stainless-Lang") == "python"
    message = response.parse()
    _assert_message_shape(message)


def test_messages_streaming_response_wrapper(
    anthropic_client: Anthropic, messages_available: bool, anthropic_model: str
) -> None:
    if not messages_available:
        pytest.skip("Skipping messages success tests: GHCP auth is unavailable")

    with anthropic_client.messages.with_streaming_response.create(
        model=anthropic_model,
        max_tokens=64,
        messages=_basic_messages(),
    ) as response:
        assert not response.is_closed
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        # Drain the SSE stream so the underlying connection closes cleanly.
        for _ in response.iter_lines():
            pass

    assert cast(Any, response.is_closed) is True


def test_messages_stream_true(
    anthropic_client: Anthropic, messages_available: bool, anthropic_model: str
) -> None:
    if not messages_available:
        pytest.skip("Skipping messages success tests: GHCP auth is unavailable")

    saw_message_start = False
    saw_message_stop = False

    stream = anthropic_client.messages.create(
        model=anthropic_model,
        max_tokens=64,
        messages=_basic_messages(),
        stream=True,
    )

    for event in stream:
        # Each raw event must be one of the protocol types and must
        # deserialize into the corresponding typed model.
        assert event.type in _RAW_STREAM_EVENT_TYPES, (
            f"Unexpected raw stream event type: {event.type}"
        )
        if event.type != "ping":
            assert isinstance(event, _RAW_STREAM_EVENT_CLASSES)
        if event.type == "message_start":
            saw_message_start = True
        elif event.type == "message_stop":
            saw_message_stop = True

    assert saw_message_start, "Expected a message_start event from the stream"
    assert saw_message_stop, "Expected a message_stop event from the stream"


def test_messages_missing_messages_is_bad_request(
    anthropic_client: Anthropic, anthropic_model: str
) -> None:
    with pytest.raises(APIStatusError) as exc_info:
        anthropic_client.messages.create(
            model=anthropic_model,
            max_tokens=64,
            messages=[],
        )

    err = exc_info.value
    assert err.status_code == 400
    assert isinstance(err.body, dict)
    body = cast(dict[str, Any], err.body)
    assert isinstance(body.get("type"), str)


def test_messages_invalid_api_key_unauthorized(anthropic_model: str) -> None:
    import os

    base_url = os.environ.get("TEST_ANTHROPIC_BASE_URL", "http://127.0.0.1:4011")
    bad_client = Anthropic(api_key="this-key-is-wrong", base_url=base_url)
    try:
        with pytest.raises(APIStatusError) as exc_info:
            bad_client.messages.create(
                model=anthropic_model,
                max_tokens=8,
                messages=_basic_messages(),
            )
        assert exc_info.value.status_code == 401
    finally:
        bad_client.close()


@pytest.mark.asyncio
async def test_async_messages_create_basic(
    async_anthropic_client: AsyncAnthropic,
    messages_available: bool,
    anthropic_model: str,
) -> None:
    if not messages_available:
        pytest.skip("Skipping messages success tests: GHCP auth is unavailable")

    message = await async_anthropic_client.messages.create(
        model=anthropic_model,
        max_tokens=64,
        messages=_basic_messages(),
    )
    _assert_message_shape(message)


@pytest.mark.asyncio
async def test_async_messages_stream_true(
    async_anthropic_client: AsyncAnthropic,
    messages_available: bool,
    anthropic_model: str,
) -> None:
    if not messages_available:
        pytest.skip("Skipping messages success tests: GHCP auth is unavailable")

    saw_message_start = False
    saw_message_stop = False

    stream = await async_anthropic_client.messages.create(
        model=anthropic_model,
        max_tokens=64,
        messages=_basic_messages(),
        stream=True,
    )

    async for event in stream:
        assert event.type in _RAW_STREAM_EVENT_TYPES, (
            f"Unexpected raw stream event type: {event.type}"
        )
        if event.type != "ping":
            assert isinstance(event, _RAW_STREAM_EVENT_CLASSES)
        if event.type == "message_start":
            saw_message_start = True
        elif event.type == "message_stop":
            saw_message_stop = True

    assert saw_message_start
    assert saw_message_stop


def test_messages_image_input(
    anthropic_client: Anthropic,
    anthropic_image_model_available: bool,
    anthropic_image_model: str,
    anthropic_image_png_b64: str,
) -> None:
    if not anthropic_image_model_available:
        pytest.skip(
            f"Skipping image input test: model {anthropic_image_model!r} is "
            "unavailable (GHCP auth, vision capability, or upstream support)"
        )

    message = anthropic_client.messages.create(
        model=anthropic_image_model,
        max_tokens=64,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image briefly."},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": anthropic_image_png_b64,
                        },
                    },
                ],
            }
        ],
    )
    _assert_message_shape(message)


def test_messages_with_tool_use(
    anthropic_client: Anthropic, messages_available: bool, anthropic_model: str
) -> None:
    if not messages_available:
        pytest.skip("Skipping messages success tests: GHCP auth is unavailable")

    tools: list[dict[str, Any]] = [
        {
            "name": "get_weather",
            "description": "Fetch the current weather for a given location.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                },
                "required": ["location"],
            },
        }
    ]

    message = anthropic_client.messages.create(
        model=anthropic_model,
        max_tokens=128,
        tools=tools,
        messages=[
            {
                "role": "user",
                "content": "What is the weather in Paris? Use the get_weather tool.",
            }
        ],
    )
    _assert_message_shape(message)
    # The model is allowed to refuse, so we only assert the response is structurally valid.
