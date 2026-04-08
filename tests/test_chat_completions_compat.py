from __future__ import annotations

from typing import Any, cast

import pytest
from openai import APIStatusError, AsyncOpenAI, OpenAI
from openai.types.chat import ChatCompletion, ChatCompletionChunk


def _basic_messages() -> list[dict[str, str]]:
    return [{"role": "user", "content": "Reply with exactly: pong"}]


def _assert_completion_shape(completion: ChatCompletion) -> None:
    assert completion.object == "chat.completion"
    assert isinstance(completion.id, str)
    assert isinstance(completion.created, int)
    assert isinstance(completion.model, str)
    assert completion.choices
    assert completion.choices[0].message.role == "assistant"


def test_chat_completion_create_basic(
    client: OpenAI, chat_available: bool, compat_model: str
) -> None:
    if not chat_available:
        pytest.skip("Skipping chat success tests: GHCP auth is unavailable")

    completion = client.chat.completions.create(
        model=compat_model,
        messages=_basic_messages(),
    )
    _assert_completion_shape(completion)


def test_chat_completion_raw_response(
    client: OpenAI, chat_available: bool, compat_model: str
) -> None:
    if not chat_available:
        pytest.skip("Skipping chat success tests: GHCP auth is unavailable")

    response = client.chat.completions.with_raw_response.create(
        model=compat_model,
        messages=_basic_messages(),
    )
    assert response.is_closed is True
    assert response.http_request.headers.get("X-Stainless-Lang") == "python"
    completion = response.parse()
    _assert_completion_shape(completion)


def test_chat_completion_streaming_response_wrapper(
    client: OpenAI, chat_available: bool, compat_model: str
) -> None:
    if not chat_available:
        pytest.skip("Skipping chat success tests: GHCP auth is unavailable")

    with client.chat.completions.with_streaming_response.create(
        model=compat_model,
        messages=_basic_messages(),
    ) as response:
        assert not response.is_closed
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        completion = response.parse()
        _assert_completion_shape(completion)

    assert cast(Any, response.is_closed) is True


def test_chat_completion_stream_true(
    client: OpenAI, chat_available: bool, compat_model: str
) -> None:
    if not chat_available:
        pytest.skip("Skipping chat success tests: GHCP auth is unavailable")

    stream = client.chat.completions.create(
        model=compat_model,
        messages=_basic_messages(),
        stream=True,
    )

    saw_chunk = False
    for chunk in stream:
        assert isinstance(chunk, ChatCompletionChunk)
        saw_chunk = True
    assert saw_chunk


def test_chat_completion_missing_messages_is_bad_request(
    client: OpenAI, compat_model: str
) -> None:
    with pytest.raises(APIStatusError) as exc_info:
        client.chat.completions.create(model=compat_model, messages=[])

    err = exc_info.value
    assert err.status_code == 400
    assert isinstance(err.body, dict)
    body = cast(dict[str, Any], err.body)
    assert body.get("type") == "invalid_request_error"
    assert isinstance(body.get("message"), str)


@pytest.mark.asyncio
async def test_async_chat_completion_create_basic(
    async_client: AsyncOpenAI, chat_available: bool, compat_model: str
) -> None:
    if not chat_available:
        pytest.skip("Skipping chat success tests: GHCP auth is unavailable")

    completion = await async_client.chat.completions.create(
        model=compat_model,
        messages=_basic_messages(),
    )
    _assert_completion_shape(completion)


@pytest.mark.asyncio
async def test_async_chat_completion_stream_true(
    async_client: AsyncOpenAI, chat_available: bool, compat_model: str
) -> None:
    if not chat_available:
        pytest.skip("Skipping chat success tests: GHCP auth is unavailable")

    stream = await async_client.chat.completions.create(
        model=compat_model,
        messages=_basic_messages(),
        stream=True,
    )

    saw_chunk = False
    async for chunk in stream:
        assert isinstance(chunk, ChatCompletionChunk)
        saw_chunk = True
    assert saw_chunk
