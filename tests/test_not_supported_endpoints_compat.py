from __future__ import annotations

from typing import Any, cast

import pytest
from openai import APIStatusError, AsyncOpenAI, BadRequestError, OpenAI
from openai.types.responses import Response


def _assert_not_supported_error(exc: BadRequestError) -> None:
    assert exc.status_code == 400
    assert isinstance(exc.body, dict)
    body = cast(dict[str, Any], exc.body)
    assert body.get("type") == "invalid_request_error"
    assert body.get("code") == "not_supported"
    assert isinstance(body.get("message"), str)


def _assert_response_shape(response: Response, expected_model_prefix: str) -> None:
    assert response.object == "response"
    assert isinstance(response.id, str)
    assert response.id
    assert isinstance(response.created_at, (float, int))
    assert isinstance(response.model, str)
    assert response.model.startswith(expected_model_prefix)


def test_embeddings_not_supported(client: OpenAI) -> None:
    with pytest.raises(BadRequestError) as exc_info:
        client.embeddings.create(
            model="text-embedding-3-small",
            input="The quick brown fox jumped over the lazy dog",
        )

    _assert_not_supported_error(exc_info.value)


def test_responses_create_supported_or_model_gated(
    client: OpenAI, responses_available: bool, responses_model: str
) -> None:
    if not responses_available:
        pytest.skip(
            "Skipping responses success tests: GHCP responses model is unavailable"
        )

    response = client.responses.create(
        model=responses_model,
        input="Reply with pong",
    )
    _assert_response_shape(response, responses_model)


def test_responses_raw_response_create_supported_or_model_gated(
    client: OpenAI, responses_available: bool, responses_model: str
) -> None:
    if not responses_available:
        pytest.skip(
            "Skipping responses success tests: GHCP responses model is unavailable"
        )

    http_response = client.responses.with_raw_response.create(
        model=responses_model,
        input="Reply with pong",
    )

    assert http_response.is_closed is True
    assert http_response.http_request.headers.get("X-Stainless-Lang") == "python"
    parsed = http_response.parse()
    _assert_response_shape(parsed, responses_model)


def test_responses_streaming_response_create_supported_or_model_gated(
    client: OpenAI, responses_available: bool, responses_model: str
) -> None:
    if not responses_available:
        pytest.skip(
            "Skipping responses success tests: GHCP responses model is unavailable"
        )

    with client.responses.with_streaming_response.create(
        model=responses_model,
        input="Reply with pong",
    ) as http_response:
        assert not http_response.is_closed
        assert http_response.http_request.headers.get("X-Stainless-Lang") == "python"
        parsed = http_response.parse()
        _assert_response_shape(parsed, responses_model)

    assert cast(Any, http_response.is_closed) is True


def test_responses_retrieve_not_found_or_not_supported(client: OpenAI) -> None:
    with pytest.raises(APIStatusError) as exc_info:
        client.responses.retrieve("resp_compat_test")

    assert exc_info.value.status_code in {400, 404}


@pytest.mark.asyncio
async def test_async_embeddings_not_supported(async_client: AsyncOpenAI) -> None:
    with pytest.raises(BadRequestError) as exc_info:
        await async_client.embeddings.create(
            model="text-embedding-3-small",
            input="The quick brown fox jumped over the lazy dog",
        )

    _assert_not_supported_error(exc_info.value)


@pytest.mark.asyncio
async def test_async_responses_create_supported_or_model_gated(
    async_client: AsyncOpenAI, responses_available: bool, responses_model: str
) -> None:
    if not responses_available:
        pytest.skip(
            "Skipping responses success tests: GHCP responses model is unavailable"
        )

    response = await async_client.responses.create(
        model=responses_model,
        input="Reply with pong",
    )
    _assert_response_shape(response, responses_model)


@pytest.mark.asyncio
async def test_async_responses_stream_true_supported_or_model_gated(
    async_client: AsyncOpenAI, responses_available: bool, responses_model: str
) -> None:
    if not responses_available:
        pytest.skip(
            "Skipping responses success tests: GHCP responses model is unavailable"
        )

    stream = await async_client.responses.create(
        model=responses_model,
        input="Reply with pong",
        stream=True,
    )
    await stream.response.aclose()


@pytest.mark.asyncio
async def test_async_responses_retrieve_not_supported(
    async_client: AsyncOpenAI,
) -> None:
    with pytest.raises(APIStatusError) as exc_info:
        await async_client.responses.retrieve("resp_compat_test")

    assert exc_info.value.status_code in {400, 404}
