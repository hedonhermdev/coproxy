from __future__ import annotations

from typing import Any, cast

import pytest
from openai import AsyncOpenAI, NotFoundError, OpenAI
from openai.pagination import AsyncPage, SyncPage
from openai.types import Model


def _first_model_id(client: OpenAI) -> str:
    models = client.models.list()
    assert models.data, "Expected at least one model from /v1/models"
    return models.data[0].id


def _find_model_id(client: OpenAI, model_id: str) -> str:
    models = client.models.list()
    ids = [m.id for m in models.data]
    assert model_id in ids, f"Expected model '{model_id}' in /v1/models response: {ids}"
    return model_id


def _assert_model(model: Model) -> None:
    assert model.object == "model"
    assert isinstance(model.id, str)
    assert isinstance(model.created, int)
    assert isinstance(model.owned_by, str)


def test_models_list(client: OpenAI) -> None:
    models = client.models.list()
    assert isinstance(models, SyncPage)
    assert models.object == "list"
    assert models.data
    _assert_model(models.data[0])


def test_models_raw_response_list(client: OpenAI) -> None:
    response = client.models.with_raw_response.list()
    assert response.is_closed is True
    assert response.http_request.headers.get("X-Stainless-Lang") == "python"

    models = response.parse()
    assert isinstance(models, SyncPage)
    assert models.data


def test_models_streaming_response_list(client: OpenAI) -> None:
    with client.models.with_streaming_response.list() as response:
        assert not response.is_closed
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        models = response.parse()
        assert isinstance(models, SyncPage)
        assert models.data

    assert cast(Any, response.is_closed) is True


def test_models_retrieve(client: OpenAI, compat_model: str) -> None:
    model_id = _find_model_id(client, compat_model)
    model = client.models.retrieve(model_id)
    _assert_model(model)
    assert model.id == model_id


def test_models_raw_response_retrieve(client: OpenAI, compat_model: str) -> None:
    model_id = _find_model_id(client, compat_model)
    response = client.models.with_raw_response.retrieve(model_id)
    assert response.is_closed is True
    assert response.http_request.headers.get("X-Stainless-Lang") == "python"
    model = response.parse()
    _assert_model(model)


def test_models_streaming_response_retrieve(client: OpenAI, compat_model: str) -> None:
    model_id = _find_model_id(client, compat_model)
    with client.models.with_streaming_response.retrieve(model_id) as response:
        assert not response.is_closed
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        model = response.parse()
        _assert_model(model)

    assert cast(Any, response.is_closed) is True


def test_models_retrieve_path_validation(client: OpenAI) -> None:
    with pytest.raises(
        ValueError, match=r"Expected a non-empty value for `model` but received ''"
    ):
        client.models.with_raw_response.retrieve("")


def test_models_retrieve_not_found(client: OpenAI) -> None:
    with pytest.raises(NotFoundError):
        client.models.retrieve("this-model-should-not-exist")


@pytest.mark.asyncio
async def test_async_models_list(async_client: AsyncOpenAI) -> None:
    models = await async_client.models.list()
    assert isinstance(models, AsyncPage)
    assert models.object == "list"
    assert models.data
    _assert_model(models.data[0])


@pytest.mark.asyncio
async def test_async_models_retrieve(
    async_client: AsyncOpenAI, compat_model: str
) -> None:
    models = await async_client.models.list()
    ids = [m.id for m in models.data]
    assert compat_model in ids, (
        f"Expected model '{compat_model}' in /v1/models response: {ids}"
    )
    model_id = compat_model
    model = await async_client.models.retrieve(model_id)
    _assert_model(model)
    assert model.id == model_id


@pytest.mark.asyncio
async def test_async_models_streaming_response_list(async_client: AsyncOpenAI) -> None:
    async with async_client.models.with_streaming_response.list() as response:
        assert not response.is_closed
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        models = await response.parse()
        assert isinstance(models, AsyncPage)
        assert models.data

    assert cast(Any, response.is_closed) is True
