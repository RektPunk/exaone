import pytest
from pydantic import ValidationError

from exaone.chat.models import (
    ChatCompletionRequest,
    ChatMessageRequest,
    StreamOptions,
)


def test_chat_request_defaults():
    request = ChatCompletionRequest(
        messages=[ChatMessageRequest(role="user", content="Hi")],
    )

    assert request.model == "k-exaone_v2.5"
    assert request.stream is False


def test_chat_request_serializes_stream_options():
    request = ChatCompletionRequest(
        messages=[ChatMessageRequest(role="user", content="Hi")],
        stream=True,
        stream_options=StreamOptions(include_usage=True),
    )

    assert request.model_dump(exclude_none=True)["stream_options"] == {
        "include_usage": True,
    }


def test_chat_request_rejects_empty_messages_and_invalid_parameters():
    with pytest.raises(ValidationError):
        ChatCompletionRequest(messages=[])
    with pytest.raises(ValidationError):
        ChatCompletionRequest(
            messages=[ChatMessageRequest(role="user", content="Hi")],
            top_p=1.1,
        )


@pytest.mark.parametrize("top_k", [0, -2])
def test_chat_request_rejects_invalid_top_k(top_k):
    with pytest.raises(ValidationError, match="top_k"):
        ChatCompletionRequest(
            messages=[ChatMessageRequest(role="user", content="Hi")],
            top_k=top_k,
        )


@pytest.mark.parametrize("top_k", [-1, 1, 10])
def test_chat_request_accepts_valid_top_k(top_k):
    request = ChatCompletionRequest(
        messages=[ChatMessageRequest(role="user", content="Hi")],
        top_k=top_k,
    )

    assert request.top_k == top_k


def test_chat_request_rejects_stream_options_without_streaming():
    with pytest.raises(ValidationError, match="stream_options"):
        ChatCompletionRequest(
            messages=[ChatMessageRequest(role="user", content="Hi")],
            stream_options=StreamOptions(include_usage=True),
        )


def test_chat_completion_request_validates_messages_as_request_models():
    request = ChatCompletionRequest.model_validate(
        {"messages": [{"role": "user", "content": "Hi"}]},
    )

    assert isinstance(request.messages[0], ChatMessageRequest)
