import httpx2

from exaone.chat.models import ChatCompletionChunkResponse, ChatCompletionResponse
from exaone.chat.sse import ServerSentEvent
from exaone.exceptions import ExaoneResponseError


def parse_chat_completion(response: httpx2.Response) -> ChatCompletionResponse:
    """Validate a non-streaming HTTP response as a chat completion."""
    try:
        return ChatCompletionResponse.model_validate_json(response.content)
    except (ValueError, TypeError) as exc:
        msg = "Invalid chat completion response"
        raise ExaoneResponseError(msg) from exc


def parse_chat_completion_event(
    event: ServerSentEvent,
) -> ChatCompletionChunkResponse | None:
    """Convert an SSE event to a chunk; return ``None`` for the API end marker."""
    if event.data == "[DONE]":
        return None

    try:
        return ChatCompletionChunkResponse.model_validate_json(event.data)
    except (TypeError, ValueError) as exc:
        msg = f"Invalid chat completion stream event: {event.data!r}"
        raise ExaoneResponseError(msg) from exc
