from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class _ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ChatMessageRequest(_ChatRequest):
    """One message included in a chat completion request."""

    role: Literal["system", "user", "assistant"]
    content: str


class StreamOptions(_ChatRequest):
    """Options for additional information in a streaming response."""

    include_usage: bool = False


class ChatTemplateKwargs(_ChatRequest):
    """Template options for multi-turn chat requests."""

    preserve_thinking: bool = False


class ChatCompletionRequest(_ChatRequest):
    """Parameters for a Chat Completions request."""

    """Parameters for a Chat Completions API request."""

    model: str = "k-exaone_v2.5"
    messages: list[ChatMessageRequest] = Field(min_length=1)
    max_tokens: int | None = Field(default=None, gt=0)
    stream: bool = False
    stream_options: StreamOptions | None = None
    temperature: float | None = Field(default=None, ge=0)
    top_p: float | None = Field(default=None, ge=0, le=1)
    top_k: int | None = None
    presence_penalty: float | None = Field(default=None, ge=-2, le=2)
    reasoning_effort: Literal["low", "high", "max"] | None = None
    chat_template_kwargs: ChatTemplateKwargs | None = None

    @field_validator("top_k")
    @classmethod
    def validate_top_k(cls, value: int | None) -> int | None:
        """Allow only ``-1`` or a positive value for ``top_k``."""
        if value is not None and value != -1 and value < 1:
            msg = "top_k must be -1 or a positive integer"
            raise ValueError(msg)
        return value

    @model_validator(mode="after")
    def validate_stream_options(self) -> Self:
        """Require streaming when stream options are provided."""
        if self.stream_options is not None and not self.stream:
            msg = "stream_options can only be used when stream=True"
            raise ValueError(msg)
        return self


class _ChatResponse(BaseModel):
    model_config = ConfigDict(extra="allow")


class ChatMessageResponse(_ChatResponse):
    """Assistant message returned by a chat completion."""

    role: Literal["assistant", "system", "user"]
    content: str
    reasoning_content: str | None = None


class Usage(_ChatResponse):
    """Token counts reported for a chat completion."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    reasoning_tokens: int | None = None


class CompletionChoice(_ChatResponse):
    """One candidate answer in a chat completion response."""

    index: int
    message: ChatMessageResponse
    finish_reason: str | None = None
    logprobs: Any | None = None


class ChatCompletionResponse(_ChatResponse):
    """Non-streaming Chat Completions API response."""

    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[CompletionChoice]
    usage: Usage | None = None


class ChunkDelta(_ChatResponse):
    """Fields emitted in one streaming response update."""

    role: str | None = None
    content: str | None = None
    reasoning_content: str | None = None


class ChunkChoice(_ChatResponse):
    """One candidate update in a streaming response chunk."""

    index: int
    delta: ChunkDelta = Field(default_factory=ChunkDelta)
    finish_reason: str | None = None


class ChatCompletionChunkResponse(_ChatResponse):
    """One server-sent event from a streaming chat completion."""

    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChunkChoice] = Field(default_factory=list)
    usage: Usage | None = None
