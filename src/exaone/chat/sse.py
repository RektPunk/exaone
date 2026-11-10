from collections.abc import AsyncIterable, AsyncIterator, Iterable, Iterator
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ServerSentEvent:
    """One dispatched SSE event."""

    data: str
    event: str = "message"
    id: str | None = None


class SSEDecoder:
    """Turn SSE lines into events, preserving the protocol's field semantics."""

    def __init__(self) -> None:
        self._data: list[str] = []
        self._event = ""
        self._last_event_id: str | None = None
        self._first_line = True

    def feed(self, line: str) -> ServerSentEvent | None:
        """Consume one line (without its line ending), returning a dispatched event."""
        if self._first_line:
            line = line.removeprefix("\ufeff")
            self._first_line = False

        if line == "":
            return self._dispatch()

        if line.startswith(":"):
            return None

        field, separator, value = line.partition(":")
        if separator and value.startswith(" "):
            value = value[1:]

        if field == "data":
            self._data.append(value)
        elif field == "event":
            self._event = value
        elif field == "id" and "\x00" not in value:
            self._last_event_id = value

        return None

    def _dispatch(self) -> ServerSentEvent | None:
        if not self._data:
            self._event = ""
            return None

        event = ServerSentEvent(
            data="\n".join(self._data),
            event=self._event or "message",
            id=self._last_event_id,
        )
        self._data.clear()
        self._event = ""

        return event

    def decode(self, lines: Iterable[str]) -> Iterator[ServerSentEvent]:
        """Decode an iterable of lines. Events dispatch at blank lines."""
        for line in lines:
            event = self.feed(line)
            if event is not None:
                yield event

    async def decode_async(
        self,
        lines: AsyncIterable[str],
    ) -> AsyncIterator[ServerSentEvent]:
        """Decode an asynchronous iterable of lines without buffering the stream."""
        async for line in lines:
            event = self.feed(line)
            if event is not None:
                yield event
