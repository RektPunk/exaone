import asyncio

from exaone.chat.sse import SSEDecoder


def test_sse_decoder_handles_multiline_data_comments_and_fields():
    events = list(
        SSEDecoder().decode(
            [
                ": ping",
                "event: message",
                "id: chunk-1",
                'data: {"text":',
                'data: "two lines"}',
                "",
            ],
        ),
    )

    assert len(events) == 1
    assert events[0].data == '{"text":\n"two lines"}'
    assert events[0].event == "message"
    assert events[0].id == "chunk-1"


def test_sse_decoder_removes_one_optional_space_and_ignores_unknown_fields():
    events = list(SSEDecoder().decode(["unknown: ignored", "data:  value", ""]))

    assert events[0].data == " value"


def test_sse_decoder_async_matches_sync_decoding():
    lines = ["event: message", "id: chunk-1", "data: first", "data: second", ""]

    async def async_lines():
        for line in lines:
            yield line

    async def collect():
        return [event async for event in SSEDecoder().decode_async(async_lines())]

    events = asyncio.run(collect())

    assert len(events) == 1
    assert events[0].data == "first\nsecond"
    assert events[0].event == "message"
    assert events[0].id == "chunk-1"
