import pytest


@pytest.fixture
def completion_payload():
    return {
        "id": "cmpl-test",
        "object": "chat.completion",
        "created": 1,
        "model": "k-exaone_v2.5",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": "Hello",
                    "reasoning_content": "Thinking",
                },
            }
        ],
        "usage": {
            "prompt_tokens": 2,
            "completion_tokens": 4,
            "total_tokens": 6,
            "reasoning_tokens": 3,
        },
    }
