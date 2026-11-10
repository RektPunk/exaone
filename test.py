import asyncio

from exaone import AsyncExaone, ChatCompletionRequest, ChatMessageRequest, Exaone

request = ChatCompletionRequest(
    messages=[ChatMessageRequest(role="user", content="Hi!")],
    reasoning_effort="low",
)


def sync_examples():
    with Exaone() as client:
        # Synchronous chat
        response = client.chat.create(request)
        print(response.choices[0].message.content)

        # Synchronous streaming
        for chunk in client.chat.stream(request):
            if chunk.choices:
                content = chunk.choices[0].delta.content
                if content:
                    print(content, end="", flush=True)
        print()


async def async_examples():
    async with AsyncExaone() as client:
        # Asynchronous chat
        response = await client.chat.create(request)
        print(response.choices[0].message.content)

        # Asynchronous streaming
        async for chunk in client.chat.stream(request):
            if chunk.choices:
                content = chunk.choices[0].delta.content
                if content:
                    print(content, end="", flush=True)
        print()


sync_examples()
asyncio.run(async_examples())
