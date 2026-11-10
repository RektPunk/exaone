# exaone

An unofficial Python client for the [EXAONE API Platform](https://docs.exaone.ai/en/docs/overview), with sync and async access to its endpoints.

> [!CAUTION]
> This project is not affiliated with or endorsed by LG AI Research. EXAONE API use is subject to the [EXAONE API Platform Terms of Service](https://docs.exaone.ai/ko/terms/terms-of-service). The terms limit API and model use to non-commercial purposes. They prohibit commercial use of the model or its outputs, and integration into a product or service that lets third parties call the model or submit inputs. Check the terms before using this client.

## Install

```bash
uv add git+https://github.com/RektPunk/exaone.git
# or
pip install git+https://github.com/RektPunk/exaone.git
```

## Set your API key

The client reads `EXAONE_API_KEY` from the process environment. It does not load `.env` files itself.

```bash
# macOS / Linux
export EXAONE_API_KEY="your-api-key"
```

```powershell
# Windows PowerShell
$env:EXAONE_API_KEY = "your-api-key"
```

## Chat

```python
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
```

## Tabular prediction

Pass a training CSV with the target column and a test CSV containing only the feature columns for prediction.

```python
from exaone import ClassificationRequest, Exaone, RegressionRequest

classification_request = ClassificationRequest(
    target_columns=["churn"],
    output_type="probability",
)
regression_request = RegressionRequest(
    target_columns=["price"],
    output_type="point",
)

with Exaone() as client:
    # Classification
    classification = client.tabular.classify(
        "classification_train.csv",
        "classification_test.csv",
        classification_request,
    )
    print("Classes:", classification.data.classes)
    print("Probabilities:", classification.data.probabilities)

    # Regression
    regression = client.tabular.regress(
        "regression_train.csv",
        "regression_test.csv",
        regression_request,
    )
    print("Predictions:", regression.data.predictions)
```
