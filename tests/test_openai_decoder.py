import math
from unittest.mock import Mock

import httpx
import pandas as pd
import pytest
from openai import BadRequestError
from openai.types.chat import ChatCompletion

from alpaca_eval import constants, utils
from alpaca_eval.annotators import SinglePairwiseAnnotator
from alpaca_eval.decoders.openai import _openai_completion_helper, openai_completions


@pytest.fixture
def chat_client(monkeypatch):
    client = Mock()

    def create(**kwargs):
        return ChatCompletion(
            id="test-completion",
            created=0,
            model=kwargs["model"],
            object="chat.completion",
            choices=[
                {
                    "index": 0,
                    "finish_reason": "length",
                    "message": {"role": "assistant", "content": "m"},
                    "logprobs": {
                        "content": [
                            {
                                "token": "m",
                                "logprob": math.log(0.8),
                                "top_logprobs": [
                                    {"token": "m", "logprob": math.log(0.8)},
                                    {"token": "M", "logprob": math.log(0.2)},
                                ],
                            }
                        ]
                    },
                }
            ],
            usage={"prompt_tokens": 100, "completion_tokens": 1, "total_tokens": 101},
        )

    client.chat.completions.create.side_effect = create
    monkeypatch.setattr(utils, "get_all_clients", lambda *args, **kwargs: [client])
    return client


def test_gpt54_evaluator(chat_client):
    config = utils.load_configs(constants.EVALUATORS_CONFIG_DIR / "weighted_alpaca_eval_gpt5_4")
    evaluator = SinglePairwiseAnnotator(
        **config["weighted_alpaca_eval_gpt5_4"], is_randomize_output_order=False, is_shuffle=False
    )
    annotated = evaluator(
        pd.DataFrame([{"instruction": "What is 1+1?", "output_1": "2", "output_2": "3"}]),
        num_procs=1,
    )

    request = chat_client.chat.completions.create.call_args.kwargs
    assert request["model"] == "gpt-5.4-2026-03-05"
    assert request["reasoning_effort"] == "none"
    assert request["max_completion_tokens"] == 1
    assert "max_tokens" not in request
    assert request["logprobs"] is True
    assert request["top_logprobs"] == 5
    assert request["temperature"] == 1
    assert any("What is 1+1?" in message["content"] for message in request["messages"])
    assert annotated["preference"].tolist() == pytest.approx([1.2])
    chat_client.completions.create.assert_not_called()


@pytest.mark.parametrize("max_tokens", [1, 7, [3, 5]])
def test_legacy_chat_token_limits(chat_client, max_tokens):
    result = openai_completions(
        ["<|im_start|>user\nHi<|im_end|>"] * 2,
        model_name="gpt-4.1-2025-04-14",
        max_tokens=max_tokens,
        num_procs=1,
    )
    requests = [call.kwargs for call in chat_client.chat.completions.create.call_args_list]
    expected = max_tokens if isinstance(max_tokens, list) else [max_tokens] * 2
    assert [request["max_tokens"] for request in requests] == expected
    assert all("max_completion_tokens" not in request for request in requests)
    assert result["completions"] == ["m", "m"]


def test_modern_limit_overrides_legacy_limit(chat_client):
    openai_completions(
        ["<|im_start|>user\nHi<|im_end|>"],
        model_name="gpt-5.4-2026-03-05",
        max_tokens=2048,
        max_completion_tokens=1,
        reasoning_effort="none",
        num_procs=1,
    )
    request = chat_client.chat.completions.create.call_args.kwargs
    assert request["max_completion_tokens"] == 1
    assert "max_tokens" not in request


@pytest.mark.parametrize("modern", [False, True])
@pytest.mark.parametrize("limit", [1, 10])
def test_context_limit_retry(chat_client, modern, limit):
    create_response = chat_client.chat.completions.create.side_effect
    calls = []
    error = BadRequestError(
        "Please reduce the length of the messages or completion.",
        response=httpx.Response(400, request=httpx.Request("POST", "https://example.test")),
        body=None,
    )

    def create(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise error
        return create_response(**kwargs)

    chat_client.chat.completions.create.side_effect = create
    kwargs = {"max_completion_tokens": limit} if modern else {}

    def run():
        return _openai_completion_helper(
            ([[{"role": "user", "content": "Hi"}]], limit),
            is_chat=True,
            model="gpt-5.4-2026-03-05" if modern else "gpt-4.1-2025-04-14",
            n_retries=2,
            **kwargs,
        )

    if limit == 1:
        with pytest.raises(BadRequestError):
            run()
        assert len(calls) == 1
    else:
        assert run()[0]["text"] == "m"
        token_key = "max_completion_tokens" if modern else "max_tokens"
        assert [request[token_key] for request in calls] == [10, 8]
        if modern:
            assert all("max_tokens" not in request for request in calls)
