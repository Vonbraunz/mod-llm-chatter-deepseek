#!/usr/bin/env python3
"""Regression checks for screenshot-agent model compatibility."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock


TOOLS_DIR = Path(__file__).resolve().parents[1]
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from screenshot_agent import _call_openai  # noqa: E402


class ScreenshotCompatibilityTests(unittest.TestCase):
    def _client(self):
        response = SimpleNamespace(choices=[SimpleNamespace(
            finish_reason="stop",
            message=SimpleNamespace(content='{"biome":"forest"}'),
        )])
        operation = MagicMock(return_value=response)
        client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=operation)
            )
        )
        return client, operation

    def test_openai_reasoning_model_uses_compatible_token_field(self):
        client, operation = self._client()
        _call_openai(
            "image", client, "gpt-5.6-luna",
            reasoning_effort="none",
        )
        request = operation.call_args.kwargs
        self.assertEqual(request["max_completion_tokens"], 300)
        self.assertEqual(request["reasoning_effort"], "none")
        self.assertNotIn("max_tokens", request)

    def test_reasoning_budget_is_multiplied(self):
        client, operation = self._client()
        _call_openai(
            "image", client, "gpt-5.6-luna",
            reasoning_effort="medium",
            max_tokens_multiplier=4,
        )
        request = operation.call_args.kwargs
        self.assertEqual(request["max_completion_tokens"], 1200)

    def test_deepseek_vision_uses_openai_compatible_shape(self):
        client, operation = self._client()
        _call_openai(
            "image", client, "deepseek-flash",
            provider="deepseek",
        )
        request = operation.call_args.kwargs
        self.assertEqual(request["model"], "deepseek-flash")
        self.assertEqual(request["max_tokens"], 300)
        self.assertNotIn("reasoning_effort", request)
        image_block = request["messages"][1]["content"][0]
        self.assertEqual(image_block["type"], "image_url")

    def test_deepseek_vision_disables_thinking(self):
        client, operation = self._client()
        _call_openai(
            "image", client, "deepseek-flash",
            provider="deepseek",
        )
        request = operation.call_args.kwargs
        self.assertEqual(
            request["extra_body"],
            {"thinking": {"type": "disabled"}},
        )


if __name__ == "__main__":
    unittest.main()
