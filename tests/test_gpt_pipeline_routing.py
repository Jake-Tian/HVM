import unittest
import sys
from types import SimpleNamespace
from unittest.mock import patch

from utils import llm_gpt

# This test validates API routing only; frame encoding is covered by the
# pipeline environment and should not require NumPy/OpenCV in a CPU-only shell.
sys.modules["cv2"] = SimpleNamespace()
sys.modules["numpy"] = SimpleNamespace(ndarray=type("ndarray", (), {}))
from utils import mllm_gpt


def fake_response(content="ok"):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(
            prompt_tokens=10,
            completion_tokens=2,
            total_tokens=12,
        ),
    )


class FakeCompletions:
    def __init__(self):
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return fake_response()


class GptPipelineRoutingTest(unittest.TestCase):
    def test_text_uses_model_defaults(self):
        completions = FakeCompletions()
        client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        with (
            patch.object(llm_gpt, "_client", return_value=client),
            patch.object(llm_gpt, "MODEL", "gpt-5-mini"),
        ):
            _, usage = llm_gpt.generate_text_response("test")
        self.assertEqual(completions.kwargs["model"], "gpt-5-mini")
        self.assertNotIn("reasoning_effort", completions.kwargs)
        self.assertNotIn("temperature", completions.kwargs)
        self.assertEqual(usage, {"total": 12})

    def test_multimodal_uses_model_defaults(self):
        completions = FakeCompletions()
        client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        messages = [{"role": "user", "content": [{"type": "text", "text": "test"}]}]
        with (
            patch.object(mllm_gpt, "_client", return_value=client),
            patch.object(mllm_gpt, "MODEL", "gpt-5-mini"),
        ):
            _, usage = mllm_gpt.get_response(messages)
        self.assertEqual(completions.kwargs["model"], "gpt-5-mini")
        self.assertNotIn("reasoning_effort", completions.kwargs)
        self.assertNotIn("temperature", completions.kwargs)
        self.assertEqual(usage["total"], 12)


if __name__ == "__main__":
    unittest.main()
