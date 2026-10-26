
import pytest
from app import summarize_results


class FakeResponse:
    def __init__(self, content):
        self.content = content


class FakeLLM:
    def __init__(self, content):
        self.content = content

    def invoke(self, prompt):
        return FakeResponse(self.content)


def test_summary_returns_exactly_three_bullets(monkeypatch):
    import app

    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setattr(
        app,
        "ChatGroq",
        lambda **kwargs: FakeLLM(
            "- First point\n- Second point\n- Third point"
        ),
    )

    result = summarize_results("Test topic", "Test search results")

    assert len(result.splitlines()) == 3
    assert all(line.startswith("- ") for line in result.splitlines())


def test_summary_rejects_wrong_number_of_bullets(monkeypatch):
    import app

    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setattr(
        app,
        "ChatGroq",
        lambda **kwargs: FakeLLM("- First point\n- Second point"),
    )

    with pytest.raises(ValueError, match="exactly 3 bullet points"):
        summarize_results("Test topic", "Test search results")
