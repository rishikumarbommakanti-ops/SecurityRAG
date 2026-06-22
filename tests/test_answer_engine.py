from app.engine.answer_engine import AnswerEngine, AnswerSource


class DummyRetriever:
    def format_context(self, documents):
        return "\n\n---\n\n".join(
            f"Source: {doc.get('metadata', {}).get('source', 'Unknown')}\n\nContent:\n{doc.get('content', '')}"
            for doc in documents
        )


class DummyGeminiClient:
    @staticmethod
    def generate_answer(question: str, context: str) -> str:
        return f"Gemini answer for: {question}"


def test_can_answer_locally_when_high_confidence_and_pattern_matches(monkeypatch):
    retriever = DummyRetriever()
    engine = AnswerEngine(retriever=retriever, confidence_threshold=0.5)

    documents = [
        {"content": "This document references T1059 execution.", "metadata": {"source": "attack.pdf"}, "score": 0.9},
    ]

    assert engine.can_answer_locally(documents) is True


def test_cannot_answer_locally_when_low_confidence(monkeypatch):
    retriever = DummyRetriever()
    engine = AnswerEngine(retriever=retriever, confidence_threshold=0.95)

    documents = [
        {"content": "This document references T1059 execution.", "metadata": {"source": "attack.pdf"}, "score": 0.9},
    ]

    assert engine.can_answer_locally(documents) is False


def test_answer_returns_local_when_condition_met(monkeypatch):
    retriever = DummyRetriever()
    engine = AnswerEngine(retriever=retriever, confidence_threshold=0.5)

    documents = [
        {"content": "This document references T1059 execution.", "metadata": {"source": "attack.pdf"}, "score": 0.9},
    ]

    result = engine.answer("Explain T1059", documents)

    assert result["answer_source"] == AnswerSource.LOCAL.value
    assert "high-confidence security match" in result["answer"]


def test_answer_uses_gemini_when_local_conditions_not_met(monkeypatch):
    retriever = DummyRetriever()
    engine = AnswerEngine(retriever=retriever, confidence_threshold=0.95)

    monkeypatch.setattr("app.engine.answer_engine.GeminiClient.generate_answer", DummyGeminiClient.generate_answer)

    documents = [
        {"content": "A lower-confidence document without known Sigma labels.", "metadata": {"source": "notes.pdf"}, "score": 0.6},
    ]

    result = engine.answer("What is this event?", documents)

    assert result["answer_source"] == AnswerSource.GEMINI.value
    assert result["answer"] == "Gemini answer for: What is this event?"
