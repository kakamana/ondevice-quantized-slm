from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_summarize_basic():
    doc = (
        "The central bank announced a revised policy. The change takes effect next quarter "
        "and applies to all licensed entities. Officials cited the need for greater transparency. "
        "The story remains developing."
    )
    r = client.post("/summarize", json={"doc": doc, "budget_ms": 100, "max_size_mb": 100})
    assert r.status_code == 200
    body = r.json()
    assert "summary" in body and len(body["summary"]) > 0
    assert body["budget_chosen"] in {"tight", "small", "medium", "large"}


def test_data_generator_deterministic():
    from slm_quant.data import make_docs

    a = make_docs(seed=42)
    b = make_docs(seed=42)
    assert (a["doc_text"].values == b["doc_text"].values).all()


def test_rouge_l_self():
    from slm_quant.features import rouge_l

    assert rouge_l("hello world", "hello world") == 1.0
    assert 0.0 < rouge_l("hello world", "hello there") < 1.0


def test_textrank_truncates_to_budget():
    from slm_quant.features import BUDGETS
    from slm_quant.models import textrank_summarize

    doc = ". ".join([f"sentence number {i} with some filler" for i in range(20)]) + "."
    s = textrank_summarize(doc, budget="tight")
    assert len(s.split()) <= BUDGETS["tight"].max_tokens
