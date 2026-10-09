"""
ChatResponse schema tests run through real Pydantic validation, without stubs.

The stubbed ChatService never returned diagnosis_en/_pl, which hid that Pydantic v2
silently dropped those fields when they were missing from the schema.
"""
from __future__ import annotations
import pytest
from pydantic import ValidationError

from backend.schemas import ChatResponse, TopDiagnosisItem


def test_chat_response_accepts_diagnosis_fields():
    """diagnosis_en/_pl must not be silently dropped by the schema."""
    payload = {
        "response": "Najbardziej prawdopodobna przyczyna to grypa (78%).",
        "symptoms": ["fever", "cough", "sore_throat"],
        "diagnosis_en": "flu",
        "diagnosis_pl": "grypa",
        "top_diagnoses": [
            {"diagnosis_en": "flu", "diagnosis_pl": "grypa", "probability": 0.78},
            {"diagnosis_en": "cold", "diagnosis_pl": "przeziębienie", "probability": 0.15},
            {"diagnosis_en": "covid_19", "diagnosis_pl": "COVID-19", "probability": 0.07},
        ],
    }

    resp = ChatResponse(**payload)

    assert resp.diagnosis_en == "flu"
    assert resp.diagnosis_pl == "grypa"

    # model_dump is what gets serialized to the frontend.
    dumped = resp.model_dump()
    assert dumped["diagnosis_en"] == "flu"
    assert dumped["diagnosis_pl"] == "grypa"
    assert "diagnosis_en" in dumped, "Frontend musi to widzieć w JSON-ie odpowiedzi"


def test_chat_response_diagnosis_fields_optional():
    """Short replies (e.g. follow-up questions) have no diagnosis yet."""
    payload = {
        "response": "Czy masz gorączkę?",
        "symptoms": ["cough"],
    }

    resp = ChatResponse(**payload)

    assert resp.diagnosis_en is None
    assert resp.diagnosis_pl is None
    assert resp.top_diagnoses == []


def test_chat_response_top_diagnoses_default_empty():
    payload = {"response": "ok", "symptoms": []}
    resp = ChatResponse(**payload)
    assert resp.top_diagnoses == []


def test_chat_response_validation_errors_on_missing_required():
    with pytest.raises(ValidationError):
        ChatResponse(symptoms=[])
    with pytest.raises(ValidationError):
        ChatResponse(response="ok")


def test_top_diagnosis_item_accepts_minimal_payload():
    item = TopDiagnosisItem(diagnosis_en="flu", probability=0.8)
    assert item.diagnosis_pl is None
    assert item.probability == 0.8


def test_chat_response_propagates_top_diagnosis_pl():
    """Each top_diagnoses item keeps its Polish name, which the frontend displays."""
    payload = {
        "response": "...",
        "symptoms": ["fever"],
        "top_diagnoses": [
            {"diagnosis_en": "flu", "diagnosis_pl": "grypa", "probability": 0.9},
        ],
    }
    resp = ChatResponse(**payload)
    assert resp.top_diagnoses[0].diagnosis_pl == "grypa"


# Runs through the real ChatService (not the global stub that hid the original bug);
# skipped until a lightweight AppState fixture exists.
@pytest.mark.integration
def test_chat_service_returns_diagnosis_fields_when_enough_symptoms(monkeypatch):
    """ChatService.handle_chat() must keep diagnosis_en/_pl in its ChatResponse."""
    pytest.importorskip("backend.services.chat_service")

    from backend.schemas import ChatRequest, ChatResponse  # noqa: F401

    pytest.skip("Wymaga lekkiego AppState fixture — uzupełnij wg lokalnej infra.")
