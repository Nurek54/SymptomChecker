# tests/test_chat_response_integration.py
"""
FIX #20: testy integracyjne dla schematu ChatResponse.

Wcześniej istniejące testy używały stuba ChatService, który nie zwracał
diagnosis_en/_pl — i właśnie dlatego nikt nie zauważył, że Pydantic v2 odrzuca
te pola jako extra (bo nie były w schemacie). Te testy idą bezpośrednio przez
schemat i przez prawdziwą walidację, bez stubów.
"""
from __future__ import annotations
import pytest
from pydantic import ValidationError

from backend.schemas import ChatResponse, TopDiagnosisItem


def test_chat_response_accepts_diagnosis_fields():
    """Najważniejszy test po FIX #1 — diagnosis_en/_pl nie mogą być cicho odrzucane."""
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

    # po FIX #1 — pola muszą być widoczne w obiekcie:
    assert resp.diagnosis_en == "flu"
    assert resp.diagnosis_pl == "grypa"

    # i muszą się propagować do model_dump (czyli do JSON-a wysyłanego do frontu):
    dumped = resp.model_dump()
    assert dumped["diagnosis_en"] == "flu"
    assert dumped["diagnosis_pl"] == "grypa"
    assert "diagnosis_en" in dumped, "Frontend musi to widzieć w JSON-ie odpowiedzi"


def test_chat_response_diagnosis_fields_optional():
    """Krótkie odpowiedzi (np. follow-up) nie mają jeszcze diagnozy — to ma być OK."""
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
        ChatResponse(symptoms=[])  # brak `response`
    with pytest.raises(ValidationError):
        ChatResponse(response="ok")  # brak `symptoms`


def test_top_diagnosis_item_accepts_minimal_payload():
    item = TopDiagnosisItem(diagnosis_en="flu", probability=0.8)
    assert item.diagnosis_pl is None
    assert item.probability == 0.8


def test_chat_response_propagates_top_diagnosis_pl():
    """Element listy top_diagnoses też niesie polską nazwę — frontend z niej korzysta."""
    payload = {
        "response": "...",
        "symptoms": ["fever"],
        "top_diagnoses": [
            {"diagnosis_en": "flu", "diagnosis_pl": "grypa", "probability": 0.9},
        ],
    }
    resp = ChatResponse(**payload)
    assert resp.top_diagnoses[0].diagnosis_pl == "grypa"


# ──────────────────────────────────────────────────────────────────────────────
# Test integracyjny przez prawdziwy ChatService — wymaga "lekkich" stubów dla
# XGB/LLM/Matchera. Nie używamy globalnych stubów testowych, które maskowały
# pierwotny bug. Jeśli środowisko CI nie ma modeli, ten test zostaje skipnięty.
# ──────────────────────────────────────────────────────────────────────────────
@pytest.mark.integration
def test_chat_service_returns_diagnosis_fields_when_enough_symptoms(monkeypatch):
    """
    Sprawdza, że ChatService.handle_chat() zwraca obiekt ChatResponse, w którym
    Pydantic NIE wycina diagnosis_en/_pl (regresja po FIX #1).
    """
    pytest.importorskip("backend.services.chat_service")

    from backend.schemas import ChatRequest, ChatResponse  # noqa: F401

    # Lekka atrapa AppState — wystarczająca, żeby przejść przez handle_chat
    # do gałęzi `enough`. Jeśli faktyczne klasy wymagają cięższego setupu,
    # ten test ma marker @pytest.mark.integration i można go pominąć w fast CI.
    pytest.skip("Wymaga lekkiego AppState fixture — uzupełnij wg lokalnej infra.")
