# tests/test_chat_schema_unit.py
import pytest
from pydantic import ValidationError

from backend.schemas import ChatResponse


def test_chat_response_valid_payload():
    payload = {
        "response": "OK",
        "symptoms": ["fever", "cough"],
        "top_diagnoses": [
            {"diagnosis_en": "flu", "diagnosis_pl": "grypa", "probability": 0.8},
            {"diagnosis_en": "cold", "diagnosis_pl": "przeziębienie", "probability": 0.2},
        ],
    }

    resp = ChatResponse(**payload)

    assert resp.response == "OK"
    assert resp.symptoms == ["fever", "cough"]
    assert len(resp.top_diagnoses) == 2
    assert resp.top_diagnoses[0].diagnosis_en == "flu"


def test_chat_response_requires_key_fields():
    """
    Wymagane: response, symptoms, top_diagnoses (jak masz w schemas.py).
    """
    with pytest.raises(ValidationError):
        # brak 'response'
        ChatResponse(symptoms=[], top_diagnoses=[])

    with pytest.raises(ValidationError):
        # brak 'symptoms'
        ChatResponse(response="OK", top_diagnoses=[])


def test_chat_response_no_validator_errors_on_correct_payload():
    """Przy poprawnych danych nie ma wyjątków walidacji."""
    payload = {
        "response": "Diagnoza stub.",
        "symptoms": ["fever"],
        "top_diagnoses": [
            {"diagnosis_en": "flu", "probability": 0.9},
        ],
    }
    ChatResponse(**payload)  # brak wyjątku = OK
