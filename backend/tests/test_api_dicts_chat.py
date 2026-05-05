# tests/test_api_dicts_chat.py
from backend.config import MIN_SYMPTOMS


def test_get_dicts_symptoms_200_and_schema(client):
    resp = client.get("/dicts/symptoms")
    assert resp.status_code == 200

    data = resp.json()
    assert isinstance(data, list)
    assert len(data) > 0

    first = data[0]
    assert set(first.keys()) >= {"en", "pl", "category"}
    assert first["en"]  # niepuste
    assert first["pl"]  # niepuste


def test_get_dicts_symptoms_no_nulls_in_key_fields(client):
    resp = client.get("/dicts/symptoms")
    data = resp.json()

    for row in data:
        assert row["en"] is not None
        assert row["pl"] is not None


def test_chat_happy_path_enough_symptoms(client):
    """
    Happy path: input z ≥ MIN_SYMPTOMS objawów → 200 + response + top_diagnoses (3).
    W naszym stubie ChatService to jest sztucznie zrobione.
    """
    session_id = "test-session-happy"

    # ustawiamy objawy przez /sessions, zgodnie z SymptomsPatch (replace)
    body = {
        "replace": ["fever", "cough", "sore_throat"],
        "lang": "pl",
    }
    r = client.post(f"/sessions/{session_id}/symptoms", json=body)
    assert r.status_code == 200

    # teraz /chat – stubowy ChatService powinien zwrócić top3
    payload = {
        "session_id": session_id,
        "user_input": "Mam gorączkę, kaszel i ból gardła.",
        "lang": "pl",
    }
    resp = client.post("/chat", json=payload)
    assert resp.status_code == 200

    data = resp.json()
    assert "response" in data
    assert isinstance(data["response"], str)
    assert "top_diagnoses" in data
    assert len(data["top_diagnoses"]) == 3


def test_chat_validation_too_few_symptoms(client):
    """
    Za mało objawów → ścieżka z komunikatem i pustym top-3.
    """
    session_id = "test-session-too-few"

    body = {
        "replace": ["fever"],  # < MIN_SYMPTOMS
        "lang": "pl",
    }
    r = client.post(f"/sessions/{session_id}/symptoms", json=body)
    assert r.status_code == 200

    payload = {
        "session_id": session_id,
        "user_input": "Mam tylko gorączkę.",
        "lang": "pl",
    }
    resp = client.post("/chat", json=payload)

    # W naszej testowej ChatService to dalej 200 + pusty top-3.
    # Jeżeli u Ciebie zwracasz 422/400 z Pydantic, zmień asercję.
    assert resp.status_code == 200

    data = resp.json()
    assert data["top_diagnoses"] == []


def test_chat_hygiene_no_doses_in_response(client):
    """
    Toksyczne wejście / tekst lekowy – sprawdzamy, że w odpowiedzi
    nie ma dawek typu '400 mg', '500 mg' itp.
    """
    payload = {
        "session_id": "test-hygiene-dose",
        "user_input": "Biorę ibuprofen 400 mg co 6 godzin i paracetamol 500 mg.",
        "lang": "pl",
    }
    resp = client.post("/chat", json=payload)
    assert resp.status_code == 200

    text = resp.json()["response"].lower()
    assert "400 mg" not in text
    assert "500 mg" not in text


def test_chat_hygiene_no_medicine_names_backend(client):
    """
    F3 (backend): odpowiedź nie zawiera nazw leków / trade names.
    Nawet jeśli ChatService jest stubem, test pilnuje,
    żeby backend nie odbijał nazw z wejścia 1:1.
    """
    payload = {
        "session_id": "test-hygiene-names",
        "user_input": "Biorę ibuprofen 400 mg, Apap i Nurofen na przemian.",
        "lang": "pl",
    }
    resp = client.post("/chat", json=payload)
    assert resp.status_code == 200

    text = resp.json()["response"].lower()

    banned_names = [
        "ibuprofen",
        "apap",
        "nurofen",
        "paracetamol",
        "aspirin",
    ]

    for name in banned_names:
        assert name not in text
