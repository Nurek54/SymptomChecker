from backend.config import MIN_SYMPTOMS


def test_sessions_add_and_remove_symptom(client):
    sid = "sess-add-remove"

    body = {"replace": ["fever", "cough"], "lang": "pl"}
    r = client.post(f"/sessions/{sid}/symptoms", json=body)
    assert r.status_code == 200
    data = r.json()

    assert sorted(data["symptoms"]) == ["cough", "fever"]

    body = {"add": ["sore_throat"], "lang": "pl"}
    r = client.post(f"/sessions/{sid}/symptoms", json=body)
    assert r.status_code == 200
    data = r.json()
    assert sorted(data["symptoms"]) == ["cough", "fever", "sore_throat"]

    body = {"remove": ["cough"], "lang": "pl"}
    r = client.post(f"/sessions/{sid}/symptoms", json=body)
    assert r.status_code == 200
    data = r.json()
    assert sorted(data["symptoms"]) == ["fever", "sore_throat"]


def test_sessions_only_known_symptoms_allowed(client):
    """Unknown symptom keys are silently filtered out of the session."""
    sid = "sess-known"

    body = {"replace": ["fever", "non_existing_symptom_x"], "lang": "pl"}
    r = client.post(f"/sessions/{sid}/symptoms", json=body)
    assert r.status_code == 200
    data = r.json()
    assert "non_existing_symptom_x" not in data["symptoms"]


def test_sessions_top3_empty_when_below_min_symptoms(client):
    sid = "sess-below-min"

    body = {"replace": ["fever"], "lang": "pl"}  # < MIN_SYMPTOMS
    r = client.post(f"/sessions/{sid}/symptoms", json=body)
    assert r.status_code == 200
    data = r.json()

    assert len(data["symptoms"]) == 1
    assert data["top_diagnoses"] == []


def test_get_top3_split_existing_session(client):
    sid = "sess-top3"

    body = {"replace": ["fever", "cough", "sore_throat"], "lang": "pl"}
    r = client.post(f"/sessions/{sid}/symptoms", json=body)
    assert r.status_code == 200

    resp = client.get(f"/sessions/{sid}/top3-split?k=3&lang=pl")
    assert resp.status_code == 200

    data = resp.json()
    assert set(data.keys()) >= {
        "xgb_enabled",
        "river_enabled",
        "symptoms",
        "xgb_top",
        "river_top",
        "ensemble_top",
        "meta",
    }

    # Symptoms come from a set in SessionStore, so order is not guaranteed.
    assert set(data["symptoms"]) == {"fever", "cough", "sore_throat"}
    assert isinstance(data["xgb_top"], list)
    assert isinstance(data["ensemble_top"], list)


def test_get_top3_split_initializes_missing_session(client):
    """An unknown session id is created on demand instead of returning 404."""
    sid = "sess-init-on-top3"

    resp = client.get(f"/sessions/{sid}/top3-split")
    assert resp.status_code == 200
    data = resp.json()
    assert data["symptoms"] == []
