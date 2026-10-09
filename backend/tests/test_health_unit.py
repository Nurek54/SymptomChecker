import types

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.api.routers.health import router as health_router


def _client_with_state(st) -> TestClient:
    app = FastAPI()
    app.include_router(health_router)
    app.state.app_state = st
    return TestClient(app)


def test_health_starting_when_no_state():
    app = FastAPI()
    app.include_router(health_router)
    client = TestClient(app)

    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()

    assert data["status"] == "starting"
    assert data["device"] is None
    assert data["xgb_ready"] is False
    assert data["river_ready"] is False
    assert data["llm_ready"] is False
    assert data["matcher_ready"] is False


def test_health_ok_when_all_models_present():
    st = types.SimpleNamespace(
        device="cpu",
        xgb=types.SimpleNamespace(model=object()),
        river=types.SimpleNamespace(model=object()),
        llm=types.SimpleNamespace(model=object()),
        matcher=object(),
    )
    client = _client_with_state(st)
    resp = client.get("/health")
    data = resp.json()

    assert data["status"] == "ok"
    assert data["device"] == "cpu"
    assert data["xgb_ready"] is True
    assert data["river_ready"] is True
    assert data["llm_ready"] is True
    assert data["matcher_ready"] is True


def test_health_offline_llm_changes_flag():
    st = types.SimpleNamespace(
        device="cuda:0",
        xgb=types.SimpleNamespace(model=object()),
        river=types.SimpleNamespace(model=object()),
        llm=types.SimpleNamespace(model=None),
        matcher=object(),
    )
    client = _client_with_state(st)
    resp = client.get("/health")
    data = resp.json()

    assert data["status"] == "ok"
    assert data["llm_ready"] is False
    assert data["xgb_ready"] is True
    assert data["matcher_ready"] is True
