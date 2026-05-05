# backend/api/routers/health.py
from fastapi import APIRouter, Request

router = APIRouter(tags=["Health"])

@router.get("/health")
async def health(request: Request):
    st = getattr(request.app.state, "app_state", None)

    def _has(obj, attr=None):
        try:
            return bool(obj if attr is None else getattr(obj, attr, None))
        except Exception:
            return False

    return {
        "status": "ok" if st else "starting",
        "device": str(getattr(st, "device", None)) if st else None,

        # modele
        "xgb_ready": bool(st and _has(st, "xgb") and _has(st.xgb, "model")),
        "river_ready": bool(st and _has(st, "river") and _has(st.river, "model")),  # ⬅️ DODANE
        "llm_ready": bool(st and _has(st, "llm") and _has(st.llm, "model")),

        # pozostałe
        "matcher_ready": bool(st and _has(st, "matcher")),
    }
