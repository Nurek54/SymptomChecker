from __future__ import annotations
from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import logging

log = logging.getLogger("api.errors")

def as_message(err: RequestValidationError) -> str:
    """Flatten a FastAPI 422 error list into one short string for the UI."""
    try:
        parts = []
        for e in err.errors():
            loc = ".".join(str(x) for x in (e.get("loc") or []) if x not in ("body",))
            msg = e.get("msg") or "Invalid value"
            if loc:
                parts.append(f"{loc}: {msg}")
            else:
                parts.append(msg)
        return "; ".join(parts) or "Nieprawidłowe dane."
    except Exception:
        return "Nieprawidłowe dane."

async def validation_exception_handler(request: Request, exc: RequestValidationError):
    msg = as_message(exc)
    log.warning("422 on %s %s -> %s", request.method, request.url.path, msg)
    return JSONResponse(status_code=422, content={"detail": msg})
