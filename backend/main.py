import os
import sys
from pathlib import Path

# The backend uses short imports (config, core, services, ...) plus the top-level
# "database" package. Both dirs go on sys.path and into PYTHONPATH (the uvicorn reloader
# subprocess inherits the env), so "python main.py" works from any cwd or IDE.
BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
for _p in (str(PROJECT_ROOT), str(BACKEND_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
_pp = [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
for _p in (str(BACKEND_DIR), str(PROJECT_ROOT)):
    if _p not in _pp:
        _pp.insert(0, _p)
os.environ["PYTHONPATH"] = os.pathsep.join(_pp)

import uvicorn

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    v = v.strip().lower()
    return v in ("1", "true", "yes", "on")


def _is_prod() -> bool:
    return os.getenv("APP_ENV", "dev").strip().lower() == "prod"


def _should_reload() -> bool:
    default = not _is_prod()
    return _env_bool("UVICORN_RELOAD", default)


def _resolve_uvicorn_target() -> str:
    # Single app module path; relies on backend/ being on sys.path (short imports).
    return "app:create_app"


def _collect_reload_dirs() -> list[str]:
    backend_dir = BACKEND_DIR
    project_root = PROJECT_ROOT

    candidates = [
        backend_dir,
        project_root / "database",
        backend_dir / "database",
    ]

    seen, out = set(), []
    for p in candidates:
        try:
            rp = str(p.resolve())
        except Exception:
            continue
        if p.exists() and rp not in seen:
            seen.add(rp)
            out.append(rp)
    return out


def _print_startup_banner(target: str, host: str, port: int, reload: bool) -> None:
    supabase_url = (os.getenv("SUPABASE_URL") or os.getenv("SUPABASE_PROJECT_URL") or "").strip()
    jwt_secret = os.getenv("SUPABASE_JWT_SECRET")
    print(
        "\n=== SymptomChecker API ===\n"
        f"  target        : {target}\n"
        f"  env           : {os.getenv('APP_ENV', 'dev')}\n"
        f"  host:port     : {host}:{port}\n"
        f"  reload        : {reload}\n"
        f"  reload_delay  : {os.getenv('UVICORN_RELOAD_DELAY', '0.5')} s\n"
        f"  log_level     : {os.getenv('LOG_LEVEL', 'info')}\n"
        f"  workers       : {os.getenv('WORKERS', '1')}\n"
        f"  SUPABASE_URL  : {'OK' if supabase_url else 'BRAK'}\n"
        f"  JWT secret    : {'set' if (jwt_secret and jwt_secret.strip()) else 'BRAK (logowanie Supabase nie zadziała)'}\n"
        "===========================\n"
    )


if __name__ == "__main__":
    target = _resolve_uvicorn_target()
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    reload = _should_reload()
    reload_dirs = _collect_reload_dirs()

    try:
        reload_delay = float(os.getenv("UVICORN_RELOAD_DELAY", "0.5"))
    except Exception:
        reload_delay = 0.5

    workers = int(os.getenv("WORKERS", "1"))
    if reload and workers != 1:
        print("Uwaga: reload wymaga WORKERS=1. Ustawiam WORKERS=1.")
        workers = 1

    _print_startup_banner(target, host, port, reload)

    uvicorn.run(
        target,
        factory=True,
        app_dir=str(BACKEND_DIR),
        host=host,
        port=port,
        workers=workers,
        reload=reload,
        reload_dirs=reload_dirs,
        reload_delay=reload_delay,
        reload_includes=["**/*.py", "**/*.sql", "**/*.yaml", "**/*.yml", "**/*.ini"],
        reload_excludes=[
            "ml/*", "weights/*", "data/*",
            "**/*.joblib", "**/*.gguf", "**/*.bin", "**/*.pt", "**/*.onnx",
            "**/*.db", "**/*.sqlite", "**/*.sqlite3", "**/*.log",
        ],
        log_level=os.getenv("LOG_LEVEL", "info"),
    )
