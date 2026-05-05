<p align="center">
  <br />
  <strong>⚕</strong>
  <br />
  <br />
</p>

<h1 align="center">SYMPTOM CHECKER</h1>

<p align="center">
  <em>A Polish-language diagnostic chat assistant powered by classical ML and a local LLM</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-blue?style=flat-square" alt="Python" />
  <img src="https://img.shields.io/badge/FastAPI-0.110+-009688?style=flat-square" alt="FastAPI" />
  <img src="https://img.shields.io/badge/Next.js-15-black?style=flat-square" alt="Next.js" />
  <img src="https://img.shields.io/badge/XGBoost-2.x-orange?style=flat-square" alt="XGBoost" />
  <img src="https://img.shields.io/badge/Bielik-11B-purple?style=flat-square" alt="Bielik LLM" />
  <img src="https://img.shields.io/badge/License-TBD-lightgrey?style=flat-square" alt="License" />
</p>

<br />

<p align="center">
  <img src="preview.png" alt="SymptomChecker - Hero" width="720" />
</p>

---

## About

**SymptomChecker** is a research-grade diagnostic chat assistant. The user describes their symptoms in Polish (English also supported), the system extracts them through a hybrid dictionary + multilingual embedding matcher, asks adaptive follow-up questions, and finally returns the **Top-3 most probable conditions** with a short clinical-style summary generated locally by a Polish LLM (Bielik 11B).

The classifier is a **dynamic-weight ensemble** of XGBoost (offline, deeply tuned) and River OVR/FTRL (online, incrementally updated) - letting the model both leverage strong batch training and adapt to drift over time.

> ⚠️ **Disclaimer.** This is an educational / research project. **It is NOT a substitute for professional medical advice.** The output is not a medical diagnosis. If your symptoms worsen or red flags appear, contact a doctor or emergency services (112 / 911).

## User Flow

```
Symptom input (PL/EN)  →  Dictionary + embedding match  →  Follow-up loop  →  Ensemble Top-3  →  LLM summary
```

The user never sees raw probabilities or model internals - only the chat, the running symptom list, and the final diagnosis card.

## Features

- **Bilingual symptom extraction** - multilingual `sentence-transformers` matcher backed by a curated PL/EN dictionary, with rule-first fallback for short / ambiguous tokens
- **Adaptive follow-up** - model-split selector picks the next best question based on which features would most reduce ensemble disagreement
- **Hybrid ensemble** - XGBoost (Bayesian-tuned, 32 iters × 3-fold CV, GPU-accelerated) fused with River OVR Logistic / FTRL (online learning) using dynamic confidence-gap weights
- **Local LLM** - Bielik 11B v2.3 Instruct (GGUF Q6_K) via `ctransformers`, no external API calls, no data leaves the machine
- **Calibrated probabilities** - temperature scaling on both XGBoost and River outputs for honest confidence
- **Stateful sessions** - server-side session store tracks symptoms, follow-ups asked, and prediction history per user
- **Auth** - Supabase JWT (HS256 + JWKS fallback) with local profile sync, plus optional guest mode
- **Persistent history** - SQLAlchemy 2.x ORM stores conversations, messages, extracted symptoms and triage results
- **Validation pipeline** - `ml_validation_runner` produces full reports: confusion matrices, per-class F1, top-50 frequent/rare slices, all rendered as SVG
- **Production-leaning hygiene** - CORS, secure cookies, request validation, sensitive-data masking in logs
- **Comprehensive test suite** - pytest for backend (units + API), vitest for component logic, Playwright for end-to-end flows

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python 3.11+, FastAPI, Pydantic v2, SQLAlchemy 2.x |
| Frontend | Next.js 15 (App Router), React 19, TypeScript, Tailwind CSS, Zustand |
| Classical ML | XGBoost 2.x (`hist` + GPU), scikit-learn, scikit-optimize |
| Online learning | River (OVR Logistic / FTRL) |
| LLM | Bielik 11B v2.3 Instruct (GGUF Q6_K) via `ctransformers` |
| Embeddings | `sentence-transformers` (multilingual MiniLM) |
| Auth | Supabase JWT (HS256 + JWKS), python-jose, passlib/bcrypt |
| Persistence | SQLite / PostgreSQL via SQLAlchemy |
| Testing | pytest, pytest-asyncio, vitest, Playwright |

## High-level Architecture

```
┌─────────────────┐     POST /chat       ┌──────────────────────────────────┐
│  Next.js (3000) │ ───────────────────► │  FastAPI (8000)                  │
│  ChatUI / Auth  │ ◄─── JSON {top3,…} ─ │                                  │
└─────────────────┘                      │  SymptomMatcher (dict + embed)   │
                                         │           │                       │
                                         │           ▼                       │
                                         │  ┌──────────────────────────┐    │
                                         │  │ Ensemble                 │    │
                                         │  │  XGBoost  ─┐             │    │
                                         │  │           ├─► Top-K      │    │
                                         │  │  River    ─┘   diagnoses │    │
                                         │  └──────────┬───────────────┘    │
                                         │             ▼                     │
                                         │  Followup selector (model split) │
                                         │             │                     │
                                         │             ▼                     │
                                         │  PromptBuilder ► LLM (Bielik)    │
                                         │             │                     │
                                         │             ▼                     │
                                         │  PromptLocalizer (PL/EN)         │
                                         └──────────────────────────────────┘
                                                       │
                                                       ▼
                                              SQLite / Postgres
```

## Project Structure

```
SymptomChecker/
├── backend/
│   ├── api/routers/          # auth, chat, dicts, history, river, sessions, errors, health
│   ├── core/                 # startup, AppState
│   ├── data/                 # PL/EN dictionaries, small testset (large CSVs gitignored)
│   ├── ml/                   # river_ovr_ftrl.joblib (in repo); xgb_model.joblib (excluded)
│   ├── models/               # llm_manager, river_manager, xgb_manager
│   ├── reports/              # ML validation outputs (SVG, JSON)
│   ├── services/             # ensemble, prompt_builder, symptom_matcher, sessions, …
│   ├── tests/                # pytest
│   ├── weights/bielik-gguf/  # ⬅ drop Bielik GGUF here (gitignored)
│   ├── config.py             # ALL ML / LLM / auth knobs
│   ├── app.py                # FastAPI factory
│   ├── main.py               # uvicorn entrypoint
│   ├── ml_validation_runner.py
│   └── requirements.txt
├── database/                 # SQLAlchemy: models, CRUD, security, db init
├── frontend/
│   └── src/
│       ├── app/              # Next.js App Router pages
│       ├── components/       # ChatUI, AuthSheet, SidePanel, …
│       ├── lib/              # api, supabaseClient, types
│       ├── store/            # Zustand store
│       └── tests/            # unit + e2e
├── logs/
├── .env.example
├── .gitignore
└── README.md
```

## Getting Started

### Prerequisites

- **Python 3.11** or newer
- **Node.js 18+** and npm
- **CUDA-capable GPU** strongly recommended (for fast XGBoost training and LLM inference). CPU works but is significantly slower.
- **~25 GB free disk space** (LLM weights, ML models, datasets)
- **Supabase project** - optional; the app can run in guest mode without it

### Backend

```bash
cd backend
python -m venv ../.venv
..\.venv\Scripts\Activate.ps1   # Windows
source ../.venv/bin/activate     # Linux / macOS

pip install -r requirements.txt
cp .env.example .env             # fill in at minimum AUTH_SECRET_KEY
```

> `bitsandbytes` is commented out by default - known issues on Windows. Uncomment on Linux if you need 8-bit quantization for the Transformers fallback path.

### Download external artifacts

These files are excluded from the repo due to size - you need to provide them yourself.

#### LLM weights - Bielik 11B GGUF

Download `Bielik-11B-v2.3-Instruct.Q6_K.gguf` (~8.7 GB) from HuggingFace:

🔗 https://huggingface.co/speakleash/Bielik-11B-v2.3-Instruct-GGUF

Place at:

```
backend/weights/bielik-gguf/Bielik-11B-v2.3-Instruct.Q6_K.gguf
```

Other quantizations (Q4_K_M, Q5_K_M) work too - update `LLM_SLOTS["bielik"]["FILENAME"]` in `backend/config.py`.

#### Training datasets

Place your training data at the paths defined in `backend/config.py`:

| Path | Required for | Approx. size |
|------|--------------|--------------|
| `backend/data/Final_Dataset_Diseases_and_Symptoms.csv` | XGBoost initial training | ~130 MB |
| `backend/data/calibration.csv` | River temperature calibration | ~30 MB |
| `backend/data/train_remainder.csv` | River incremental updates (optional) | ~100 MB |
| `backend/data/Final_Testset_for_Eval.csv` | held-out evaluation (already in repo) | ~13 MB |

> *TODO (repo owner): if your training data is public, link it here. If private, document how to regenerate it from `Dataset_1/` or `Dataset_2/`.*

### Frontend

```bash
cd frontend
npm install
cp .env.example .env.local       # fill in NEXT_PUBLIC_SUPABASE_URL etc.
```

### Running

```bash
# Terminal 1 - backend
cd backend
python main.py
# default: http://127.0.0.1:8000  (Swagger docs at /docs)

# Terminal 2 - frontend
cd frontend
npm run dev
# http://localhost:3000
```

## First Run - Training Pipeline

The very first time you start the backend, **the system trains the ML models from scratch** on your dataset. Subsequent runs reuse the trained artifacts and start in seconds.

### What happens on first start

`StartupManager` checks for the presence of:

- `backend/ml/xgb_model.joblib` - main XGBoost classifier
- `backend/ml/label_encoder.joblib` - disease label encoder
- `backend/ml/xgb_best_params.joblib` - Bayes-optimized hyperparameters
- `backend/ml/river_ovr_ftrl.joblib` - River OVR/FTRL model (shipped in the repo)
- `backend/ml/river_ovr_ftrl.calibration.joblib` - River temperature calibrator

If any of the XGB artifacts are missing, the full pipeline runs:

1. **Load training data** from `DATA_CSV` (default: `Final_Dataset_Diseases_and_Symptoms.csv`)
2. **Bayesian hyperparameter search** - `XGB_N_ITER_BAYES` × `XGB_CV_SPLITS`-fold CV (default: 32 × 3 = 96 fits)
3. **Final training** on the full dataset with the best params
4. **Temperature calibration** for probability outputs
5. **Save artifacts** to `backend/ml/`

After that, every subsequent startup just **loads the saved models** (~5–10 s).

### Approximate training times

Estimates assume a dataset on the order of ~100k–1M rows × ~400 binary symptom features. Your mileage will vary.

| Hardware | Bayes search | Final training | Total first run |
|----------|--------------|----------------|-----------------|
| **CUDA GPU** (RTX 3080+, `XGB_USE_GPU=True`) | ~10–25 min | ~1–3 min | **~15–30 min** |
| **Modern CPU** (12+ cores) | ~1.5–4 h | ~5–15 min | **~2–4 h** |
| **Older CPU** (4–8 cores) | ~4–8 h | ~15–30 min | **~5–9 h** |

Speed up dev iterations by lowering `XGB_N_ITER_BAYES` to `8–16` in `backend/config.py`.

### River calibration

- **First start with `RIVER_CALIBRATE_ON_START=True`**: reads `calibration.csv`, fits a temperature scalar - typically **30–90 seconds**
- **Subsequent starts**: calibrator loaded from disk (~1 s)

### LLM loading

Bielik 11B Q6_K is **not trained** - only loaded into memory:

| Setup | Load time |
|-------|-----------|
| GPU + NVMe SSD | ~30–90 s |
| CPU + NVMe | ~1–2 min |
| HDD | painfully slow - move the GGUF to an SSD |

Once loaded, generation is fast (a few seconds per Top-3 summary on GPU).

## Dataset Format

The XGBoost trainer expects a **wide binary feature matrix** in CSV form.

### Required columns

- **`diseases`** *(string)* - the target label (disease name). Required.
- **One column per symptom** *(integer 0 or 1)* - binary presence/absence. Names must match the symptom keys in `backend/data/symptoms_pl_en.json` (English side).
- **`weight`** *(float, optional)* - sample weight. Auto-derived from `class_weight.csv` if absent, or defaults to `1.0`.

Reserved column names: `diseases`, `xgb_label`, `weight` - don't use these as features.

### Example

```csv
diseases,fever,cough,headache,nausea,fatigue,sore_throat,...
Influenza,1,1,1,0,1,1,...
COVID-19,1,1,0,1,1,1,...
Migraine,0,0,1,1,1,0,...
Common Cold,0,1,1,0,1,1,...
```

Notes:
- Rows with `NaN` are dropped automatically.
- Symptom column names are **case-sensitive**.
- Internally the classifier uses `uint8` 0/1 - strings or floats will be coerced or rejected.

The `calibration.csv` file (used by River) follows the same schema.

## API Endpoints

### Chat
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/chat` | Main entry - send user message, get extracted symptoms + Top-3 diagnoses + LLM summary |

### Auth
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/auth/session` | Create a session from a Supabase JWT |
| `GET` | `/auth/me` | Get current user profile |
| `PUT` | `/auth/me` | Update current user profile |
| `POST` | `/auth/logout` | End session |

### Sessions (chat state)
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/sessions/{session_id}/symptoms` | Patch the symptom set for a session |
| `GET` | `/sessions/{session_id}/top3-split` | Get Top-3 with per-model breakdown (XGB vs River) |

### History
| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/history` | List all conversations for the current user |
| `GET` | `/history/{conv_id}` | Conversation detail (messages, extractions, triage) |
| `DELETE` | `/history/{conv_id}` | Delete a conversation |

### Dictionaries
| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/dicts/symptoms` | Full PL/EN symptom dictionary |

### River (online learning)
| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/river/status` | Model status, sample counts, calibration state |
| `GET` | `/river/features` | Current River feature list |
| `POST` | `/river/predict_topk` | Predict from a feature vector |
| `POST` | `/river/train_one` | Single-sample online update |
| `POST` | `/river/incremental_batch` | Mini-batch online update |
| `POST` | `/river/fit_from_csv` | Fit/re-fit from a CSV |
| `GET` | `/river/evaluate` | Evaluate against the test set |

### Health
| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Liveness probe |

Full Swagger docs at `http://127.0.0.1:8000/docs` once the backend is running.

## Configuration

All ML / LLM / auth knobs live in **`backend/config.py`**. Runtime overrides come from `.env`. Below are the most relevant settings - full list in the file itself.

### Runtime / environment

| Setting | Source | Default | Purpose |
|---------|--------|---------|---------|
| `APP_ENV` | env | `dev` | `dev` or `prod` |
| `HOST` / `PORT` | env | `127.0.0.1` / `8000` | uvicorn bind |
| `FRONTEND_ORIGIN` | env | `http://localhost:3000` | CORS origin |
| `LOG_LEVEL` | env | `info` | logger level |

### Auth / sessions

| Setting | Source | Default | Purpose |
|---------|--------|---------|---------|
| `AUTH_SECRET_KEY` | env | - | JWT secret (fallback when no Supabase) |
| `SUPABASE_URL` | env | - | Supabase project URL |
| `SUPABASE_JWT_SECRET` | env | - | Supabase HS256 JWT secret |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | env | `60` | JWT lifetime |
| `SESSION_COOKIE_SECURE` | env | `false` | set `true` over HTTPS |

### XGBoost training

| Setting | Default | Effect |
|---------|---------|--------|
| `XGB_N_ITER_BAYES` | `32` | Bayes search iterations. **Lower = faster first run, slightly less optimal model.** Try `8–16` for dev |
| `XGB_CV_SPLITS` | `3` | CV folds |
| `XGB_USE_GPU` | `True` | Use CUDA |
| `XGB_TREE_METHOD` | `"hist"` | Fastest tree algorithm |
| `XGB_CALIBRATE` | `True` | Apply temperature scaling |

### River

| Setting | Default | Purpose |
|---------|---------|---------|
| `RIVER_CALIBRATE_ON_START` | `True` | Recalibrate on every startup |
| `RIVER_ALPHA` | `0.1` | FTRL learning rate |
| `RIVER_L1` / `RIVER_L2` | `1e-6` / `1e-4` | Regularization |
| `RIVER_WEIGHT_CAP` | `25.0` | Max class-weight clamp |

### Ensemble

| Setting | Default | Purpose |
|---------|---------|---------|
| `ENSEMBLE_W_XGB` / `ENSEMBLE_W_RIVER` | `0.8` / `0.2` | Base weights |
| `ENSEMBLE_K_FUSE` | `9` | Top-K from each model before merge |
| `ENSEMBLE_MIN_XGB_WEIGHT` | `0.55` | XGB floor (River never fully dominates) |
| `ENSEMBLE_CONF_GAP` | `0.20` | "One model is clearly more confident" threshold |

### LLM generation

| Setting | Default | Purpose |
|---------|---------|---------|
| `LLM_MAX_NEW_TOKENS` | `2000` | Max generated tokens |
| `LLM_TEMP` | `0.55` | Sampling temperature |
| `LLM_TOP_P` / `LLM_TOP_K` | `0.9` / `40` | Nucleus / top-k sampling |
| `LLM_CONTEXT_LEN` | `8192` | Context window |
| `LLM_GPU_LAYERS` | `999` | Layers offloaded to GPU (`999` = all) |

### Chat flow

| Setting | Default | Purpose |
|---------|---------|---------|
| `MIN_SYMPTOMS` | `3` | Minimum symptoms before allowing diagnosis |
| `STOP_FOLLOWUPS_AT` | `4` | Force diagnosis after N follow-up turns |
| `TOPN_DIAGNOSES` | `3` | Number of diagnoses returned |
| `SUMMARY_MODE` | `"llm"` | `"llm"` or `"template"` |

## Tests

```bash
# Backend
cd backend
pytest

# Frontend - unit (vitest)
cd frontend
npm run test

# Frontend - e2e (Playwright)
cd frontend
npx playwright install      # first run only
npm run test:e2e
```

## Roadmap / Known Issues

- [ ] Move `_uid()` from three routers into a shared `Depends`
- [ ] `SessionStore` - add TTL + eviction; not safe under multi-worker setups
- [ ] Refactor `prompt_builder.py` (620 lines, four responsibilities)
- [ ] Replace `print()` with the project logger in ML modules
- [ ] Replace `conftest.py` monkey-patches with proper integration tests
- [ ] Move hardcoded thresholds in `symptom_matcher` into config

## Acknowledgments

- **Bielik LLM** by [SpeakLeash](https://huggingface.co/speakleash) - the open Polish-language model that powers the diagnosis summaries
- Built with the help of **vibe coding** - AI-assisted development through natural-language conversation with [Claude](https://claude.ai)
- XGBoost, scikit-learn, River, FastAPI, Next.js - the open-source giants this project stands on

## License

*TODO - choose one (MIT / Apache-2.0 / GPL / proprietary).*

## Contact

Grzegorz Sienkiewicz - [@Nurek54](https://github.com/Nurek54)
