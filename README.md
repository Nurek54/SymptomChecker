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
  <img src="https://img.shields.io/badge/License-All_Rights_Reserved-red?style=flat-square" alt="License" />
</p>

<br />

<p align="center">
  <img src="preview.png" alt="SymptomChecker - Hero" width="720" />
</p>

---

## About

**SymptomChecker** is a research-grade diagnostic chat assistant developed as part of an engineering thesis (*praca inżynierska*). The user describes their symptoms in Polish (English also supported), the system extracts them through a hybrid dictionary + multilingual embedding matcher, asks adaptive follow-up questions, and finally returns the **Top-3 most probable conditions** with a short clinical-style summary generated locally by a Polish LLM (Bielik 11B).

The classifier is a **dynamic-weight ensemble** of XGBoost (offline, deeply tuned) and River OVR/FTRL (online, incrementally updated) - letting the model both leverage strong batch training and adapt to drift over time.

> ⚠️ **Disclaimer.** This is an educational / research project. **It is NOT a substitute for professional medical advice.** The output is not a medical diagnosis. If your symptoms worsen or red flags appear, contact a doctor or emergency services.

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
├── LICENSE
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

The training datasets used for this project are **not redistributed in this repository** for size and licensing reasons. If you would like to obtain them for academic review, evaluation, or reproducing the results of the thesis, please contact the author (see [Contact](#contact)) - datasets will be shared on request.

The application expects the following files at the paths defined in `backend/config.py`:

| Path | Required for | Approx. size |
|------|--------------|--------------|
| `backend/data/Final_Dataset_Diseases_and_Symptoms.csv` | XGBoost initial training | ~130 MB |
| `backend/data/Final_Dataset_Diseases_and_Symptoms.class_weight.csv` | Per-class sample weights (auto-applied) | ~30 KB |
| `backend/data/calibration.csv` | River temperature calibration | ~30 MB |
| `backend/data/train_remainder.csv` | River incremental updates (optional) | ~100 MB |
| `backend/data/Final_Testset_for_Eval.csv` | Held-out evaluation (already in repo) | ~13 MB |

**Recommended: rebuild the splits once** with `python data/build_datasets.py` (run in `backend/`). It writes `backend/data/v2/` (`train.csv`, `calibration.csv`, `test.csv`, `class_weight.csv`, `build_report.json`), which `config.py` uses automatically when present. The rebuild fixes problems found in the original files:

- every row of `Final_Testset_for_Eval.csv` also appears in `train_remainder.csv` or `calibration.csv` (test metrics were inflated); the new splits share no symptom vector,
- duplicate labels for the same disease coming from the second source dataset (e.g. `bronchial_asthma`/`asthma`, `dengue`/`dengue_fever`) are merged,
- diseases with fewer than 10 examples are removed (`--min-rows`),
- class weights are `sqrt(N / (K * n_c))` clipped to `[0.5, 5]` instead of reaching ~200.

After rebuilding, delete the old model artifacts (see [What happens on first start](#what-happens-on-first-start)) so the models are retrained on the new data.

If you want to use **your own dataset** instead, see the [Dataset Format](#dataset-format) section below - as long as your CSVs match the expected schema, the pipeline will train, calibrate, and evaluate without any code changes.

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

1. **Load training data** from `DATA_CSV` (default: `train_remainder.csv` if present, otherwise `Final_Dataset_Diseases_and_Symptoms.csv`)
2. **Bayesian hyperparameter search** - `XGB_N_ITER_BAYES` × `XGB_CV_SPLITS`-fold CV (default: 32 × 3 = 96 fits)
3. **Final training** on the training set with the best params
4. **Temperature calibration** on held-out data the model has never seen (`calibration.csv`; if it is missing, a stratified hold-out is cut from the training set *before* training)
5. **Save artifacts** to `backend/ml/` (plus `xgb_model.meta.json` recording which data the model was trained on)

> `train_remainder.csv` + `calibration.csv` together form `Final_Dataset_Diseases_and_Symptoms.csv`. Training on the remainder keeps calibration honest. Models trained by older versions (on the full dataset, calibrated on training rows) are still loaded, but the backend logs a warning. To retrain, delete `backend/ml/xgb_model.joblib`, `label_encoder.joblib`, `xgb_model.calibration.joblib`, `river_ovr_ftrl.joblib` and `river_ovr_ftrl.calibration.joblib`.

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

If you don't have the original training data and want to use your own, the CSV files must follow the schema below. Match the schema and the entire training, calibration and evaluation pipeline will work without code changes.

### 1. Main training CSV - `Final_Dataset_Diseases_and_Symptoms.csv`

A **wide binary feature matrix** in CSV form. One row = one labeled patient case.

**Required columns:**

| Column | Type | Description |
|--------|------|-------------|
| `diseases` | string | Target label - disease name (e.g. `"Influenza"`). **Required.** |
| `<symptom_name>` × N | int (0/1) | One column per symptom, binary presence/absence. Names must match the **English keys** in `backend/data/symptoms_pl_en.json`. |
| `weight` | float | *(Optional)* Per-sample weight. Auto-derived from `class_weight.csv` if absent; defaults to `1.0`. |

**Reserved column names** (do not use as features): `diseases`, `xgb_label`, `weight`.

**Example:**

```csv
diseases,fever,cough,headache,nausea,fatigue,sore_throat,chest_pain,...
Influenza,1,1,1,0,1,1,0,...
COVID-19,1,1,0,1,1,1,0,...
Migraine,0,0,1,1,1,0,0,...
Common Cold,0,1,1,0,1,1,0,...
Pneumonia,1,1,0,0,1,0,1,...
```

**Notes:**
- Rows with `NaN` are dropped automatically.
- Symptom column names are **case-sensitive** and must match the dictionary keys exactly.
- The classifier internally uses `uint8` 0/1 - strings or floats will be coerced or rejected.
- Each disease class should have **at least 3 samples** for stratified 3-fold CV; classes with fewer samples are kept for final training but excluded from hyperparameter search.

### 2. Class weight CSV - `Final_Dataset_Diseases_and_Symptoms.class_weight.csv`

Used as a fallback to populate `weight` when the main CSV doesn't include it. Already shipped in the repo, but you can regenerate it from your dataset.

| Column | Type | Description |
|--------|------|-------------|
| `disease` | string | Disease name (must match values in `diseases` column of main CSV) |
| `count` | int | Number of samples for this disease |
| `class_weight` | float | Weight multiplier applied to samples of this class |

**Example:**

```csv
disease,count,class_weight
Influenza,1850,1.0
COVID-19,2100,0.88
Migraine,420,4.4
Pneumonia,950,1.95
```

### 3. Calibration CSV - `calibration.csv`

Same schema as the main training CSV. A held-out portion (typically 5–10% of training data, stratified by disease) used by the River temperature calibrator. The `weight` column is optional here.

### 4. Test set CSV - `Final_Testset_for_Eval.csv`

Same schema as the main training CSV. Used by `ml_validation_runner` for end-to-end evaluation. **Already shipped in the repo.**

### 5. Symptom dictionary - `symptoms_pl_en.json`

Maps Polish symptom phrases to English keys (which become CSV column names). Already shipped in the repo. Format:

```json
{
  "gorączka": "fever",
  "kaszel": "cough",
  "ból głowy": "headache",
  "ból gardła": "sore_throat",
  ...
}
```

If you add new symptoms to your CSV, you also need to add the PL↔EN mapping here so the symptom matcher can extract them from user input.

### 6. Disease dictionary - `diseases_pl_en.json`

Same idea, but for diseases - used by the LLM prompt localizer to translate Top-3 results into Polish. Format:

```json
{
  "Influenza": "Grypa",
  "Migraine": "Migrena",
  "Pneumonia": "Zapalenie płuc",
  ...
}
```

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

Read endpoints (`status`, `features`, `predict_topk`) are open. Endpoints that change the model or read CSV files require the `X-Admin-Token` header matching `ADMIN_API_TOKEN` (they are disabled when the variable is empty) and accept only `.csv` files from `backend/data/`. They operate on the same River model the chat ensemble uses.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/river/status` | Model status, sample counts, calibration state |
| `GET` | `/river/features` | Current River feature list |
| `POST` | `/river/predict_topk` | Predict from a feature vector |
| `POST` | `/river/train_one` | Single-sample online update (admin) |
| `POST` | `/river/incremental_batch` | Mini-batch online update (admin) |
| `POST` | `/river/fit_from_csv` | Incremental fit from a CSV in `backend/data/` (admin) |
| `GET` | `/river/evaluate` | Evaluate against the test set (admin) |

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
| `ADMIN_API_TOKEN` | env | empty | Token for `/river/*` admin endpoints (empty = disabled) |

### XGBoost training

| Setting | Default | Effect |
|---------|---------|--------|
| `XGB_N_ITER_BAYES` | `32` | Bayes search iterations. **Lower = faster first run, slightly less optimal model.** Try `8–16` for dev |
| `XGB_CV_SPLITS` | `3` | CV folds |
| `XGB_USE_GPU` | `True` | Use CUDA |
| `XGB_TREE_METHOD` | `"hist"` | Fastest tree algorithm |
| `XGB_CALIBRATE` | `True` | Apply temperature scaling (fitted on held-out data) |
| `XGB_CALIBRATION_CSV` | `data/calibration.csv` | Held-out calibration set (must differ from `DATA_CSV`) |

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
| `ENSEMBLE_K_FUSE` | `0` | `0` = fuse full probability distributions (recommended); `>0` = only Top-K from each model |
| `ENSEMBLE_MIN_XGB_WEIGHT` / `ENSEMBLE_MIN_RIVER_WEIGHT` | `0.55` / `0.15` | Minimum share of each model |
| `ENSEMBLE_CONF_GAP` | `0.20` | When one model is more confident by this much, it gets its maximum allowed share |

### LLM generation

| Setting | Default | Purpose |
|---------|---------|---------|
| `LLM_MAX_NEW_TOKENS` | `2000` | Max generated tokens |
| `LLM_TEMP` | `0.72` | Sampling temperature |
| `LLM_TOP_P` / `LLM_TOP_K` | `0.92` / `40` | Nucleus / top-k sampling |
| `LLM_CONTEXT_LEN` | `8192` | Context window |
| `LLM_GPU_LAYERS` | `999` | Layers offloaded to GPU (`999` = all) |

### Chat flow

| Setting | Default | Purpose |
|---------|---------|---------|
| `MIN_SYMPTOMS` | `3` | Minimum symptoms before allowing diagnosis |
| `STOP_FOLLOWUPS_AT` | `4` | After N follow-up questions without reaching `MIN_SYMPTOMS`, give an indicative (low-confidence) assessment instead of asking forever |
| `TOPN_DIAGNOSES` | `3` | Number of diagnoses returned |
| `SUMMARY_MODE` | `"llm"` | `"llm"` or `"template"` |

## License

**Copyright © 2026 Grzegorz Sienkiewicz. All Rights Reserved.**

This software was developed as part of an engineering thesis (*praca inżynierska*). It is published on GitHub for **academic review and portfolio purposes only**. No license is granted to use, copy, modify, distribute, sublicense, or commercialize the code, in whole or in part, without the prior written consent of the author.

If you would like to use any part of this project - for academic citation, research, evaluation, or any other purpose - please contact the author (see [Contact](#contact) below). Permissions will be considered on a case-by-case basis.

Third-party open-source dependencies remain governed by their own licenses (see `backend/requirements.txt` and `frontend/package.json`).

See the [LICENSE](LICENSE) file for full terms.

## Contact

**Grzegorz Sienkiewicz**
✉️ gsienkiewicz03@gmail.com
🐙 GitHub: [@Nurek54](https://github.com/Nurek54)

For training dataset access, license inquiries, or questions about the thesis, please reach out via email.
