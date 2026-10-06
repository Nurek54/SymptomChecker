<a href="#see-it-in-action">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/banner-dark.svg" />
    <source media="(prefers-color-scheme: light)" srcset="docs/banner-light.svg" />
    <img src="docs/banner-light.svg" alt="Symptom Checker: describe how you feel in Polish, get the three most likely causes, explained" width="100%" />
  </picture>
</a>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white" alt="FastAPI" />
  <img src="https://img.shields.io/badge/Next.js-15-000000?style=flat-square&logo=nextdotjs&logoColor=white" alt="Next.js" />
  <img src="https://img.shields.io/badge/React-19-61DAFB?style=flat-square&logo=react&logoColor=black" alt="React" />
  <img src="https://img.shields.io/badge/TypeScript-3178C6?style=flat-square&logo=typescript&logoColor=white" alt="TypeScript" />
  <img src="https://img.shields.io/badge/Tailwind_CSS-06B6D4?style=flat-square&logo=tailwindcss&logoColor=white" alt="Tailwind CSS" />
  <img src="https://img.shields.io/badge/XGBoost-GPU-EB6E25?style=flat-square" alt="XGBoost" />
  <img src="https://img.shields.io/badge/River-online_learning-1F6FEB?style=flat-square" alt="River" />
  <img src="https://img.shields.io/badge/LLM-Bielik_11B_(GGUF)-8B5CF6?style=flat-square" alt="Bielik" />
  <img src="https://img.shields.io/badge/Supabase-auth-3FCF8E?style=flat-square&logo=supabase&logoColor=white" alt="Supabase" />
</p>

<p align="center">
  <a href="#see-it-in-action"><b>Demo</b></a> &nbsp;·&nbsp;
  <a href="#highlights"><b>Highlights</b></a> &nbsp;·&nbsp;
  <a href="#screenshots"><b>Screenshots</b></a> &nbsp;·&nbsp;
  <a href="#how-it-works"><b>How it works</b></a> &nbsp;·&nbsp;
  <a href="#getting-started"><b>Getting started</b></a> &nbsp;·&nbsp;
  <a href="#api-endpoints"><b>API</b></a>
</p>

> [!WARNING]
> **Educational / research project, not a medical device.** The output is not a diagnosis. If symptoms worsen or red flags appear, contact a doctor or emergency services (112).

**Symptom Checker** is a diagnostic chat assistant built as part of an engineering thesis (*praca inżynierska*). You describe how you feel in plain Polish; the app recognizes the symptoms (inflection and negation included, e.g. *"gorączki nie mam"*), asks the follow-up questions that best separate the likely diseases, and returns the **Top-3 most probable conditions** with a short, doctor-style explanation from a Polish LLM running on your own GPU.

Predictions come from a **calibrated ensemble** of XGBoost (offline, Bayesian-tuned on GPU) and River OVR / FTRL-Proximal (online, can keep learning), fused in log space with confidence-based weights.

## See it in action

<p align="center">
  <img src="docs/demo.gif" alt="Demo: the user describes symptoms, answers follow-up questions, gets the Top-3 and an urgency assessment, then opens the per-model breakdown" width="100%" />
</p>

<p align="center"><sub>First message → follow-up questions → Top-3 with explanation → quick action <em>Pilność</em> → XGBoost / River / Ensemble breakdown.</sub></p>

## Highlights

<table>
  <tr>
    <td width="33%" valign="top">
      <img src="docs/icons/languages.svg" width="44" alt="" /><br />
      <b>Understands Polish</b><br />
      <sub>Dictionary + multilingual embeddings, aware of inflection (<i>boli mnie głowa</i>) and negation (<i>gorączka już minęła</i>).</sub>
    </td>
    <td width="33%" valign="top">
      <img src="docs/icons/message-circle-question.svg" width="44" alt="" /><br />
      <b>Asks the right question</b><br />
      <sub>Picks the symptom that best separates the likely diseases, skips near-duplicates and stops after a limit instead of asking forever.</sub>
    </td>
    <td width="33%" valign="top">
      <img src="docs/icons/layers.svg" width="44" alt="" /><br />
      <b>Calibrated ensemble</b><br />
      <sub>XGBoost and River are each temperature-calibrated on held-out data, then the fusion gets its own temperature.</sub>
    </td>
  </tr>
  <tr>
    <td width="33%" valign="top">
      <img src="docs/icons/cpu.svg" width="44" alt="" /><br />
      <b>Local LLM</b><br />
      <sub>Bielik 11B v2.3 (GGUF) via <code>ctransformers</code> writes the explanation in natural Polish prose. No external AI API.</sub>
    </td>
    <td width="33%" valign="top">
      <img src="docs/icons/eye.svg" width="44" alt="" /><br />
      <b>Transparent</b><br />
      <sub>Per-model Top-3 and the fusion weights are one click away in the UI; full metrics land in <code>training_report.json</code>.</sub>
    </td>
    <td width="33%" valign="top">
      <img src="docs/icons/zap.svg" width="44" alt="" /><br />
      <b>Quick actions</b><br />
      <sub>One click for diagnosis, recommendations, urgency, OTC medicines, prevention or a summary.</sub>
    </td>
  </tr>
  <tr>
    <td width="33%" valign="top">
      <img src="docs/icons/user-round-check.svg" width="44" alt="" /><br />
      <b>Accounts or guest mode</b><br />
      <sub>Supabase auth with saved history, or start chatting right away as a guest.</sub>
    </td>
    <td width="33%" valign="top">
      <img src="docs/icons/timer.svg" width="44" alt="" /><br />
      <b>Training with ETA</b><br />
      <sub>Every long step logs elapsed and remaining time; startup is split into 5 timed stages.</sub>
    </td>
    <td width="33%" valign="top">
      <img src="docs/icons/flask-conical.svg" width="44" alt="" /><br />
      <b>Tested</b><br />
      <sub>pytest (unit, API, regressions), vitest and Playwright end-to-end; leak-free train / calibration / test split.</sub>
    </td>
  </tr>
</table>

## Screenshots

<p align="center">
  <img src="docs/hero.png" alt="Light and dark theme side by side" width="100%" />
</p>

<table>
  <tr>
    <td width="50%" valign="top">
      <img src="docs/screenshots/02-followup.png" alt="Follow-up questions" />
      <p align="center"><b>Adaptive follow-up questions</b><br /><sub>Recognized symptoms appear on the right as you type.</sub></p>
    </td>
    <td width="50%" valign="top">
      <img src="docs/screenshots/03-model-split.png" alt="XGBoost vs River vs Ensemble" />
      <p align="center"><b>Look inside the ensemble</b><br /><sub>XGBoost, River and the fused result side by side, with the weights used.</sub></p>
    </td>
  </tr>
  <tr>
    <td width="50%" valign="top">
      <img src="docs/screenshots/04-symptom-picker.png" alt="Symptom picker" />
      <p align="center"><b>Symptom picker</b><br /><sub>Hundreds of Polish symptom names grouped by body system.</sub></p>
    </td>
    <td width="50%" valign="top">
      <img src="docs/screenshots/06-history.png" alt="Conversation history" />
      <p align="center"><b>Conversation history</b><br /><sub>Saved chats with extracted symptoms and the diagnoses at that time.</sub></p>
    </td>
  </tr>
</table>

<details>
<summary><b>More screens</b> (visit preparation, help, account, login)</summary>
<br />

| Prepare a description for the doctor | Built-in guide |
|:---:|:---:|
| <img src="docs/screenshots/08-prep.png" alt="Visit preparation" /> | <img src="docs/screenshots/09-howto.png" alt="How to use" /> |
| **Account menu** | **Login or guest mode** |
| <img src="docs/screenshots/07-user-menu.png" alt="Account menu" /> | <img src="docs/screenshots/10-login.png" alt="Login" /> |

</details>

<sub>Screenshots and the demo come from the real frontend; the backend responses in them are scripted so the example is reproducible.</sub>

## How it works

### One chat turn

```mermaid
sequenceDiagram
    autonumber
    actor U as User
    participant FE as Next.js UI
    participant API as FastAPI /chat
    participant M as Symptom matcher
    participant E as Ensemble (XGB + River)
    participant Q as Follow-up selector
    participant L as Bielik 11B

    U->>FE: "Od wczoraj mam gorączkę i boli mnie gardło"
    FE->>API: POST /chat {session_id, user_input}
    API->>M: extract symptoms (+ negations)
    M-->>API: fever, sore_throat
    API->>E: posterior over ~570 diseases
    alt fewer than MIN_SYMPTOMS or not confident yet
        API->>Q: most informative next symptom
        Q-->>API: swollen_lymph_nodes
        API-->>FE: "Czy masz powiększone węzły chłonne?"
    else enough evidence
        API->>L: prompt with symptoms + Top-3
        L-->>API: explanation in Polish
        API-->>FE: answer + Top-3 + symptoms
    end
    FE-->>U: chat bubble, symptom chips, Top-3 bars
```

### Architecture

```mermaid
flowchart LR
    subgraph FE["Frontend · Next.js 15"]
        UI["ChatUI · SidePanel<br/>History · Overlays"]
        ST[("Zustand store")]
    end
    subgraph BE["Backend · FastAPI"]
        R["Routers<br/>chat · sessions · history<br/>dicts · river · auth"]
        CS["ChatService"]
        MA["SymptomMatcher<br/>dictionary + embeddings"]
        FS["Follow-up selector<br/>P(symptom | disease)"]
        subgraph ENS["Calibrated ensemble"]
            X["XGBoost<br/>T + prior correction"]
            RV["River OVR / FTRL<br/>(λ, T)"]
            F["log-linear fusion<br/>+ T_ens"]
        end
        LLM["Bielik 11B GGUF"]
    end
    DB[("PostgreSQL / SQLite")]
    SB["Supabase Auth"]

    UI <--> ST
    UI -- "REST + JWT" --> R
    UI -. login .-> SB
    R --> CS
    CS --> MA
    CS --> FS
    CS --> ENS
    X --> F
    RV --> F
    CS --> LLM
    R --> DB
```

### Training pipeline (first start)

```mermaid
flowchart LR
    A["Data split<br/>build_datasets.py"] --> B["1 · XGBoost<br/>Bayes search<br/>+ calibration"]
    B --> C["2 · Bielik LLM"]
    C --> D["3 · Symptom<br/>statistics"]
    D --> E["4 · River<br/>online pass<br/>+ (λ, T)"]
    E --> F["5 · Ensemble<br/>T_ens + test<br/>metrics"]
    F --> G[("training_report<br/>.json")]
```

Every stage logs elapsed and remaining time, so a long first run is never a black box ([more](#progress-and-remaining-time)):

```text
[XGB][Bayes 3/32] runda 410/1446 | test mlogloss=0.2121 | minęło 2m10s | do limitu rund ~5m30s
[XGB][Train] 40% (560/1400 rund) | minęło 1m12s | zostało ~1m48s
[River/trening]  31.3% (35571/113416) 52.1 wierszy/s | minęło 11m22s | zostało ~25m
```

### Calibrated probabilities

Example from a 100-disease subset of the v2 data (held-out test set). Full numbers for your run are written to `backend/ml/training_report.json`.

| Model | Top-1 | NLL before → after | ECE before → after |
|---|:---:|:---:|:---:|
| XGBoost | 98.3% | 0.064 → **0.044** | 2.1% → **0.4%** |
| River | 96.6% | 0.441 → **0.091** | 22.6% → **1.3%** |
| **Ensemble** | **98.6%** | **0.040** | **0.5%** |

<sub>NLL = negative log-likelihood, ECE = expected calibration error (15 bins). Lower is better.</sub>

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python 3.11+, FastAPI, Pydantic v2, SQLAlchemy 2.x |
| Frontend | Next.js 15 (App Router), React 19, TypeScript, Tailwind CSS, Zustand |
| Classical ML | XGBoost (`hist`, CUDA), scikit-learn, scikit-optimize |
| Online learning | River 0.23 (OneVsRest LogisticRegression, FTRL-Proximal) |
| LLM | Bielik 11B v2.3 Instruct (GGUF Q6_K) via `ctransformers` |
| Embeddings | `sentence-transformers` (multilingual MiniLM) |
| Auth | Supabase JWT (HS256 + JWKS), python-jose |
| Persistence | PostgreSQL (Supabase) or SQLite via SQLAlchemy |
| Testing | pytest, pytest-asyncio, vitest, Playwright |

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
├── docs/                     # README images: hero, demo.gif, screenshots/
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

1. **Load training data** from `DATA_CSV` (default: `data/v2/train.csv` when present, see [Training datasets](#training-datasets))
2. **Bayesian hyperparameter search** - `XGB_N_ITER_BAYES` × `XGB_CV_SPLITS`-fold CV (default: 32 × 3 = 96 fits)
3. **Final training** on the training set with the best params
4. **Temperature calibration** on held-out data the model has never seen (`calibration.csv`; if it is missing, a stratified hold-out is cut from the training set *before* training)
5. **Save artifacts** to `backend/ml/` (plus `xgb_model.meta.json` recording which data the model was trained on)

> To retrain from scratch, delete `backend/ml/xgb_model.joblib`, `label_encoder.joblib`, `xgb_best_params.joblib`, `xgb_model.calibration.joblib`, `river_ovr_ftrl.joblib`, `river_ovr_ftrl.calibration.joblib` and `ensemble_calibration.json`. Models trained by older versions are still loaded, but the backend logs a warning.

After that, every subsequent startup just **loads the saved models** (~5–10 s).

### Progress and remaining time

Every long step logs its progress with elapsed time and an estimate of the remaining time (see the example in [How it works](#training-pipeline-first-start)).

Startup is split into stages `[ETAP 1/5]` (XGB) ... `[ETAP 5/5]` (ensemble calibration), each with its duration, and the total time is logged at the end. The River bootstrap is a single online pass over the training set (one update per class per row), so with ~570 classes it takes roughly 15–40 min, once.

### Calibration and quality report

All three predictors are calibrated on `calibration.csv` (never seen in training) and measured on `test.csv`:

- **XGBoost**: `p = softmax((margins - log w_c) / T)`. `T` is fitted by minimum NLL. The class-weight prior correction `log w_c` (training with class weights inflates rare classes) is used only if it lowers NLL on the calibration set.
- **River (OVR, FTRL-Proximal)**: `p = softmax((z - lam * z_base) / T)`, where `z` are the raw per-class logits and `z_base` the logits for an empty input; `(lam, T)` are fitted together and the same transform is used at prediction time.
- **Ensemble**: the log-linear fusion of both models (dynamic weights and gating) gets its own temperature `T_ens`, cached in `backend/ml/ensemble_calibration.json` and recomputed only when a model changes.

The metrics (top-1, top-3, NLL, ECE, Brier, mean confidence) before and after calibration are logged as tables and saved to `backend/ml/training_report.json`.

### LLM loading

Bielik 11B Q6_K is **not trained** - only loaded into memory:

| Setup | Load time |
|-------|-----------|
| GPU + NVMe SSD | ~30–90 s |
| CPU + NVMe | ~1–2 min |
| HDD | painfully slow - move the GGUF to an SSD |

Once loaded, generation is fast (a few seconds per Top-3 summary on GPU).

## Dataset Format

<details>
<summary><b>Expected CSV and dictionary formats</b></summary>

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

A list of entries mapping Polish symptom names (and synonyms) to feature columns. Already shipped in the repo. Format:

```json
[
  {
    "en": "Sore throat",
    "pl": "ból gardła",
    "xgb_label": ["sore_throat"],
    "synonyms_pl": ["drapanie w gardle", "ból przy przełykaniu"],
    "synonyms_en": [],
    "category": "Otolaryngologiczny"
  }
]
```

`xgb_label` must match CSV column names. When it lists several columns, the matcher picks the one with the most training data and skips columns that never occur in the training set (`build_datasets.py` reports dictionary entries without data).

### 6. Disease dictionary - `diseases_pl_en.json`

Same idea for diseases, used to show Polish names in the UI and in LLM prompts:

```json
[
  { "en": "Strep throat", "pl": "Angina paciorkowcowa", "xgb_label": "strep_throat" }
]
```

</details>

## API Endpoints

<details>
<summary><b>All endpoints</b></summary>

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

</details>

## Configuration

<details>
<summary><b>All settings</b></summary>

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

</details>

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
