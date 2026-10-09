# Future plans

Ideas that are scoped but not implemented yet.

## English language support

Today the whole app is Polish: UI, symptom recognition, follow-up questions, bot messages and the LLM (Bielik). The models themselves work on English feature names, the dictionaries already have `en` fields and the API accepts a `lang` parameter, so the groundwork exists. Suggested order: three stages, each a separate commit.

### 1. UI with a language switch (medium effort)

- Move the UI texts (about 15 components, a few hundred strings) into a PL/EN dictionary.
- PL/EN switch in the header. The choice is remembered; the default follows the browser language.
- Set `<html lang>` accordingly, show symptom and disease names in the selected language, add tests.

### 2. English conversation in the backend (largest effort)

- Symptom recognition relies on Polish aliases and Polish negation. Add English aliases for all 470 symptoms and English negation patterns ("no fever", "I don't have...").
- Yes/no answers in English ("yes", "nope", ...).
- Translate the 390 follow-up questions (`backend/data/followup_questions_pl.json`) into English.
- Bot messages, urgency assessment and unmodeled symptoms in English (e.g. "photophobia", "light hurts my eyes").
- Bielik is a Polish model. It can answer in English, but with lower quality. Two options:
  - an English prompt for Bielik (answer quality must be checked on a machine with the model),
  - for English, a template-based summary without the LLM.
- Regression tests in both languages; run `tools/simulate_followups.py` for English as well to compare accuracy with Polish.

### 3. README in Polish and English with screenshots and GIF (medium effort)

- `README.md` (EN) and `README.pl.md` with a language switch at the top.
- New screenshots and `docs/demo.gif` in both languages (can be recorded with Playwright; the Bielik summary on the recording requires a machine with the model, otherwise template mode).
- Banners and SVG diagrams contain text, so they need Polish versions too.
