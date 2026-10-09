from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field

# Input limits: short symptom ids and a bounded item count, so one request cannot send
# hundreds of thousands of items and exhaust server memory.
SymptomId = Annotated[str, Field(max_length=100)]
MAX_SYMPTOM_ITEMS = 200

# Chat schemas only; user and history schemas live in database/schemas.py.


class TopDiagnosisItem(BaseModel):
    diagnosis_en: str
    probability: float
    diagnosis_pl: str | None = None


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=64)
    user_input: str = Field(max_length=4000)
    lang: str = Field(default="pl", max_length=8)
    # Symptoms the frontend remembers; used only when the backend does not know the
    # session (e.g. after a server restart), so the chat does not start from scratch.
    client_symptoms: list[SymptomId] | None = Field(default=None, max_length=MAX_SYMPTOM_ITEMS)


class ChatResponse(BaseModel):
    response: str
    symptoms: list[str]
    top_diagnoses: list[TopDiagnosisItem] = Field(default_factory=list)
    diagnosis_en: str | None = None
    diagnosis_pl: str | None = None
