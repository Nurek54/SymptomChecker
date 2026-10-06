from __future__ import annotations

from pydantic import BaseModel, Field

# Schematy czatu. Schematy użytkownika i historii są w database/schemas.py
# (jedno źródło prawdy, wcześniej były zduplikowane tutaj).


class TopDiagnosisItem(BaseModel):
    diagnosis_en: str
    probability: float
    diagnosis_pl: str | None = None


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=64)
    user_input: str = Field(max_length=4000)
    lang: str = "pl"
    # Objawy, które pamięta frontend. Używane tylko wtedy, gdy backend nie zna
    # sesji (np. po restarcie serwera), żeby nie zaczynać rozmowy od zera.
    client_symptoms: list[str] | None = None


class ChatResponse(BaseModel):
    response: str
    symptoms: list[str]
    top_diagnoses: list[TopDiagnosisItem] = Field(default_factory=list)
    diagnosis_en: str | None = None
    diagnosis_pl: str | None = None
