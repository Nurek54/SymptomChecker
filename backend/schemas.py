from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field

# Limity wejścia: krótkie identyfikatory objawów, rozsądna liczba elementów (bez nich jedno żądanie
# mogło wysłać setki tysięcy pozycji i zająć pamięć serwera).
SymptomId = Annotated[str, Field(max_length=100)]
MAX_SYMPTOM_ITEMS = 200

# Schematy czatu. Schematy użytkownika i historii są w database/schemas.py
# (jedno źródło prawdy, wcześniej były zduplikowane tutaj).


class TopDiagnosisItem(BaseModel):
    diagnosis_en: str
    probability: float
    diagnosis_pl: str | None = None


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=64)
    user_input: str = Field(max_length=4000)
    lang: str = Field(default="pl", max_length=8)
    # Objawy, które pamięta frontend. Używane tylko wtedy, gdy backend nie zna
    # sesji (np. po restarcie serwera), żeby nie zaczynać rozmowy od zera.
    client_symptoms: list[SymptomId] | None = Field(default=None, max_length=MAX_SYMPTOM_ITEMS)


class ChatResponse(BaseModel):
    response: str
    symptoms: list[str]
    top_diagnoses: list[TopDiagnosisItem] = Field(default_factory=list)
    diagnosis_en: str | None = None
    diagnosis_pl: str | None = None
