from __future__ import annotations
from pydantic import BaseModel, ConfigDict, EmailStr, Field
from datetime import datetime, date

# —— AUTH / USER —— #
class UserOut(BaseModel):
    id: int
    username: str
    email: EmailStr
    first_name: str | None = None
    last_name: str | None = None
    date_of_birth: date | None = None
    age: int | None = None
    auth_provider: str = "local"

    model_config = ConfigDict(from_attributes=True)

class RegisterIn(BaseModel):
    username: str = Field(min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)
    first_name: str | None = None
    last_name: str | None = None
    date_of_birth: date | None = None

class LoginIn(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1)

class UpdateProfileIn(BaseModel):
    username: str | None = Field(default=None, min_length=3, max_length=50)
    first_name: str | None = None
    last_name: str | None = None
    date_of_birth: date | None = None

# —— HISTORIA CZATÓW —— #
class ConversationItem(BaseModel):
    id: int
    title: str
    last_message_at: datetime
    message_count: int

class MessageOut(BaseModel):
    role: str
    content: str
    created_at: datetime

class SymptomOut(BaseModel):
    text: str
    canonical: str | None = None
    lang: str | None = None
    score: float | None = None

class DiagnosisOut(BaseModel):
    label: str
    score: float | None = None
    rank: int | None = None

class ConversationDetail(BaseModel):
    id: int
    title: str
    messages: list[MessageOut]
    symptoms: list[SymptomOut] = Field(default_factory=list)
    diagnoses: list[DiagnosisOut] = Field(default_factory=list)
