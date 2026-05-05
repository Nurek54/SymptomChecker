from __future__ import annotations
from datetime import datetime, date
from sqlalchemy import String, DateTime, Integer, ForeignKey, Text, Float, Date, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .db import Base

class User(Base):
    __tablename__ = "users"
    __table_args__ = (
        # unikalność per dostawca OAuth
        UniqueConstraint("auth_provider", "provider_sub", name="uq_users_provider_sub_per_provider"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    username: Mapped[str] = mapped_column(String(50), unique=True, index=True)
    email:    Mapped[str] = mapped_column(String(254), unique=True, index=True)

    password_hash: Mapped[str | None] = mapped_column(String(255), nullable=True)

    created_at:    Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    first_name: Mapped[str | None] = mapped_column(String(80), nullable=True)
    last_name:  Mapped[str | None] = mapped_column(String(80), nullable=True)
    date_of_birth: Mapped[date | None] = mapped_column(Date, nullable=True)

    # 'local' | 'google' | 'supabase'
    auth_provider: Mapped[str] = mapped_column(String(20), default="local")
    provider_sub:  Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)

    conversations: Mapped[list["Conversation"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )

class Conversation(Base):
    __tablename__ = "conversations"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id:   Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True, index=True)
    session_id: Mapped[str] = mapped_column(String(64), index=True)
    title:      Mapped[str] = mapped_column(String(120), default="Rozmowa")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    last_message_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    user: Mapped["User"] = relationship(back_populates="conversations")
    messages: Mapped[list["Message"]] = relationship(back_populates="conversation", cascade="all, delete-orphan")
    extractions: Mapped[list["Extraction"]] = relationship(back_populates="conversation", cascade="all, delete-orphan")
    triage: Mapped[list["TriageResult"]] = relationship(back_populates="conversation", cascade="all, delete-orphan")

class Message(Base):
    __tablename__ = "messages"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    conversation_id: Mapped[int] = mapped_column(ForeignKey("conversations.id", ondelete="CASCADE"), index=True)
    role: Mapped[str] = mapped_column(String(16))      # "user" | "assistant"
    content: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    conversation: Mapped["Conversation"] = relationship(back_populates="messages")

class Extraction(Base):
    __tablename__ = "extractions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    conversation_id: Mapped[int] = mapped_column(ForeignKey("conversations.id", ondelete="CASCADE"), index=True)
    text: Mapped[str] = mapped_column(String(255))
    canonical: Mapped[str | None] = mapped_column(String(255))
    lang: Mapped[str | None] = mapped_column(String(8))
    score: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    conversation: Mapped["Conversation"] = relationship(back_populates="extractions")

class TriageResult(Base):
    __tablename__ = "triage_results"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    conversation_id: Mapped[int] = mapped_column(ForeignKey("conversations.id", ondelete="CASCADE"), index=True)
    label: Mapped[str] = mapped_column(String(255))
    score: Mapped[float | None] = mapped_column(Float)
    rank: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    conversation: Mapped["Conversation"] = relationship(back_populates="triage")
