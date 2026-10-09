"use client";

import React, { useEffect, useMemo, useRef, useState } from "react";
import { useAuth } from "./AuthProvider";
import {
  X, LogIn, UserPlus, User2, Mail, Lock, Calendar, PencilLine,
  Loader2, ShieldAlert, Eye, EyeOff, ChevronDown,
} from "lucide-react";

function msgFromError(e: unknown): string {
  const err = (e ?? {}) as {
    message?: unknown;
    response?: { data?: { detail?: unknown; message?: unknown } };
  };
  const detail =
    err.response?.data?.detail ||
    err.response?.data?.message ||
    err.message ||
    "Wystąpił błąd.";
  return typeof detail === "string" ? detail : JSON.stringify(detail);
}

export default function AuthSheet() {
  const { authOpen, closeAuth, authMode, login, register, continueAsGuest } = useAuth();
  const [mode, setMode] = useState<"login" | "register">(authMode);

  const [username, setUsername] = useState("");
  const [email, setEmail]       = useState("");
  const [password, setPassword] = useState("");
  const [firstName, setFirstName] = useState("");
  const [lastName,  setLastName]  = useState("");
  const [dob, setDob] = useState(""); // YYYY-MM-DD

  const [showPass, setShowPass] = useState(false);
  const [showOptional, setShowOptional] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => setMode(authMode), [authMode]);

  // Focus the email field on every open (autoFocus ran only once on mount,
  // focusing the still-hidden dialog)
  const emailRef = useRef<HTMLInputElement | null>(null);
  useEffect(() => {
    if (!authOpen) return;
    const id = requestAnimationFrame(() => emailRef.current?.focus());
    return () => cancelAnimationFrame(id);
  }, [authOpen]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") closeAuth(); };
    if (authOpen) window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [authOpen, closeAuth]);

  const canSubmit = useMemo(() => {
    const emailOk = /\S+@\S+\.\S+/.test(email);
    const passOk = password.length >= 8;
    if (mode === "login") return emailOk && passOk;
    const userOk = username.trim().length >= 3 && username.trim().length <= 50;
    return userOk && emailOk && passOk;
  }, [mode, email, password, username]);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!canSubmit) return;
    setBusy(true); setErr(null);
    try {
      if (mode === "login") {
        await login(email, password);
      } else {
        await register(email, password, username, firstName || undefined, lastName || undefined, dob || undefined);
      }
      closeAuth();
    } catch (e: unknown) {
      setErr(msgFromError(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    // inert keeps the closed (merely transparent) dialog out of Tab focus order
    <div className={`fixed inset-0 z-50 ${authOpen ? "" : "pointer-events-none"}`} aria-hidden={!authOpen} inert={!authOpen}>
      <div
        onClick={closeAuth}
        className={`absolute inset-0 bg-black/40 transition-opacity ${authOpen ? "opacity-100" : "opacity-0"}`}
      />

      <div
        role="dialog"
        aria-modal="true"
        aria-label={mode === "login" ? "Okno logowania" : "Okno rejestracji"}
        className="absolute inset-0 m-0 flex items-center justify-center p-4"
      >
        <div
          className={[
            "w-full max-w-[420px] rounded-2xl border border-base bg-card shadow-2xl",
            "transition will-change-transform",
            authOpen ? "scale-100 opacity-100" : "scale-95 opacity-0",
            "flex flex-col",
          ].join(" ")}
        >
          <div className="h-1 w-full rounded-t-2xl bg-gradient-to-r from-emerald-400 via-sky-400 to-violet-400" />

          <div className="flex items-center justify-between border-b border-base bg-card px-4 py-3">
            <div className="flex items-center gap-2">
              <span className="grid h-8 w-8 place-items-center rounded-lg border border-base bg-input">
                {mode === "login" ? <LogIn className="h-4 w-4" /> : <UserPlus className="h-4 w-4" />}
              </span>
              <h2 className="text-base font-semibold">
                {mode === "login" ? "Logowanie" : "Rejestracja"}
              </h2>
            </div>
            <button
              onClick={closeAuth}
              className="rounded-lg border border-base bg-card p-2 text-xs shadow-sm transition hover:brightness-110 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-neutral-300"
              aria-label="Zamknij"
            >
              <X className="h-4 w-4" />
            </button>
          </div>

          <div className="px-4 pt-3">
            <div className="inline-flex overflow-hidden rounded-xl border border-base bg-card">
              <button
                onClick={() => setMode("login")}
                className={[
                  "inline-flex items-center gap-1.5 px-3 py-1.5 text-sm transition",
                  mode === "login" ? "bg-input font-medium text-basecolor" : "text-muted hover:brightness-110",
                ].join(" ")}
              >
                <LogIn className="h-4 w-4" />
                Mam konto
              </button>
              <button
                onClick={() => setMode("register")}
                className={[
                  "inline-flex items-center gap-1.5 px-3 py-1.5 text-sm transition",
                  mode === "register" ? "bg-input font-medium text-basecolor" : "text-muted hover:brightness-110",
                ].join(" ")}
              >
                <UserPlus className="h-4 w-4" />
                Utwórz konto
              </button>
            </div>
          </div>

          <form onSubmit={submit} className="px-4 pb-4 space-y-3">
            {mode === "register" && (
              <>
                <div className="space-y-1">
                  <label className="text-xs text-muted">Nazwa użytkownika</label>
                  <div className="flex items-center gap-2 rounded-lg border border-base bg-input px-2 focus-within:bg-card focus-within:ring-2 focus-within:ring-neutral-400">
                    <User2 className="h-4 w-4 opacity-70" />
                    <input
                      value={username}
                      onChange={(e) => setUsername(e.target.value)}
                      required
                      autoComplete="username"
                      minLength={3}
                      maxLength={50}
                      className="h-10 flex-1 bg-transparent text-sm outline-none"
                      placeholder="np. jan_kowalski"
                    />
                  </div>
                </div>

                <div className="mt-1">
                  <button
                    type="button"
                    onClick={() => setShowOptional((v) => !v)}
                    className="group flex w-full items-center justify-between rounded-lg border border-base bg-card px-3 py-2 text-xs font-medium transition hover:brightness-110"
                    aria-expanded={showOptional}
                    aria-controls="optional-fields"
                  >
                    Dane opcjonalne
                    <ChevronDown className={`h-4 w-4 transition-transform ${showOptional ? "rotate-180" : ""}`} />
                  </button>
                  {showOptional && (
                    <div id="optional-fields" className="mt-2 space-y-2">
                      <div className="grid grid-cols-2 gap-2">
                        <div className="space-y-1">
                          <label className="text-xs text-muted">Imię</label>
                          <div className="flex items-center gap-2 rounded-lg border border-base bg-input px-2 focus-within:bg-card focus-within:ring-2 focus-within:ring-neutral-400">
                            <PencilLine className="h-4 w-4 opacity-70" />
                            <input
                              value={firstName}
                              maxLength={80}
                              onChange={(e) => setFirstName(e.target.value)}
                              className="h-10 flex-1 bg-transparent text-sm outline-none"
                              placeholder="Jan"
                            />
                          </div>
                        </div>
                        <div className="space-y-1">
                          <label className="text-xs text-muted">Nazwisko</label>
                          <div className="flex items-center gap-2 rounded-lg border border-base bg-input px-2 focus-within:bg-card focus-within:ring-2 focus-within:ring-neutral-400">
                            <PencilLine className="h-4 w-4 opacity-70" />
                            <input
                              value={lastName}
                              maxLength={80}
                              onChange={(e) => setLastName(e.target.value)}
                              className="h-10 flex-1 bg-transparent text-sm outline-none"
                              placeholder="Kowalski"
                            />
                          </div>
                        </div>
                      </div>

                      <div className="space-y-1">
                        <label className="text-xs text-muted">Data urodzenia</label>
                        <div className="flex items-center gap-2 rounded-lg border border-base bg-input px-2 focus-within:bg-card focus-within:ring-2 focus-within:ring-neutral-400">
                          <Calendar className="h-4 w-4 opacity-70" />
                          <input
                            type="date"
                            value={dob}
                            onChange={(e) => setDob(e.target.value)}
                            className="h-10 flex-1 bg-transparent text-sm outline-none"
                          />
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              </>
            )}

            <div className="space-y-1">
              <label className="text-xs text-muted">Email</label>
              <div className="flex items-center gap-2 rounded-lg border border-base bg-input px-2 focus-within:bg-card focus-within:ring-2 focus-within:ring-neutral-400">
                <Mail className="h-4 w-4 opacity-70" />
                <input
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  type="email"
                  required
                  autoComplete="email"
                  ref={emailRef}
                  className="h-10 flex-1 bg-transparent text-sm outline-none"
                  placeholder="ty@przyklad.pl"
                />
              </div>
            </div>

            <div className="space-y-1">
              <label className="text-xs text-muted">Hasło</label>
              <div className="flex items-center gap-2 rounded-lg border border-base bg-input px-2 focus-within:bg-card focus-within:ring-2 focus-within:ring-neutral-400">
                <Lock className="h-4 w-4 opacity-70" />
                <input
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  type={showPass ? "text" : "password"}
                  required
                  minLength={8}
                  autoComplete={mode === "login" ? "current-password" : "new-password"}
                  className="h-10 flex-1 bg-transparent text-sm outline-none"
                  placeholder="min. 8 znaków"
                />
                <button
                  type="button"
                  onClick={() => setShowPass((v) => !v)}
                  className="rounded-md px-2 py-1 text-xs text-muted transition hover:brightness-110"
                  aria-pressed={showPass}
                  aria-label={showPass ? "Ukryj hasło" : "Pokaż hasło"}
                >
                  {showPass ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                </button>
              </div>
            </div>

            {err && (
              <div
                className="rounded-xl border border-base px-3 py-2 text-xs"
                role="alert"
                style={{ background: "color-mix(in srgb, currentColor 8%, transparent)" }}
              >
                {err}
              </div>
            )}

            <button
              type="submit"
              disabled={!canSubmit || busy}
              className="btn-primary w-full inline-flex items-center justify-center gap-2 rounded-xl px-4 py-2 text-sm font-medium shadow-sm transition hover:shadow-md focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-neutral-300 disabled:opacity-60"
            >
              {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : null}
              {mode === "login" ? "Zaloguj się" : "Zarejestruj się"}
            </button>

            <div className="space-y-2 pt-1">
              <div className="flex items-start gap-2 text-xs text-muted">
                <ShieldAlert className="mt-0.5 h-4 w-4 opacity-70" />
                <p>Tryb gościa: historia nie będzie zapisana.</p>
              </div>
              <button
                type="button"
                onClick={continueAsGuest}
                className="w-full rounded-xl border border-base bg-input px-4 py-2 text-sm font-medium transition hover:brightness-110"
              >
                Kontynuuj jako gość
              </button>
            </div>

            <p className="pt-1 text-center text-[11px] text-muted">
              Logując się akceptujesz regulamin i politykę prywatności.
            </p>
          </form>
        </div>
      </div>
    </div>
  );
}
