"use client";

import React from "react";
import Link from "next/link";
import { useAuth } from "./AuthProvider";
import PrepOverlay from "./PrepOverlay"; // ← istniejący overlay 1:1
import HowToOverlay from "./HowToOverlay"; // ← NEW: overlay z instrukcją
import {
  LogOut,
  ScrollText,
  Settings2,
  ChevronLeft,
  ChevronRight,
  User2,
  Mail,
  Calendar,
  PencilLine,
  ClipboardList,
  BookOpenText, // ← NEW ikona dla „Jak używać”
} from "lucide-react";

/* —————————————————— Mini spinner —————————————————— */
function MiniSpinner() {
  return (
    <svg className="h-4 w-4 animate-spin" viewBox="0 0 24 24" aria-hidden="true">
      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z" />
    </svg>
  );
}

export default function UserMenu() {
  const { user, openAuth, logout, loading, updateProfile } = useAuth();
  const [open, setOpen] = React.useState(false);
  const [showSettings, setShowSettings] = React.useState(false);
  const [showPrep, setShowPrep] = React.useState(false); // ← istniejący
  const [showGuide, setShowGuide] = React.useState(false); // ← NEW: overlay „Jak używać”

  // formularz ustawień
  const [username, setUsername] = React.useState(user?.username ?? "");
  const [firstName, setFirstName] = React.useState(user?.first_name ?? "");
  const [lastName, setLastName] = React.useState(user?.last_name ?? "");
  const [dob, setDob] = React.useState(user?.date_of_birth ?? "");
  const [busy, setBusy] = React.useState(false);
  const [err, setErr] = React.useState<string | null>(null);
  const [ok, setOk] = React.useState<string | null>(null);

  /* Sync user fields on change */
  React.useEffect(() => {
    setUsername(user?.username ?? "");
    setFirstName(user?.first_name ?? "");
    setLastName(user?.last_name ?? "");
    setDob(user?.date_of_birth ?? "");
  }, [user]);

  /* Esc zamyka (drawer lub podwidok) */
  React.useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        if (showSettings) setShowSettings(false);
        else setOpen(false);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [showSettings]);

  if (loading && !user) return <span className="text-xs text-muted">…</span>;

  // ——— Niezalogowany ———
  if (!user) {
    return (
      <button
        onClick={() => openAuth("login")}
        className="rounded-xl border border-base bg-card px-3 py-1.5 text-sm shadow-sm transition hover:brightness-110 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-neutral-300"
      >
        Zaloguj się
      </button>
    );
  }

  const initials = user.username?.trim()?.slice(0, 1)?.toUpperCase() || "U";

  const handleLogout = async () => {
    try {
      await logout();
    } finally {
      setOpen(false);
      setShowSettings(false);
    }
  };

  const saveSettings = async (e: React.FormEvent) => {
    e.preventDefault();
    setBusy(true);
    setErr(null);
    setOk(null);
    try {
      await updateProfile({
        username: username || undefined,
        first_name: firstName || undefined,
        last_name: lastName || undefined,
        date_of_birth: dob || undefined,
      });
      setOk("Zapisano zmiany.");
    } catch (e: unknown) {
      setErr((e instanceof Error ? e.message : "") || "Nie udało się zapisać.");
    } finally {
      setBusy(false);
    }
  };

  return (
    <>
      {/* — Trigger — */}
      <button
        onClick={() => setOpen((val) => !val)}
        title={user.username}
        aria-haspopup="dialog"
        aria-expanded={open}
        aria-controls="user-drawer"
        className="group relative inline-flex items-center gap-2 rounded-full border border-base bg-card p-1 sm:pr-3 text-sm shadow-sm transition hover:shadow-md focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-neutral-300"
      >
        {/* delikatny aurora-shine w tle przy hover */}
        <span
          aria-hidden
          className="pointer-events-none absolute inset-0 -z-10 rounded-full opacity-0 transition-opacity duration-300 group-hover:opacity-20"
          style={{
            background:
              "linear-gradient(120deg, rgba(52,211,153,.6), rgba(56,189,248,.6), rgba(167,139,250,.6))",
          }}
        />
        <span className="relative inline-grid place-items-center rounded-full p-0.5">
          <span className="absolute inset-0 -z-10 rounded-full bg-gradient-to-br from-emerald-400/60 via-sky-400/60 to-violet-400/60 opacity-60 transition group-hover:opacity-80" />
          <span className="grid h-8 w-8 place-items-center rounded-full border border-base bg-input text-xs font-semibold">
            {initials}
          </span>
          <span
            className="absolute -right-0.5 -bottom-0.5 h-2.5 w-2.5 rounded-full border border-card bg-emerald-500"
            aria-hidden="true"
          />
        </span>
        <span className="hidden max-w-[140px] truncate sm:inline">{user.username}</span>
      </button>

      {/* — Drawer — */}
      <div
        id="user-drawer"
        className={`fixed right-0 top-0 z-40 h-full w-[360px] max-w-[92vw] transform border-l border-base bg-card/85 backdrop-blur-md shadow-2xl transition-transform ${
          open ? "translate-x-0" : "translate-x-full"
        }`}
        role="dialog"
        aria-modal="true"
        aria-label="Menu użytkownika"
      >
        {/* Header drawer-a */}
        <div className="flex flex-col">
          <div className="flex items-center gap-2 border-b border-base bg-card/80 px-3 py-2">
            {showSettings ? (
              <button
                onClick={() => setShowSettings(false)}
                className="rounded-lg border border-base bg-input p-1.5 text-xs transition hover:brightness-110 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-neutral-300"
                title="Wróć"
              >
                <ChevronLeft className="h-4 w-4" />
              </button>
            ) : null}
            <div className="text-sm font-semibold">
              {showSettings ? "Ustawienia konta" : "Twoje konto"}
            </div>
            <div className="ml-auto text-xs text-muted">
              {user.age != null ? `Wiek: ${user.age} lat` : ""}
            </div>
          </div>

          {/* Aurora accent bar */}
          <div className="relative h-[3px] w-full overflow-hidden">
            <div
              className="h-full w-full"
              style={{
                background:
                  "linear-gradient(90deg, #34d399, #38bdf8 55%, #a78bfa)",
              }}
            />
            <div
              aria-hidden
              className="pointer-events-none absolute left-0 top-0 h-full w-1/3 opacity-30"
              style={{
                background:
                  "linear-gradient(90deg, transparent, white, transparent)",
                animation: "auroraSlide 2.4s linear infinite",
              }}
            />
          </div>
        </div>

        {/* Treść */}
        {!showSettings ? (
          <div className="space-y-3 p-3">
            {/* Karta użytkownika */}
            <div className="relative overflow-hidden rounded-2xl border border-base bg-card">
              <div className="absolute -left-10 -top-10 h-28 w-28 rounded-full bg-gradient-to-br from-sky-400/20 to-violet-400/20" />
              <div className="absolute -right-10 -bottom-10 h-28 w-28 rounded-full bg-gradient-to-br from-emerald-400/20 to-teal-400/20" />
              <div className="relative flex items-center gap-3 p-3">
                <div className="grid h-12 w-12 place-items-center rounded-full border border-base bg-input text-sm font-semibold">
                  {initials}
                </div>
                <div className="min-w-0">
                  <div className="flex items-center gap-1 text-sm font-medium">
                    <User2 className="h-4 w-4 opacity-70" />
                    <span className="truncate">{user.username}</span>
                  </div>
                  <div className="mt-0.5 flex items-center gap-1 text-xs text-muted">
                    <Mail className="h-3.5 w-3.5 opacity-70" />
                    <span className="truncate">{user.email}</span>
                  </div>
                  {user.first_name || user.last_name ? (
                    <div className="mt-0.5 flex items-center gap-1 text-xs text-muted">
                      <PencilLine className="h-3.5 w-3.5 opacity-70" />
                      <span className="truncate">
                        {(user.first_name ?? "")} {(user.last_name ?? "")}
                      </span>
                    </div>
                  ) : null}
                  {user.date_of_birth ? (
                    <div className="mt-0.5 flex items-center gap-1 text-xs text-muted">
                      <Calendar className="h-3.5 w-3.5 opacity-70" />
                      <span className="truncate">{user.date_of_birth}</span>
                    </div>
                  ) : null}
                </div>
              </div>
            </div>

            {/* Akcje */}
            <nav aria-label="Akcje konta" className="grid gap-2">
              {/* — Przygotuj opis (istniejące) — */}
              <button
                onClick={() => {
                  setShowPrep(true);
                  setOpen(false);
                }}
                className="group relative flex items-center justify-between rounded-xl border border-base bg-card px-3 py-2 text-sm shadow-sm transition hover:bg-input focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-neutral-300"
                aria-haspopup="dialog"
              >
                <span
                  aria-hidden
                  className="pointer-events-none absolute left-0 top-1/2 h-[60%] w-[3px] -translate-y-1/2 rounded-full opacity-0 transition-opacity group-hover:opacity-100"
                  style={{ background: "linear-gradient(180deg, #34d399, #38bdf8, #a78bfa)" }}
                />
                <span className="inline-flex items-center gap-2">
                  <ClipboardList className="h-4 w-4 opacity-80" />
                  Przygotuj opis
                </span>
                <ChevronRight className="h-4 w-4 opacity-0 transition group-hover:opacity-60" />
              </button>

              {/* — NEW: Jak używać — */}
              <button
                onClick={() => {
                  setShowGuide(true);
                  setOpen(false); // domknij drawer dla czystego widoku
                }}
                className="group relative flex items-center justify-between rounded-xl border border-base bg-card px-3 py-2 text-sm shadow-sm transition hover:bg-input focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-neutral-300"
                aria-haspopup="dialog"
              >
                <span
                  aria-hidden
                  className="pointer-events-none absolute left-0 top-1/2 h-[60%] w-[3px] -translate-y-1/2 rounded-full opacity-0 transition-opacity group-hover:opacity-100"
                  style={{ background: "linear-gradient(180deg, #34d399, #38bdf8, #a78bfa)" }}
                />
                <span className="inline-flex items-center gap-2">
                  <BookOpenText className="h-4 w-4 opacity-80" />
                  Jak używać
                </span>
                <ChevronRight className="h-4 w-4 opacity-0 transition group-hover:opacity-60" />
              </button>

              <Link
                href="/history"
                onClick={() => setOpen(false)}
                className="group relative flex items-center justify-between rounded-xl border border-base bg-card px-3 py-2 text-sm shadow-sm transition hover:bg-input focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-neutral-300"
              >
                <span
                  aria-hidden
                  className="pointer-events-none absolute left-0 top-1/2 h-[60%] w-[3px] -translate-y-1/2 rounded-full opacity-0 transition-opacity group-hover:opacity-100"
                  style={{ background: "linear-gradient(180deg, #34d399, #38bdf8, #a78bfa)" }}
                />
                <span className="inline-flex items-center gap-2">
                  <ScrollText className="h-4 w-4 opacity-80" />
                  Historia czatów
                </span>
                <ChevronRight className="h-4 w-4 opacity-0 transition group-hover:opacity-60" />
              </Link>

              <button
                onClick={() => setShowSettings(true)}
                className="group relative flex items-center justify-between rounded-xl border border-base bg-card px-3 py-2 text-sm shadow-sm transition hover:bg-input focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-neutral-300"
              >
                <span
                  aria-hidden
                  className="pointer-events-none absolute left-0 top-1/2 h-[60%] w-[3px] -translate-y-1/2 rounded-full opacity-0 transition-opacity group-hover:opacity-100"
                  style={{ background: "linear-gradient(180deg, #34d399, #38bdf8, #a78bfa)" }}
                />
                <span className="inline-flex items-center gap-2">
                  <Settings2 className="h-4 w-4 opacity-80" />
                  Ustawienia konta
                </span>
                <ChevronRight className="h-4 w-4 opacity-0 transition group-hover:opacity-60" />
              </button>

              <button
                onClick={handleLogout}
                className="group relative flex items-center justify-between rounded-xl border border-base bg-card px-3 py-2 text-sm text-rose-600 shadow-sm transition hover:bg-input focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-neutral-300"
              >
                <span
                  aria-hidden
                  className="pointer-events-none absolute left-0 top-1/2 h-[60%] w-[3px] -translate-y-1/2 rounded-full opacity-0 transition-opacity group-hover:opacity-100"
                  style={{ background: "linear-gradient(180deg, #34d399, #38bdf8, #a78bfa)" }}
                />
                <span className="inline-flex items-center gap-2">
                  <LogOut className="h-4 w-4" />
                  Wyloguj
                </span>
              </button>
            </nav>
          </div>
        ) : (
          /* — Ustawienia — */
          <form onSubmit={saveSettings} className="flex h-[calc(100%-41px)] flex-col gap-3 p-3">
            {err && (
              <div
                className="rounded-xl border border-rose-200/60 bg-rose-50/70 px-3 py-2 text-xs text-rose-700 dark:border-rose-400/30 dark:bg-rose-500/10 dark:text-rose-300"
                role="alert"
              >
                {err}
              </div>
            )}
            {ok && (
              <div
                className="rounded-xl border border-emerald-200/60 bg-emerald-50/70 px-3 py-2 text-xs text-emerald-700 dark:border-emerald-400/30 dark:bg-emerald-500/10 dark:text-emerald-300"
                role="status"
                aria-live="polite"
              >
                {ok}
              </div>
            )}

            <div className="space-y-1.5">
              <label className="text-xs text-muted">Nazwa użytkownika</label>
              <div className="flex items-center gap-2 rounded-lg border border-base bg-input px-2">
                <User2 className="h-4 w-4 opacity-70" />
                <input
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  required
                  className="h-9 flex-1 bg-transparent text-sm outline-none"
                />
              </div>
            </div>

            <div className="grid grid-cols-2 gap-2">
              <div className="space-y-1.5">
                <label className="text-xs text-muted">Imię</label>
                <div className="flex items-center gap-2 rounded-lg border border-base bg-input px-2">
                  <PencilLine className="h-4 w-4 opacity-70" />
                  <input
                    value={firstName}
                    onChange={(e) => setFirstName(e.target.value)}
                    className="h-9 flex-1 bg-transparent text-sm outline-none"
                  />
                </div>
              </div>
              <div className="space-y-1.5">
                <label className="text-xs text-muted">Nazwisko</label>
                <div className="flex items-center gap-2 rounded-lg border border-base bg-input px-2">
                  <PencilLine className="h-4 w-4 opacity-70" />
                  <input
                    value={lastName}
                    onChange={(e) => setLastName(e.target.value)}
                    className="h-9 flex-1 bg-transparent text-sm outline-none"
                  />
                </div>
              </div>
            </div>

            <div className="space-y-1.5">
              <label className="text-xs text-muted">Data urodzenia</label>
              <div className="flex items-center gap-2 rounded-lg border border-base bg-input px-2">
                <Calendar className="h-4 w-4 opacity-70" />
                <input
                  type="date"
                  value={dob ?? ""}
                  onChange={(e) => setDob(e.target.value)}
                  className="h-9 flex-1 bg-transparent text-sm outline-none"
                />
              </div>
            </div>

            <div className="mt-auto flex items-center gap-2">
              <button
                type="button"
                onClick={() => {
                  setShowSettings(false);
                  setErr(null);
                  setOk(null);
                }}
                className="rounded-xl border border-base bg-card px-4 py-2 text-sm shadow-sm transition hover:brightness-110 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-neutral-300"
              >
                Anuluj
              </button>
              <button
                disabled={busy}
                className="btn-primary inline-flex items-center gap-2 rounded-xl px-4 py-2 text-sm font-medium shadow-sm transition hover:shadow-md focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-neutral-300 disabled:opacity-60"
              >
                {busy ? <MiniSpinner /> : null}
                Zapisz
              </button>
            </div>
          </form>
        )}
      </div>

      {/* Backdrop: bez blur (tylko przyciemnienie) */}
      <div
        onClick={() => {
          setOpen(false);
          setShowSettings(false);
        }}
        className={`fixed inset-0 z-30 bg-black/30 transition-opacity ${
          open ? "opacity-100" : "pointer-events-none opacity-0"
        }`}
        aria-hidden="true"
      />

      {/* — Overlays — */}
      <PrepOverlay open={showPrep} onClose={() => setShowPrep(false)} />
      <HowToOverlay open={showGuide} onClose={() => setShowGuide(false)} />

      {/* Keyframes do aurora accent bar (lokalne dla komponentu) */}
      <style jsx>{`
        @keyframes auroraSlide {
          0% { transform: translateX(-100%); }
          100% { transform: translateX(100%); }
        }
      `}</style>
    </>
  );
}
