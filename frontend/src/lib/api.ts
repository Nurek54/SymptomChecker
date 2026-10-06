import axios, { AxiosHeaders, type InternalAxiosRequestConfig } from "axios";
import { emitUnauthorized } from "@/lib/authBus";
import { supabase } from "@/lib/supabaseClient";
import { ChatApiResponse, SymptomDictItem, SymptomsPatchResponse } from "./types";

export { supabase } from "@/lib/supabaseClient";

/** Podstawa URL API (FastAPI) */
export const API_BASE =
    process.env.NEXT_PUBLIC_API_BASE_URL ??
    process.env.NEXT_PUBLIC_API_BASE ??
    "http://localhost:8000";

/** axios – z cookies + Bearer (Supabase JWT) */
export const api = axios.create({
    baseURL: API_BASE,
    withCredentials: true,
});

/* ───────────────────────── helpers: token & retry ───────────────────────── */

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

/** Szybkie pobranie tokenu (bez czekania) */
async function getTokenRaw(): Promise<string | null> {
    const { data } = await supabase.auth.getSession();
    return data.session?.access_token ?? null;
}

/** Po nawigacji Supabase może chwilę „nie mieć” sesji → krótki wait z retry */
async function getTokenWait(maxWaitMs = 2000, stepMs = 150): Promise<string | null> {
    const deadline = Date.now() + maxWaitMs;
    let tok = await getTokenRaw();
    while (!tok && Date.now() < deadline) {
        await sleep(stepMs);
        tok = await getTokenRaw();
    }
    return tok;
}

/** Interceptor requestów: Content-Type + Bearer */
api.interceptors.request.use(
    async (config: InternalAxiosRequestConfig) => {
        const headers = AxiosHeaders.from(config.headers || {});
        const isForm = typeof FormData !== "undefined" && config.data instanceof FormData;
        const method = (config.method || "GET").toUpperCase();

        if (!isForm && method !== "GET" && !headers.has("Content-Type")) {
            headers.set("Content-Type", "application/json");
        }

        const tok = await getTokenRaw(); // szybki — bez czekania
        if (tok) headers.set("Authorization", `Bearer ${tok}`);

        config.headers = headers;
        return config;
    },
    (error) => Promise.reject(error)
);

let exchangePromise: Promise<boolean> | null = null;

const PREWARM_TTL_MS = 60_000;
let _lastPrewarmOkAt = 0;

/** Utrzymanie backendowego cookie sesji (export – do pre-warmu) */
export async function ensureBackendSession(force = false): Promise<boolean> {
    if (!force && Date.now() - _lastPrewarmOkAt < PREWARM_TTL_MS) {
        return true; // świeże, nie ma sensu znowu pytać
    }
    if (exchangePromise) return exchangePromise;
    exchangePromise = (async () => {
        const access = await getTokenWait(2000);
        if (!access) return false;
        try {
            const r = await fetch(`${API_BASE}/auth/session`, {
                method: "POST",
                headers: { Authorization: `Bearer ${access}` },
                credentials: "include",
            });
            if (r.ok) {
                _lastPrewarmOkAt = Date.now();
            }
            return r.ok;
        } catch {
            return false;
        }
    })();
    try {
        return await exchangePromise;
    } finally {
        exchangePromise = null;
    }
}

/** Interceptor odpowiedzi: 401 → /auth/session + retry 1× */
api.interceptors.response.use(
    (r) => r,
    async (err) => {
        const status = err?.response?.status;
        const cfg = err?.config as (InternalAxiosRequestConfig & { _retried?: boolean }) | undefined;

        if (status === 401 && cfg && !cfg._retried) {
            const ok = await ensureBackendSession();
            if (ok) {
                cfg._retried = true;
                return api(cfg);
            }
        }

        if (status === 401) emitUnauthorized();
        return Promise.reject(err);
    }
);

/** Zbuduj nagłówki dla fetch (Headers akceptuje każdy HeadersInit) */
async function buildHeaders(init?: RequestInit): Promise<Headers> {
    const h = new Headers(init?.headers);
    const method = (init?.method || "GET").toUpperCase();
    const isForm = init?.body instanceof FormData;

    if (!isForm && method !== "GET" && !h.has("Content-Type")) {
        h.set("Content-Type", "application/json");
    }

    const tok = await getTokenRaw();
    if (tok) h.set("Authorization", `Bearer ${tok}`);

    return h;
}

const DEFAULT_TIMEOUT_MS = 180_000; // 3 min — wystarczy na długą generację LLM

function _isMutating(method?: string): boolean {
    const m = (method || "GET").toUpperCase();
    return m === "POST" || m === "PUT" || m === "PATCH" || m === "DELETE";
}

function _isChatEndpoint(path: string): boolean {
    return path === "/chat" || path.startsWith("/chat?");
}

/** Ujednolicony wrapper na fetch (Bearer + COOKIES) z timeoutem i selektywnym retry. */
/** Wyciąga komunikat z dowolnego błędu (catch daje `unknown`). */
export function errorMessage(e: unknown): string {
    if (e instanceof Error) return e.message;
    if (typeof e === "string") return e;
    return String((e as { message?: unknown } | null)?.message ?? "");
}

export async function apiFetch<T = unknown>(
    path: string,
    init: RequestInit = {},
    opts: { timeoutMs?: number } = {}
): Promise<T> {
    const timeoutMs = opts.timeoutMs ?? DEFAULT_TIMEOUT_MS;
    const mutating = _isMutating(init.method);

    // v9: prewarm TYLKO dla /chat (nie dla każdego POST). TTL=60s i tak
    // sprawi że faktyczny request idzie najwyżej raz na minutę.
    if (mutating && _isChatEndpoint(path)) {
        try {
            await ensureBackendSession();
        } catch {
            // ignorujemy — i tak spróbujemy
        }
    }

    // Łączymy zewnętrzny signal (jeśli jest) z naszym timeoutem.
    const localCtrl = new AbortController();
    const timeoutId = setTimeout(() => localCtrl.abort(), timeoutMs);

    const externalSignal = init.signal;
    const onExternalAbort = () => localCtrl.abort();
    if (externalSignal) {
        if (externalSignal.aborted) localCtrl.abort();
        else externalSignal.addEventListener("abort", onExternalAbort);
    }

    const cleanup = () => {
        clearTimeout(timeoutId);
        externalSignal?.removeEventListener("abort", onExternalAbort);
    };

    try {
        const headers = await buildHeaders(init);
        const doFetch = () =>
            fetch(`${API_BASE}${path}`, {
                ...init,
                headers,
                credentials: "include",
                signal: localCtrl.signal,
            });

        let res: Response;
        try {
            res = await doFetch();
        } catch (e: unknown) {
            // Rozróżniamy timeout od innych błędów sieciowych
            if ((e as { name?: string } | null)?.name === "AbortError") {
                if (localCtrl.signal.aborted && !externalSignal?.aborted) {
                    throw new Error(
                        `API timeout po ${Math.round(timeoutMs / 1000)}s: ${path}. ` +
                        `Backend może wciąż liczyć — odśwież stronę za chwilę.`
                    );
                }
                throw new Error(`Żądanie anulowane: ${path}`);
            }
            throw e;
        }

        // Retry 401 TYLKO dla idempotentnych metod (GET/HEAD).
        // Dla POST /chat dwa requesty = dwa wywołania LLM równolegle = deadlock.
        if (res.status === 401 && !mutating) {
            const ok = await ensureBackendSession();
            if (ok) res = await doFetch();
        }

        const text = await res.text().catch(() => "");
        let data: unknown = null;
        try {
            data = text ? JSON.parse(text) : null;
        } catch {
            data = text || null;
        }

        if (!res.ok) {
            if (res.status === 401) emitUnauthorized();
            const obj = data && typeof data === "object" ? (data as { detail?: unknown; message?: unknown }) : null;
            const detail =
                (obj && (obj.detail || obj.message)) ||
                (typeof data === "string" ? data : "") ||
                res.statusText;
            throw new Error(`API ${res.status}: ${detail}`);
        }

        if (res.status === 204) return {} as T;
        return (data ?? {}) as T;
    } finally {
        cleanup();
    }
}

/* ─────────── PRE-WARM: publiczne API do „dogrzewania” sesji ─────────── */

/**
 * Publiczny pre-warm (używany przez strażników/klient-bootstrap).
 * v9: `force=true` omija TTL cache (useful po świeżym logowaniu).
 */
export async function prewarmAuth(force = false): Promise<boolean> {
    return ensureBackendSession(force);
}

/** Jednorazowa instalacja strażników focus/visibility/online */
export function installSessionGuards() {
    if (typeof window === "undefined") return;
    // uniknij wielokrotnej instalacji
    const w = window as Window & { __scGuardsInstalled?: boolean };
    if (w.__scGuardsInstalled) return;
    w.__scGuardsInstalled = true;

    // v9: bez force — TTL cache zadba, żeby nie spamić.
    const warm = () => { void prewarmAuth(false); };

    window.addEventListener("focus", warm);
    window.addEventListener("online", warm);
    document.addEventListener("visibilitychange", () => {
        if (document.visibilityState === "visible") warm();
    });

    // szybkie dogrzanie po załadowaniu — force=true, bo to pierwszy raz
    void prewarmAuth(true);
}

/* =======================
   ======== AUTH =========
   ======================= */

let _exchangeSessionPromise: Promise<unknown> | null = null;

/** Zwraca UserOut z backendu albo null (brak sesji Supabase). */
export async function exchangeSession(): Promise<unknown> {
    if (_exchangeSessionPromise) return _exchangeSessionPromise;
    _exchangeSessionPromise = (async () => {
        const access = await getTokenWait(2000);
        if (!access) return null;

        const r = await fetch(`${API_BASE}/auth/session`, {
            method: "POST",
            headers: { Authorization: `Bearer ${access}` },
            credentials: "include",
        });
        if (!r.ok) throw new Error(await r.text());
        // v9: świeży exchange = reset TTL, żeby kolejne ensureBackendSession()
        // nie strzelało następnego /auth/session bez powodu.
        _lastPrewarmOkAt = Date.now();
        return await r.json(); // UserOut
    })();
    try {
        return await _exchangeSessionPromise;
    } finally {
        // krótkie okno deduplikacji (~500ms) na dogonienie pozostałych wywołań
        setTimeout(() => { _exchangeSessionPromise = null; }, 500);
    }
}

export async function login(email: string, password: string): Promise<void> {
    const { error } = await supabase.auth.signInWithPassword({ email, password });
    if (error) throw new Error(error.message);
    await exchangeSession();
}

export async function register(
    username: string,
    email: string,
    password: string,
    first_name?: string,
    last_name?: string,
    date_of_birth?: string, // "YYYY-MM-DD"
): Promise<void> {
    const { data, error } = await supabase.auth.signUp({
        email,
        password,
        options: { data: { username, first_name, last_name, date_of_birth } },
    });
    if (error) throw new Error(error.message);

    if (!data.session?.access_token) {
        await supabase.auth.signInWithPassword({ email, password }).catch(() => {});
    }
    await exchangeSession();
}

export async function logout(): Promise<void> {
    await supabase.auth.signOut();
    await fetch(`${API_BASE}/auth/logout`, { method: "POST", credentials: "include" }).catch(() => {});
}

export async function getMe<T = unknown>(): Promise<T | null> {
    // v10: po Ctrl+R AuthProvider robi tu pierwsze sprawdzenie kim jest user.
    // Jeśli to padnie (401, timing race z exchangeSession), AuthProvider
    // myśli że user nie jest zalogowany — nawigacja idzie na login, wszystko
    // wygląda jakby "backend odpadł". Retry z prewarm naprawia to.
    for (let attempt = 0; attempt < 3; attempt++) {
        try {
            const { data } = await api.get<T>("/auth/me");
            return (data as T) ?? null;
        } catch (e: unknown) {
            const status = axios.isAxiosError(e) ? e.response?.status : undefined;
            if (status === 401 && attempt < 2) {
                try { await ensureBackendSession(true); } catch {}
                await new Promise(r => setTimeout(r, 400 * Math.pow(2, attempt)));
                continue;
            }
            return null;
        }
    }
    return null;
}

export async function me<T = unknown>(): Promise<T> {
    const { data } = await api.get<T>("/auth/me");
    return data;
}

/* =======================
   ========= CHAT ========
   ======================= */
/**
 * `clientSymptoms` = objawy, które pamięta frontend. Backend używa ich tylko wtedy,
 * gdy nie zna sesji (np. po restarcie serwera), żeby rozmowa nie zaczynała się od zera.
 */
export async function sendChat(
    user_input: string,
    session_id: string,
    signal?: AbortSignal,
    clientSymptoms?: string[]
): Promise<ChatApiResponse> {
    type RawChat = Partial<ChatApiResponse> & { content?: string };
    const raw = await apiFetch<RawChat | null>("/chat", {
        method: "POST",
        body: JSON.stringify({ user_input, session_id, client_symptoms: clientSymptoms ?? null }),
        signal,
    } as RequestInit);
    return {
        response: raw?.response ?? raw?.content ?? "",
        symptoms: Array.isArray(raw?.symptoms) ? raw.symptoms : [],
        top_diagnoses: Array.isArray(raw?.top_diagnoses) ? raw.top_diagnoses : null,
        diagnosis_en: raw?.diagnosis_en ?? null,
        diagnosis_pl: raw?.diagnosis_pl ?? null,
    };
}

/* =======================
   ======== DICTS ========
   ======================= */
export function listSymptomsDict(): Promise<SymptomDictItem[]> {
    return apiFetch<SymptomDictItem[]>("/dicts/symptoms");
}

/* =======================
   ===== SESSION OPS =====
   ======================= */
export function patchSessionSymptoms(
    sessionId: string,
    payload: { add?: string[]; remove?: string[]; replace?: string[]; lang?: "pl" | "en"; current?: string[] },
    signal?: AbortSignal
): Promise<SymptomsPatchResponse> {
    return apiFetch<SymptomsPatchResponse>(`/sessions/${encodeURIComponent(sessionId)}/symptoms`, {
        method: "POST",
        body: JSON.stringify(payload ?? {}),
        signal,
    } as RequestInit);
}

/* =======================
   ======  TOP3 SPLIT ====
   ======================= */
export function getTop3Split(
    sessionId: string,
    k = 3,
    signal?: AbortSignal
): Promise<{
    xgb_enabled: boolean;
    river_enabled: boolean;
    symptoms: string[];
    xgb_top: { diagnosis_en: string; diagnosis_pl?: string; probability: number }[];
    river_top: { diagnosis_en: string; diagnosis_pl?: string; probability: number }[];
    ensemble_top: { diagnosis_en: string; diagnosis_pl?: string; probability: number }[];
    meta: { conf_x?: number; conf_r?: number; wx_dyn?: number; wr_dyn?: number; n_active?: number };
}> {
    return apiFetch(`/sessions/${encodeURIComponent(sessionId)}/top3-split?k=${k}`, { signal } as RequestInit);
}

/* =======================
   ========= HEALTH ======
   ======================= */
export async function healthcheck(): Promise<boolean> {
    const ctrl = new AbortController();
    const t = setTimeout(() => ctrl.abort(), 5_000);
    try {
        const res = await fetch(`${API_BASE}/health`, {
            cache: "no-store",
            credentials: "include",
            signal: ctrl.signal,
        });
        return res.ok;
    } catch {
        return false;
    } finally {
        clearTimeout(t);
    }
}

export type HealthStatus = { xgb_ready?: boolean; matcher_ready?: boolean; llm_ready?: boolean };

export function getHealthDetailed(): Promise<HealthStatus> {
    // v10: 10s → 30s. Pod load'em (np. w trakcie generacji LLM) Supabase
    // getTokenRaw() + buildHeaders bywa wolniejsze, a 10s wystarczało żeby
    // pojedyncze opóźnienie wystrzeliło status "Błąd" w UI, mimo że backend
    // żyje. Status w UI ma być sygnałem realnego problemu, nie spike'ów.
    return apiFetch<HealthStatus>("/health", { cache: "no-store" }, { timeoutMs: 30_000 });
}

/* =======================
   ========= HISTORY =====
   ======================= */
export type HistorySymptom = {
    text: string;
    canonical?: string | null;
    lang?: string | null;
    score?: number | null;
};

export type HistoryDiagnosis = {
    label: string;
    score?: number | null;
    rank?: number | null;
};

export type HistoryItem = {
    id: number;
    title: string;
    /** może być null dla świeżych rozmów */
    last_message_at: string | null;
    message_count: number;
};

export type HistoryDetail = {
    id: number;
    title: string;
    messages: { role: "user" | "assistant"; content: string; created_at: string }[];
    symptoms: HistorySymptom[];
    diagnoses: HistoryDiagnosis[];
};

export function listHistory(signal?: AbortSignal): Promise<HistoryItem[]> {
    // v10: history ma być ZAWSZE dostępna — opakowane w retry z backoffem.
    // Po Ctrl+R apiFetch potrafi trafić moment, gdy Supabase rehydratuje sesję
    // z localStorage i backend nie ma jeszcze cookie. Domyślny retry-on-401
    // w apiFetch obsługuje to dla pojedynczego wywołania, ale jeśli sam
    // ensureBackendSession() potrzebuje 1-2s na refresh tokenu Supabase,
    // wciąż możemy dostać 401. Retry z backoffem pokrywa ten przypadek.
    return _withAuthRetry(
        () => apiFetch<HistoryItem[]>("/history", { signal } as RequestInit, { timeoutMs: 15_000 }),
        signal
    );
}

export function getConversation(convId: number, signal?: AbortSignal): Promise<HistoryDetail> {
    return _withAuthRetry(
        () => apiFetch<HistoryDetail>(`/history/${convId}`, { signal } as RequestInit, { timeoutMs: 15_000 }),
        signal
    );
}

export async function deleteConversation(convId: number): Promise<void> {
    await apiFetch(`/history/${convId}`, { method: "DELETE" });
}

async function _withAuthRetry<T>(
    fn: () => Promise<T>,
    signal?: AbortSignal,
    maxAttempts = 3
): Promise<T> {
    let lastErr: unknown = null;
    for (let attempt = 0; attempt < maxAttempts; attempt++) {
        if (signal?.aborted) throw new Error("Aborted");
        try {
            return await fn();
        } catch (e: unknown) {
            lastErr = e;
            const msg = errorMessage(e);
            // Tylko 401 retryujemy z wymuszonym prewarm. 4xx (inne) i 5xx —
            // przekazujemy wyżej bez retry (powtarzanie i tak nie pomoże).
            const is401 = msg.includes("API 401:");
            if (!is401) throw e;
            if (attempt === maxAttempts - 1) break;
            // Wymuś świeży exchange sesji — może Supabase już zdążył odświeżyć token
            try {
                await ensureBackendSession(true);
            } catch {}
            // Backoff: 400ms, 1200ms, 2800ms
            const delay = 400 * Math.pow(2.5, attempt);
            await new Promise(r => setTimeout(r, delay));
        }
    }
    throw lastErr;
}