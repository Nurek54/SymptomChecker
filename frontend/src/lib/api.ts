import axios, { AxiosHeaders, type InternalAxiosRequestConfig } from "axios";
import { emitUnauthorized } from "@/lib/authBus";
import { supabase } from "@/lib/supabaseClient";
import { ChatApiResponse, SymptomDictItem, SymptomsPatchResponse } from "./types";

export { supabase } from "@/lib/supabaseClient";

export const API_BASE =
    process.env.NEXT_PUBLIC_API_BASE_URL ??
    process.env.NEXT_PUBLIC_API_BASE ??
    "http://localhost:8000";

/** Sends the backend session cookie and the Supabase JWT as Bearer. */
export const api = axios.create({
    baseURL: API_BASE,
    withCredentials: true,
});

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

async function getTokenRaw(): Promise<string | null> {
    const { data } = await supabase.auth.getSession();
    return data.session?.access_token ?? null;
}

/** Right after navigation Supabase may briefly report no session, so poll for a short while. */
async function getTokenWait(maxWaitMs = 2000, stepMs = 150): Promise<string | null> {
    const deadline = Date.now() + maxWaitMs;
    let tok = await getTokenRaw();
    while (!tok && Date.now() < deadline) {
        await sleep(stepMs);
        tok = await getTokenRaw();
    }
    return tok;
}

api.interceptors.request.use(
    async (config: InternalAxiosRequestConfig) => {
        const headers = AxiosHeaders.from(config.headers || {});
        const isForm = typeof FormData !== "undefined" && config.data instanceof FormData;
        const method = (config.method || "GET").toUpperCase();

        if (!isForm && method !== "GET" && !headers.has("Content-Type")) {
            headers.set("Content-Type", "application/json");
        }

        const tok = await getTokenRaw();
        if (tok) headers.set("Authorization", `Bearer ${tok}`);

        config.headers = headers;
        return config;
    },
    (error) => Promise.reject(error)
);

let exchangePromise: Promise<boolean> | null = null;

const PREWARM_TTL_MS = 60_000;
let _lastPrewarmOkAt = 0;

/** Exchanges the Supabase JWT for the backend session cookie; deduplicated and cached for PREWARM_TTL_MS. */
export async function ensureBackendSession(force = false): Promise<boolean> {
    if (!force && Date.now() - _lastPrewarmOkAt < PREWARM_TTL_MS) {
        return true;
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

/** On 401, refresh the backend session and retry once. */
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

const DEFAULT_TIMEOUT_MS = 180_000; // long enough for slow LLM generation

function _isMutating(method?: string): boolean {
    const m = (method || "GET").toUpperCase();
    return m === "POST" || m === "PUT" || m === "PATCH" || m === "DELETE";
}

function _isChatEndpoint(path: string): boolean {
    return path === "/chat" || path.startsWith("/chat?");
}

/** Named "AbortError" so callers can recognise cancellations and ignore them. */
function abortError(message: string): Error {
    const err = new Error(message);
    err.name = "AbortError";
    return err;
}

/** Extracts a message from any caught value. */
export function errorMessage(e: unknown): string {
    if (e instanceof Error) return e.message;
    if (typeof e === "string") return e;
    return String((e as { message?: unknown } | null)?.message ?? "");
}

/** fetch wrapper with Bearer + cookies, a timeout and retry on 401 for idempotent methods only. */
export async function apiFetch<T = unknown>(
    path: string,
    init: RequestInit = {},
    opts: { timeoutMs?: number } = {}
): Promise<T> {
    const timeoutMs = opts.timeoutMs ?? DEFAULT_TIMEOUT_MS;
    const mutating = _isMutating(init.method);

    // Prewarm only for POST /chat; the TTL limits real exchanges to one per minute.
    if (mutating && _isChatEndpoint(path)) {
        try {
            await ensureBackendSession();
        } catch {
            // best effort, send the request anyway
        }
    }

    // Combine the caller's signal (if any) with our timeout.
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
            // Distinguish our timeout from a caller abort
            if ((e as { name?: string } | null)?.name === "AbortError") {
                if (localCtrl.signal.aborted && !externalSignal?.aborted) {
                    throw new Error(
                        `API timeout po ${Math.round(timeoutMs / 1000)}s: ${path}. ` +
                        `Backend może wciąż liczyć, odśwież stronę za chwilę.`
                    );
                }
                throw abortError(`Żądanie anulowane: ${path}`);
            }
            throw e;
        }

        // Retry 401 only for idempotent methods: a repeated POST /chat would run
        // two LLM generations in parallel and deadlock the backend.
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

/** Warms the backend session; `force` bypasses the TTL cache (e.g. right after login). */
export async function prewarmAuth(force = false): Promise<boolean> {
    return ensureBackendSession(force);
}

/** Re-warms the session on focus/visibility/online; installs only once per window. */
export function installSessionGuards() {
    if (typeof window === "undefined") return;
    const w = window as Window & { __scGuardsInstalled?: boolean };
    if (w.__scGuardsInstalled) return;
    w.__scGuardsInstalled = true;

    // No force here: the TTL cache prevents spamming /auth/session.
    const warm = () => { void prewarmAuth(false); };

    window.addEventListener("focus", warm);
    window.addEventListener("online", warm);
    document.addEventListener("visibilitychange", () => {
        if (document.visibilityState === "visible") warm();
    });

    void prewarmAuth(true);
}

let _exchangeSessionPromise: Promise<unknown> | null = null;

/** Returns the backend UserOut, or null when there is no Supabase session. */
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
        // A fresh exchange resets the TTL so ensureBackendSession() does not repeat it.
        _lastPrewarmOkAt = Date.now();
        return await r.json();
    })();
    try {
        return await _exchangeSessionPromise;
    } finally {
        // Keep the promise ~500ms so near-simultaneous callers share it
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
    // After a reload this is AuthProvider's first identity check. A 401 from a race with
    // exchangeSession would look like a logout, so retry with a forced prewarm and backoff.
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

/**
 * `clientSymptoms` are the symptoms the frontend remembers. The backend uses them only when
 * it does not know the session (e.g. after a server restart) so the chat does not start over.
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

export function listSymptomsDict(): Promise<SymptomDictItem[]> {
    return apiFetch<SymptomDictItem[]>("/dicts/symptoms");
}

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

/** Per-model top-k (XGBoost, River, ensemble) for the model comparison view. */
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
    // 30s: under load (e.g. during LLM generation) a single slow response must not
    // flip the UI status to an error while the backend is alive.
    return apiFetch<HealthStatus>("/health", { cache: "no-store" }, { timeoutMs: 30_000 });
}

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
    /** null for a conversation with no messages yet */
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
    // After a reload Supabase may still be rehydrating and the backend cookie missing.
    // apiFetch retries once, but a token refresh can take 1-2s, so back off and retry.
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
        if (signal?.aborted) throw abortError("Aborted");
        try {
            return await fn();
        } catch (e: unknown) {
            lastErr = e;
            const msg = errorMessage(e);
            // Only 401 is retried (with a forced prewarm); other errors would not improve.
            const is401 = msg.includes("API 401:");
            if (!is401) throw e;
            if (attempt === maxAttempts - 1) break;
            try {
                await ensureBackendSession(true);
            } catch {}
            // Backoff: 400ms, 1000ms (2.5x per attempt)
            const delay = 400 * Math.pow(2.5, attempt);
            await new Promise(r => setTimeout(r, delay));
        }
    }
    throw lastErr;
}