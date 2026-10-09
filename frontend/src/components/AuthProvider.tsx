"use client";

import React, { createContext, useCallback, useContext, useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { useChatStore } from "@/store/chatStore";
import type { Session } from "@supabase/supabase-js";
import { api, getMe, supabase, exchangeSession, API_BASE } from "@/lib/api";

export type User = {
    id: number;
    username: string;
    email: string;
    first_name?: string | null;
    last_name?: string | null;
    date_of_birth?: string | null;
    age?: number | null;
    auth_provider?: "local" | "google" | "supabase";
    email_verified?: boolean;
    avatar_url?: string | null;
};

type UpdateProfileData = {
    username?: string;
    first_name?: string;
    last_name?: string;
    date_of_birth?: string; // YYYY-MM-DD
};

type AuthMode = "login" | "register";

type Ctx = {
    user: User | null;
    loading: boolean;
    isGuest: boolean;

    openAuth: (mode?: AuthMode) => void;
    closeAuth: () => void;
    authOpen: boolean;
    authMode: AuthMode;

    login: (email: string, password: string) => Promise<void>;
    register: (
        email: string,
        password: string,
        username?: string,
        first_name?: string,
        last_name?: string,
        date_of_birth?: string
    ) => Promise<void>;
    updateProfile: (data: UpdateProfileData) => Promise<User>;
    logout: () => Promise<void>;
    refreshMe: () => Promise<void>;
    continueAsGuest: () => void;
};

// Per-user local data that must be wiped on logout or account switch
const PERSONAL_LS_KEYS = ["sc_session_id", "sc_prep_state_split_overlay_v1"];

/** Starts a fresh conversation (new session_id) without the previous user's data; ChatUI listens for "sc:logout". */
function resetLocalSession() {
    try {
        useChatStore.getState().clearAll();
    } catch {
        /* ignore */
    }
    for (const k of PERSONAL_LS_KEYS) {
        try {
            localStorage.removeItem(k);
        } catch {}
    }
    try {
        window.dispatchEvent(new CustomEvent("sc:logout"));
    } catch {}
}

// The provider always wraps the app (layout.tsx), so the default value is never used.
const AuthCtx = createContext<Ctx>(null as unknown as Ctx);
export const useAuth = () => useContext(AuthCtx);

export default function AuthProvider({ children }: { children: React.ReactNode }) {
    const router = useRouter();

    const [user, setUser] = useState<User | null>(null);
    const [loading, setLoading] = useState(true);
    const [authOpen, setAuthOpen] = useState(false);
    const [authMode, setAuthMode] = useState<AuthMode>("login");
    const [guest, setGuest] = useState<boolean>(false);

    const refreshing = useRef<Promise<void> | null>(null);
    const loggingOut = useRef<Promise<void> | null>(null);
    // Only the first load shows "loading"; later refreshes (e.g. token renewal) are silent,
    // otherwise the user menu would unmount along with an open settings form
    const loadedOnce = useRef(false);
    // Supabase user id from the last known session (null = no session, undefined = not known yet)
    const lastAuthUid = useRef<string | null | undefined>(undefined);

    // Stable references: AutoOpenAuth lists these as effect deps (a new function per render
    // would reopen the login panel right after it was closed)
    const openAuth = useCallback((mode: AuthMode = "login") => {
        setAuthMode(mode);
        setAuthOpen(true);
    }, []);
    const closeAuth = useCallback(() => setAuthOpen(false), []);

    const _getSessionWithTimeout = async (ms = 4000) => {
        return Promise.race([
            supabase.auth.getSession(),
            new Promise<never>((_, rej) =>
                setTimeout(() => rej(new Error("getSession timeout")), ms)
            ),
        ]).catch(() => ({ data: { session: null as Session | null } }));
    };

    const refreshMe = async () => {
        if (refreshing.current) return refreshing.current;
        const work = (async () => {
            if (!loadedOnce.current) setLoading(true);
            try {
                const { data: sess } = await _getSessionWithTimeout(4000);

                if (!sess.session) {
                    setUser(null);
                    try {
                        setGuest(localStorage.getItem("sc_guest") === "1");
                    } catch {}
                    return;
                }

                try {
                    await exchangeSession();
                } catch {}

                try {
                    const me = await getMe<User>();
                    setUser(me ?? null);
                    setGuest(false);
                    try {
                        localStorage.removeItem("sc_guest");
                    } catch {}
                } catch {
                    setUser(null);
                }
            } finally {
                loadedOnce.current = true;
                setLoading(false);
            }
        })();

        const timeout = new Promise<void>((res) =>
            setTimeout(() => {
                setLoading(false);
                res();
            }, 15_000)
        );

        refreshing.current = Promise.race([work, timeout]).finally(() => {
            refreshing.current = null;
        });

        return refreshing.current;
    };

    useEffect(() => {
        try {
            setGuest(localStorage.getItem("sc_guest") === "1");
        } catch {}
        void refreshMe();
        void _getSessionWithTimeout(4000).then(({ data }) => {
            if (lastAuthUid.current === undefined) lastAuthUid.current = data.session?.user?.id ?? null;
        });

        const { data: sub } = supabase.auth.onAuthStateChange((event, session) => {
            // User changed without our logout() (expired session, logout in another tab, login to
            // another account): start a new conversation so we neither write to the previous user's
            // session (403) nor show their data. We check the Supabase session, not /auth/me, because a
            // transient network error must not wipe the conversation. Guest -> logged in keeps it.
            const uid = session?.user?.id ?? null;
            const prev = lastAuthUid.current;
            if (prev !== undefined && prev !== null && prev !== uid && !loggingOut.current) {
                resetLocalSession();
            }
            lastAuthUid.current = uid;

            if (event === "INITIAL_SESSION") return; // already handled by the mount effect

            // Return synchronously so the Supabase auth lock is released immediately
            setTimeout(() => {
                void (async () => {
                    if (session) {
                        try { await exchangeSession(); } catch {}
                    }
                    await refreshMe();
                })();
            }, 0);
        });

        return () => {
            try {
                sub.subscription.unsubscribe();
            } catch {}
        };
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    useEffect(() => {
        const onUnauthorized = () => openAuth("login");
        window.addEventListener("auth:unauthorized", onUnauthorized as EventListener);
        return () => window.removeEventListener("auth:unauthorized", onUnauthorized as EventListener);
    }, [openAuth]);

    useEffect(() => {
        if (!loading && !user && !guest) openAuth("login");
    }, [loading, user, guest, openAuth]);

    const _withTimeout = <T,>(p: Promise<T>, ms: number, label: string): Promise<T> =>
        Promise.race([
            p,
            new Promise<T>((_, rej) =>
                setTimeout(() => rej(new Error(`${label}: timeout po ${Math.round(ms / 1000)}s`)), ms)
            ),
        ]);

    const login = async (email: string, password: string) => {
        const { error } = await _withTimeout(
            supabase.auth.signInWithPassword({ email, password }),
            12_000,
            "Logowanie do Supabase"
        );
        if (error) throw new Error(error.message);

        try {
            await _withTimeout(exchangeSession(), 8_000, "Wymiana sesji");
        } catch {}

        try {
            localStorage.removeItem("sc_guest");
        } catch {}

        // refreshMe has its own 15s race, but the outer 16s guard covers the case where the
        // onAuthStateChange dedup inherits a stuck refreshing.current.
        try {
            await _withTimeout(refreshMe(), 16_000, "Odświeżanie profilu");
        } catch {
            // Close the panel even if refreshMe hangs; getMe will run on the next action
        }

        closeAuth();
    };

    const register = async (
        email: string,
        password: string,
        username?: string,
        first_name?: string,
        last_name?: string,
        date_of_birth?: string
    ) => {
        const { data, error } = await _withTimeout(
            supabase.auth.signUp({
                email,
                password,
                options: { data: { username, first_name, last_name, date_of_birth } },
            }),
            12_000,
            "Rejestracja"
        );
        if (error) throw new Error(error.message);

        if (!data.session?.access_token) {
            try {
                await _withTimeout(
                    supabase.auth.signInWithPassword({ email, password }),
                    12_000,
                    "Logowanie po rejestracji"
                );
            } catch {}
        }

        try {
            await _withTimeout(exchangeSession(), 8_000, "Wymiana sesji");
        } catch {}

        try {
            await _withTimeout(refreshMe(), 16_000, "Odświeżanie profilu");
        } catch {}

        closeAuth();
    };

    const updateProfile = async (data: UpdateProfileData) => {
        const { data: u } = await api.put<User>("/auth/me", data);
        setUser(u);
        return u;
    };

    const clearAfterLogout = () => {
        try {
            localStorage.removeItem("sc_guest");
        } catch {}
        resetLocalSession();
        lastAuthUid.current = null;   // user change already handled here
    };

    const hardBackendLogout = async () => {
        try {
            await fetch(`${API_BASE}/auth/logout`, {
                method: "POST",
                credentials: "include",
            });
        } catch {
            // Best effort: the dev backend may not respond
        }
    };

    const logout = async () => {
        if (loggingOut.current) return loggingOut.current;
        loggingOut.current = (async () => {
            try {
                // Timeout guards against signOut deadlocking
                await _withTimeout(supabase.auth.signOut(), 8_000, "Wylogowywanie");
            } catch {}

            await hardBackendLogout();

            clearAfterLogout();

            setUser(null);
            setGuest(false);

            try {
                router.push("/");
            } catch {}

            openAuth("login");
        })().finally(() => {
            loggingOut.current = null;
        });

        return loggingOut.current;
    };

    const continueAsGuest = () => {
        try {
            localStorage.setItem("sc_guest", "1");
        } catch {}
        // New session required: clearing only the view kept the old session_id, and the backend restored its symptoms
        resetLocalSession();
        setUser(null);
        setGuest(true);
        closeAuth();
        try {
            window.dispatchEvent(new CustomEvent("sc:guest"));
        } catch {}
    };

    return (
        <AuthCtx.Provider
            value={{
                user,
                loading,
                isGuest: guest && !user,
                openAuth,
                closeAuth,
                authOpen,
                authMode,
                login,
                register,
                updateProfile,
                logout,
                refreshMe,
                continueAsGuest,
            }}
        >
            {children}
        </AuthCtx.Provider>
    );
}
