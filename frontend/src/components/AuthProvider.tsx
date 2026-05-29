"use client";

import React, { createContext, useContext, useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { useChatStore } from "@/store/chatStore";
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

const AuthCtx = createContext<Ctx>(null as any);
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

    const openAuth = (mode: AuthMode = "login") => {
        setAuthMode(mode);
        setAuthOpen(true);
    };
    const closeAuth = () => setAuthOpen(false);

    const _getSessionWithTimeout = async (ms = 4000) => {
        return Promise.race([
            supabase.auth.getSession(),
            new Promise<never>((_, rej) =>
                setTimeout(() => rej(new Error("getSession timeout")), ms)
            ),
        ]).catch(() => ({ data: { session: null } as any }));
    };

    const refreshMe = async () => {
        if (refreshing.current) return refreshing.current;
        const work = (async () => {
            setLoading(true);
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

    // Bootstrap + Supabase events
    useEffect(() => {
        try {
            setGuest(localStorage.getItem("sc_guest") === "1");
        } catch {}
        void refreshMe();

        const { data: sub } = supabase.auth.onAuthStateChange((event, session) => {
            if (event === "INITIAL_SESSION") return; // mount-effect już to obsłużył

            // synchroniczny return — lock zwolniony natychmiast
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

    // 401 → pokaż panel
    useEffect(() => {
        const onUnauthorized = () => openAuth("login");
        window.addEventListener("auth:unauthorized", onUnauthorized as EventListener);
        return () => window.removeEventListener("auth:unauthorized", onUnauthorized as EventListener);
    }, []);

    // auto-open auth, jeśli brak sesji i nie wybrano gościa
    useEffect(() => {
        if (!loading && !user && !guest) openAuth("login");
    }, [loading, user, guest]);

    const _withTimeout = <T,>(p: Promise<T>, ms: number, label: string): Promise<T> =>
        Promise.race([
            p,
            new Promise<T>((_, rej) =>
                setTimeout(() => rej(new Error(`${label}: timeout po ${Math.round(ms / 1000)}s`)), ms)
            ),
        ]);

    const login = async (email: string, password: string) => {
        // Supabase signInWithPassword: max 12s
        const { error } = await _withTimeout(
            supabase.auth.signInWithPassword({ email, password }),
            12_000,
            "Logowanie do Supabase"
        );
        if (error) throw new Error(error.message);

        // exchangeSession: max 8s
        try {
            await _withTimeout(exchangeSession(), 8_000, "Wymiana sesji");
        } catch {}

        try {
            localStorage.removeItem("sc_guest");
        } catch {}

        // refreshMe ma już swój wewnętrzny race(15s) — ale daję dodatkową
        // ochronę 16s na wypadek, gdyby dedup z onAuthStateChange odziedziczył
        // utknięte refreshing.current.
        try {
            await _withTimeout(refreshMe(), 16_000, "Odświeżanie profilu");
        } catch {
            // nawet jeśli refreshMe wisi — zamykamy panel; user zaloguje się,
            // a getMe odpali się przy następnej akcji
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

    const clearChatUI = () => {
        try {
            useChatStore.getState().clearAll();
        } catch {
            /* ignore */
        }
    };

    const clearAfterLogout = () => {
        clearChatUI();

        try {
            localStorage.removeItem("sc_session_id");
        } catch {}

        try {
            localStorage.removeItem("sc_guest");
        } catch {}

        try {
            window.dispatchEvent(new CustomEvent("sc:logout"));
        } catch {}
    };

    const hardBackendLogout = async () => {
        try {
            await fetch(`${API_BASE}/auth/logout`, {
                method: "POST",
                credentials: "include",
            });
        } catch {
            // dev backend może nie odpowiedzieć — ok
        }
    };

    const logout = async () => {
        if (loggingOut.current) return loggingOut.current;
        loggingOut.current = (async () => {
            try {
                // v10c: timeout 8s — signOut też może paść w deadlock teoretycznie
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
        clearChatUI();
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
