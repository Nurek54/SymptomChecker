"use client";

import React, { createContext, useContext, useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { useChatStore } from "@/store/chatStore";
import { api, getMe, supabase, exchangeSession } from "@/lib/api";

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

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

export default function AuthProvider({ children }: { children: React.ReactNode }) {
    const router = useRouter();

    const [user, setUser] = useState<User | null>(null);
    const [loading, setLoading] = useState(true);
    const [authOpen, setAuthOpen] = useState(false);
    const [authMode, setAuthMode] = useState<AuthMode>("login");
    const [guest, setGuest] = useState<boolean>(false);

    // re-entrancy guards
    const refreshing = useRef<Promise<void> | null>(null);
    const loggingOut = useRef<Promise<void> | null>(null);

    const openAuth = (mode: AuthMode = "login") => {
        setAuthMode(mode);
        setAuthOpen(true);
    };
    const closeAuth = () => setAuthOpen(false);

    const refreshMe = async () => {
        if (refreshing.current) return refreshing.current;
        refreshing.current = (async () => {
            setLoading(true);
            try {
                const { data: sess } = await supabase.auth.getSession();

                if (!sess.session) {
                    setUser(null);
                    try {
                        setGuest(localStorage.getItem("sc_guest") === "1");
                    } catch {}
                    return;
                }

                try {
                    await exchangeSession(); // cookie backendu
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
                refreshing.current = null;
            }
        })();
        return refreshing.current;
    };

    // Bootstrap + Supabase events
    useEffect(() => {
        try {
            setGuest(localStorage.getItem("sc_guest") === "1");
        } catch {}
        void refreshMe();

        const { data: sub } = supabase.auth.onAuthStateChange(async (_event, session) => {
            if (session) {
                try {
                    await exchangeSession();
                } catch {}
            }
            await refreshMe();
        });

        return () => {
            try {
                sub.subscription.unsubscribe();
            } catch {}
        };
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    // 401 z backendu → pokaż panel
    useEffect(() => {
        const onUnauthorized = () => openAuth("login");
        window.addEventListener("auth:unauthorized", onUnauthorized as EventListener);
        return () => window.removeEventListener("auth:unauthorized", onUnauthorized as EventListener);
    }, []);

    // auto-open auth, jeśli brak sesji i nie wybrano gościa
    useEffect(() => {
        if (!loading && !user && !guest) openAuth("login");
    }, [loading, user, guest]);

    const login = async (email: string, password: string) => {
        const { error } = await supabase.auth.signInWithPassword({ email, password });
        if (error) throw new Error(error.message);

        try {
            await exchangeSession();
        } catch {}

        try {
            localStorage.removeItem("sc_guest");
        } catch {}

        await refreshMe();
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
        const { data, error } = await supabase.auth.signUp({
            email,
            password,
            options: { data: { username, first_name, last_name, date_of_birth } },
        });
        if (error) throw new Error(error.message);

        if (!data.session?.access_token) {
            try {
                await supabase.auth.signInWithPassword({ email, password });
            } catch {}
        }

        try {
            await exchangeSession();
        } catch {}

        await refreshMe();
        closeAuth();
    };

    const updateProfile = async (data: UpdateProfileData) => {
        const { data: u } = await api.put<User>("/auth/me", data);
        setUser(u);
        return u;
    };

    /** Czyści BIEŻĄCY czat (oraz bezpiecznie fallbackuje do full clear, jeśli brak API) */
    const clearCurrentChatUI = () => {
        try {
            const st = useChatStore.getState() as any;

            // preferowane metody, jeśli istnieją
            if (typeof st.clearCurrent === "function") return st.clearCurrent();
            if (typeof st.resetActive === "function") return st.resetActive();

            // łagodny fallback: wyczyść aktualny wątek
            if ("messages" in st || "input" in st || "activeSessionId" in st) {
                useChatStore.setState({
                    messages: [],
                    input: "",
                    activeSessionId: null,
                } as any);
            }

            // twardy fallback: wyczyść całość (gdyby struktura była inna)
            if (typeof st.clearAll === "function") return st.clearAll();
            useChatStore.setState({ bySession: {}, summary: {} } as any);
        } catch {
            /* ignore */
        }
    };

    const clearAfterLogout = () => {
        // czat
        clearCurrentChatUI();

        // identyfikator sesji czatu
        try {
            localStorage.removeItem("sc_session_id");
        } catch {}

        // flaga guest
        try {
            localStorage.removeItem("sc_guest");
        } catch {}

        // eventy UI
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
            // w dev backend może nie odpowiedzieć – to ok
        }
    };

    const logout = async () => {
        if (loggingOut.current) return loggingOut.current;
        loggingOut.current = (async () => {
            // 1) Supabase sesja
            try {
                await supabase.auth.signOut();
            } catch {}

            // 2) Cookie backendu
            await hardBackendLogout();

            // 3) Czyść lokalny stan (przede wszystkim AKTUALNY czat)
            clearAfterLogout();

            // 4) Zgaś usera i otwórz panel logowania
            setUser(null);
            setGuest(false);

            // Przenieś na / (opcjonalnie — lekko odświeża layout), ale BEZ twardego reloadu
            try {
                router.push("/");
            } catch {}

            // pokaż od razu panel logowania
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
        clearCurrentChatUI();
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
