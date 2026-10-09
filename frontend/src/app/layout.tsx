import "./globals.css";
import type { Metadata } from "next";
import { Suspense } from "react";
import AuthProvider from "@/components/AuthProvider";
import AuthSheet from "@/components/AuthSheet";
import AutoOpenAuth from "@/components/AutoOpenAuth";
import ClientWarmup from "@/components/ClientWarmup";

export const metadata: Metadata = {
    title: "Symptom Checker",
    description: "Czat do wstępnej oceny objawów",
};

/** Applies the theme class before hydration to avoid a flash of the wrong theme. */
const noFlashTheme = `
(function () {
  try {
    var t = localStorage.getItem("theme");
    if (t !== "light" && t !== "dark") {
      t = "light";
      localStorage.setItem("theme", t);
    }
    if (t === "dark") document.documentElement.classList.add("dark");
    else document.documentElement.classList.remove("dark");
  } catch (_) {}
})();
`;

export default function RootLayout({ children }: { children: React.ReactNode }) {
    return (
        <html lang="pl" suppressHydrationWarning>
        <head>
            <script id="theme-init" dangerouslySetInnerHTML={{ __html: noFlashTheme }} />
            <meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover" />
        </head>
        <body className="min-h-screen antialiased transition-colors">
        {/* Backend warm-up and session guards (focus/visibility/online); useSearchParams() needs a Suspense boundary or "next build" fails */}
        <Suspense fallback={null}>
            <ClientWarmup />
        </Suspense>
        <AuthProvider>
            {/* Opens the auth sheet when there is no session and guest mode was not chosen */}
            <AutoOpenAuth />
            {children}
            <AuthSheet />
        </AuthProvider>
        </body>
        </html>
    );
}
