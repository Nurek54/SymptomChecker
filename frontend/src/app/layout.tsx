// app/layout.tsx
import "./globals.css";
import type { Metadata } from "next";
import AuthProvider from "@/components/AuthProvider";
import AuthSheet from "@/components/AuthSheet";
import AutoOpenAuth from "@/components/AutoOpenAuth";
import ClientWarmup from "@/components/ClientWarmup"; // ⬅️ pre-warm + strażnicy sesji

export const metadata: Metadata = {
    title: "Symptom Checker",
    description: "Czat do wstępnej oceny objawów",
};

/** No-flash: ustaw motyw zanim React się zhydratuje */
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
        {/* ⬇️ Jednorazowy globalny warm-up i strażnicy (focus/visibility/online) */}
        <ClientWarmup />
        <AuthProvider>
            {/* auto pokaże panel, jeśli brak sesji i nie wybrano gościa */}
            <AutoOpenAuth />
            {children}
            <AuthSheet />
        </AuthProvider>
        </body>
        </html>
    );
}
