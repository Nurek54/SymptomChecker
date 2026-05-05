"use client";

import type { Role } from "@/lib/types";
import React, { useMemo, useState } from "react";
import Markdown from "react-markdown";
import { Copy } from "lucide-react";

// ——— Prosta normalizacja nagłówków (np. „Zalecenia:”, „Podsumowanie:”) ———
function normalize(text: string) {
  // Bezpieczne: nie ruszamy markdownu, tylko poprawiamy typowe duplikaty spacji i trim
  return text.replace(/[ \t]+\n/g, "\n").trim();
}

export default function MessageBubble({ role, content }: { role: Role; content: string }) {
  const isUser = role === "user";
  const [copied, setCopied] = useState(false);
  const safeContent = useMemo(() => normalize(content || ""), [content]);

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(safeContent);
      setCopied(true);
      setTimeout(() => setCopied(false), 900);
    } catch {}
  };

  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"}`}>
      <div
        className={[
          "group relative prose max-w-[85%] whitespace-pre-wrap rounded-2xl px-4 py-2 text-sm shadow-sm transition",
          isUser
            // usunięto hover:-translate-y-0.5 i active:translate-y-[1px]
            ? "rounded-br-md bg-bubble-user text-bubble-user border border-bubble-user hover:shadow-md hover:ring-1 hover:ring-[rgb(var(--border))]"
            : "rounded-bl-md bg-card text-basecolor border border-base hover:shadow-md hover:ring-1 hover:ring-[rgb(var(--border))]",
        ].join(" ")}
      >
        {/* Akcje na dymku asystenta */}
        {!isUser && (
          <div className="absolute -top-3 right-2 flex items-center gap-1 opacity-0 transition-opacity group-hover:opacity-100">
            <button
              onClick={copy}
              className="inline-flex items-center gap-1 rounded-full border border-base bg-card px-2 py-0.5 text-[10px] shadow-sm hover:brightness-110"
              title="Skopiuj odpowiedź"
            >
              <Copy className="h-3 w-3" />
              Kopiuj
            </button>
            {copied && (
              <span className="rounded-full border border-base bg-card px-2 py-0.5 text-[10px] shadow-sm">
                Skopiowano!
              </span>
            )}
          </div>
        )}

        <Markdown
          components={{
            // Linki: podkreślone i czytelne
            a: (props) => <a {...props} className="underline underline-offset-2" />,
            // Listy: odrobina wcięcia
            li: (props) => <li {...props} className="ml-4" />,
            // Kod inline: delikatne tło (zostawiamy minimalne)
            code: (props) => <code {...props} className="rounded px-1 py-0.5" />,
            // Pseudo-nagłówki: w markdown użytkownika mogą pojawić się linie z dwukropkiem
            p: (props) => {
              const text = String(props.children ?? "");
              const isHeader =
                /^(podsumowanie|summary|możliwe|possible|zalecenia|recommendations|triage|pilność|urgency)\s*:$/i.test(
                  text.trim()
                );
              return (
                <p
                  {...props}
                  className={
                    (props.className ?? "") +
                    (isHeader ? " font-semibold text-basecolor mt-2" : " text-basecolor")
                  }
                />
              );
            },
          }}
        >
          {safeContent}
        </Markdown>
      </div>
    </div>
  );
}
