"use client";

import type { Role } from "@/lib/types";
import React, { useMemo, useState } from "react";
import Markdown from "react-markdown";
import { Copy, Stethoscope } from "lucide-react";

// Strip trailing spaces before newlines so header lines (e.g. "Zalecenia:") match exactly
function normalize(text: string) {
  return text.replace(/[ \t]+\n/g, "\n").trim();
}

function fmtTime(iso?: string) {
  if (!iso) return "";
  try {
    const d = new Date(iso);
    return new Intl.DateTimeFormat("pl-PL", {
      hour: "2-digit",
      minute: "2-digit",
    }).format(d);
  } catch {
    return "";
  }
}

export default function MessageBubble({
  role,
  content,
  createdAt,
}: {
  role: Role;
  content: string;
  createdAt?: string;
}) {
  const isUser = role === "user";
  const [copied, setCopied] = useState(false);
  const safeContent = useMemo(() => normalize(content || ""), [content]);
  const time = useMemo(() => fmtTime(createdAt), [createdAt]);

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(safeContent);
      setCopied(true);
      setTimeout(() => setCopied(false), 900);
    } catch {}
  };

  return (
    <div className={`group flex w-full ${isUser ? "justify-end" : "justify-start"}`}>
      {!isUser && (
        <div className="mr-2 mt-0.5 hidden shrink-0 sm:block">
          <span
            className="relative grid h-8 w-8 place-items-center rounded-full border border-base shadow-sm"
            style={{
              background:
                "linear-gradient(135deg, rgba(52,211,153,.18), rgba(56,189,248,.18) 55%, rgba(167,139,250,.18))",
            }}
            aria-hidden
          >
            <Stethoscope className="h-4 w-4 text-emerald-500 dark:text-emerald-300" />
            <span
              aria-hidden
              className="pointer-events-none absolute -inset-px rounded-full opacity-60 blur-[3px]"
              style={{
                background:
                  "linear-gradient(135deg, rgba(52,211,153,.20), rgba(167,139,250,.20))",
              }}
            />
          </span>
        </div>
      )}

      <div className="flex max-w-[85%] flex-col">
        <div
          className={[
            "relative prose whitespace-pre-wrap rounded-2xl px-4 py-2 text-sm shadow-sm transition",
            isUser
              ? // user: right-aligned, narrower, flat bottom-right corner
                "ml-auto max-w-[60ch] rounded-br-md border border-emerald-500/30 " +
                "bg-gradient-to-br from-emerald-500/20 to-cyan-500/15 " +
                "text-basecolor hover:shadow-md hover:from-emerald-500/25 hover:to-cyan-500/20"
              : // assistant: card style, flat bottom-left corner
                "rounded-bl-md bg-card text-basecolor border border-base " +
                "hover:shadow-md hover:ring-1 hover:ring-[rgb(var(--border))]",
          ].join(" ")}
        >
          {!isUser && (
            <div className="absolute -top-3 right-2 flex items-center gap-1 opacity-0 transition-opacity group-hover:opacity-100 focus-within:opacity-100 [@media(hover:none)]:opacity-100">
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
              a: (props) => <a {...props} className="underline underline-offset-2" />,
              li: (props) => <li {...props} className="ml-4" />,
              code: (props) => <code {...props} className="rounded px-1 py-0.5" />,
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

        {time && (
          <span
            className={[
              "mt-1 text-[10px] tabular-nums text-white/30 opacity-0 transition-opacity duration-200",
              "group-hover:opacity-100",
              isUser ? "text-right pr-1" : "pl-1",
            ].join(" ")}
            aria-hidden
          >
            {time}
          </span>
        )}
      </div>
    </div>
  );
}
