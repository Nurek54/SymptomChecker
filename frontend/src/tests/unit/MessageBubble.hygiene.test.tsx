// src/tests/unit/MessageBubble.hygiene.test.tsx
import React from "react";
import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import MessageBubble from "@/components/MessageBubble";

describe("F3 — Higiena odpowiedzi (UI, MessageBubble)", () => {
    it("renderuje pojedynczy, czytelny akapit bez dawek/nazw leków", () => {
        const safeText =
            "Podejrzewam infekcję górnych dróg oddechowych. Objawy są łagodne, obserwuj stan i w razie pogorszenia skontaktuj się z lekarzem.";

        const { container } = render(
            <MessageBubble role="assistant" content={safeText} />
        );

        const bubble = screen.getByText(/podejrzewam infekcję/i).closest("div");
        expect(bubble).toBeInTheDocument();

        // 1) Nie pojawiają się typowe wzorce dawek / nazw handlowych
        expect(bubble).not.toHaveTextContent(/\b\d+\s*mg\b/i);
        expect(bubble).not.toHaveTextContent(/ibuprofen|apap|ibufen|nurofen/i);

        // 2) Treść jest w jednym akapicie markdown (bez sztucznego dzielenia)
        const paragraphs = container.querySelectorAll("p");
        expect(paragraphs.length).toBe(1);
        expect(paragraphs[0]).toHaveTextContent(/infekcję górnych dróg oddechowych/i);
    });
});
