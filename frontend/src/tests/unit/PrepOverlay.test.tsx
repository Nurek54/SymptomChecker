import React from "react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import PrepOverlay from "@/components/PrepOverlay";
import * as api from "@/lib/api";

const DICT = [
    { en: "fever", pl: "gorączka", category: "Ogólny", aliases: [] },
    { en: "swollen_lymph_nodes", pl: "powiększone węzły chłonne", category: "Ogólny", aliases: [] },
];

describe("PrepOverlay: adding symptoms with Enter", () => {
    const user = userEvent.setup();

    beforeEach(() => {
        vi.resetAllMocks();
        localStorage.clear();
        vi.spyOn(api, "listSymptomsDict").mockResolvedValue(DICT as never);
    });

    async function typeAndEnter(text: string) {
        const input = screen.getByPlaceholderText(/ból głowy/i);
        await user.type(input, `${text}{Enter}`);
    }

    it("a partial name picks the first matching dictionary entry", async () => {
        render(<PrepOverlay open onClose={() => {}} />);
        await screen.findByTitle("Dodaj: powiększone węzły chłonne");

        await typeAndEnter("powiększone węzły");

        await waitFor(() =>
            expect(screen.getByText(/Zgłaszam objawy: powiększone węzły chłonne\./)).toBeInTheDocument()
        );
        expect(screen.queryByText(/powiekszone_wezy/)).not.toBeInTheDocument();
    });

    it("text without a match is kept as words, ł no longer disappears", async () => {
        render(<PrepOverlay open onClose={() => {}} />);
        await screen.findByTitle("Dodaj: gorączka");

        await typeAndEnter("ból łokcia");

        await waitFor(() =>
            expect(screen.getByText(/Zgłaszam objawy: bol lokcia\./)).toBeInTheDocument()
        );
    });
});
