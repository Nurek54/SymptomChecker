import React, { useState } from "react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { SymptomsCard, TopDiagCard } from "@/components/SidePanel";
import * as api from "@/lib/api";
import type { SymptomsPatchResponse, TopDiagnosis } from "@/lib/types";

describe("F2 — Panel objawów (SymptomsCard + TopDiagCard)", () => {
    const user = userEvent.setup();

    beforeEach(() => {
        vi.resetAllMocks();
    });

    /** Minimal ChatUI stand-in: holds `symptoms` and `top` and updates them from `onPatched`. */
    function SymptomsHarness() {
        const [symptoms, setSymptoms] = useState<string[]>([]);
        const [top, setTop] = useState<TopDiagnosis[] | null>(null);

        return (
            <div>
                <SymptomsCard
                    sessionId="test-session"
                    symptoms={symptoms}
                    lang="pl"
                    onPatched={(res: SymptomsPatchResponse) => {
                        setSymptoms(res.symptoms ?? []);
                        setTop(res.top_diagnoses ?? null);
                    }}
                />
                <div aria-label="Top-3 harness">
                    <TopDiagCard items={top} />
                </div>
            </div>
        );
    }

    it("po wpisaniu 'kaszel' pokazuje wynik PL/EN, wywołuje PATCH i aktualizuje listę oraz Top-3", async () => {
        vi.spyOn(api, "listSymptomsDict").mockResolvedValue([
            { en: "cough", pl: "kaszel", category: "oddechowe" },
            { en: "sore_throat", pl: "ból gardła", category: "oddechowe" },
        ]);

        const patchSpy = vi
            .spyOn(api, "patchSessionSymptoms")
            .mockResolvedValue({
                symptoms: ["cough", "sore_throat"],
                top_diagnoses: [
                    { diagnosis_en: "common_cold", diagnosis_pl: "Przeziębienie", probability: 0.7 },
                    { diagnosis_en: "flu", diagnosis_pl: "Grypa", probability: 0.2 },
                    { diagnosis_en: "covid_19", diagnosis_pl: "COVID-19", probability: 0.1 },
                ],
            });

        render(<SymptomsHarness />);

        const addButton = screen.getByRole("button", { name: /dodaj/i });
        await user.click(addButton);

        const searchInput = await screen.findByPlaceholderText(/szukaj objawu/i);
        await user.type(searchInput, "kaszel");

        const option = await screen.findByRole("option", { name: /kaszel/i });
        await user.click(option);

        // PATCH must send the canonical EN name
        await waitFor(() => {
            expect(patchSpy).toHaveBeenCalledTimes(1);
        });

        expect(patchSpy).toHaveBeenCalledWith(
            "test-session",
            expect.objectContaining({ add: ["cough"], lang: "pl" })
        );

        await waitFor(() => {
            expect(screen.getAllByText(/kaszel/i).length).toBeGreaterThan(0);
        });

        const topCard = screen.getByText(/Najbardziej prawdopodobne/i).closest("div");
        expect(topCard).toBeInTheDocument();
        expect(screen.getByText(/przeziębienie/i)).toBeInTheDocument();
    });
});
