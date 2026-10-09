// Review regressions: symptom panel shows a save error; the XGBoost/River overlay closes on backdrop click.
import React from "react";
import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen, fireEvent, act } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { SymptomsCard } from "@/components/SidePanel";
import ModelSplitOverlay from "@/components/ModelSplitOverlay";
import * as api from "@/lib/api";

describe("poprawki po przeglądzie", () => {
    afterEach(() => vi.restoreAllMocks());

    it("panel objawów pokazuje komunikat, gdy zapis się nie uda", async () => {
        const user = userEvent.setup();
        vi.spyOn(api, "listSymptomsDict").mockResolvedValue([]);
        vi.spyOn(api, "patchSessionSymptoms").mockRejectedValue(new Error("API 503: Model XGB niezaładowany."));
        render(<SymptomsCard sessionId="s1" symptoms={["fever"]} lang="pl" onPatched={() => {}} />);
        await user.click(screen.getByRole("button", { name: /wyczyść/i }));
        expect(await screen.findByRole("alert")).toHaveTextContent(/uruchamia/i);
    });

    it("okno XGBoost / River zamyka się kliknięciem w tło", async () => {
        vi.spyOn(api, "getTop3Split").mockResolvedValue({ xgb_enabled: true, river_enabled: true, symptoms: [] } as never);
        const onClose = vi.fn();
        let container!: HTMLElement;
        // Render inside act and let the fetch settle to avoid the "not wrapped in act" warning
        await act(async () => {
            ({ container } = render(<ModelSplitOverlay open onClose={onClose} sessionId="s1" />));
        });
        const backdrop = container.querySelector('[role="dialog"] > div[aria-hidden]') as HTMLElement;
        fireEvent.click(backdrop);
        expect(onClose).toHaveBeenCalled();
    });
});
