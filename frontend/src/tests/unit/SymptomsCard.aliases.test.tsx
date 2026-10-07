// Osobny plik: SidePanel trzyma słownik w cache modułu, więc każdy słownik testowy dostaje świeży moduł.
import React from "react";
import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { SymptomsCard } from "@/components/SidePanel";
import * as api from "@/lib/api";

describe("Panel objawów: synonimy tej samej cechy", () => {
  afterEach(() => vi.restoreAllMocks());

  it("nie dubluje kluczy i znajduje objaw po synonimie", async () => {
    const user = userEvent.setup();
    const errSpy = vi.spyOn(console, "error").mockImplementation(() => {});
    vi.spyOn(api, "listSymptomsDict").mockResolvedValue([
      { en: "mouth_pain", pl: "ból jamy ustnej", category: null },
      { en: "mouth_pain", pl: "ból języka", category: null },
      { en: "hip_pain", pl: "ból biodra", category: null, aliases: ["ból stawu biodrowego"] },
    ]);
    const patchSpy = vi.spyOn(api, "patchSessionSymptoms").mockResolvedValue({ symptoms: ["mouth_pain"], top_diagnoses: [] });

    render(<SymptomsCard sessionId="s1" symptoms={[]} lang="pl" onPatched={() => {}} />);
    await user.click(screen.getByRole("button", { name: /dodaj/i }));

    expect(await screen.findAllByRole("option")).toHaveLength(2);
    expect(errSpy.mock.calls.some((c) => String(c[0]).includes("same key"))).toBe(false);

    await user.type(await screen.findByPlaceholderText(/szukaj objawu/i), "języka");
    const opt = await screen.findByRole("option", { name: /ból jamy ustnej/i });
    expect(opt).toHaveTextContent("ból języka");
    await user.click(opt);
    expect(patchSpy).toHaveBeenCalledWith("s1", expect.objectContaining({ add: ["mouth_pain"] }));
  });
});
