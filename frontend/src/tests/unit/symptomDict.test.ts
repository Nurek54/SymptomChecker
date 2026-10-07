import { describe, it, expect } from "vitest";
import { matchSymptom, normalizeSymptomDict } from "@/lib/symptomDict";

describe("słownik objawów: jeden wpis na cechę", () => {
  it("scala duplikaty `en` w aliasy", () => {
    const out = normalizeSymptomDict([
      { en: "Mouth_pain", pl: "ból jamy ustnej" },
      { en: "mouth_pain", pl: "ból języka" },
      { en: "mouth_pain", pl: "ból jamy ustnej" },
      { en: "hip_pain", pl: "ból biodra", aliases: ["ból stawu biodrowego"] },
    ]);
    expect(out.map((x) => x.en)).toEqual(["mouth_pain", "hip_pain"]);
    expect(out[0]).toMatchObject({ pl: "ból jamy ustnej", aliases: ["ból języka"] });
    expect(out[1].aliases).toEqual(["ból stawu biodrowego"]);
  });

  it("szuka po etykiecie i aliasach", () => {
    const it0 = { en: "mouth_pain", pl: "ból jamy ustnej", aliases: ["ból języka"] };
    expect(matchSymptom(it0, "jamy")).toBe("");
    expect(matchSymptom(it0, "języka")).toBe("ból języka");
    expect(matchSymptom(it0, "kolano")).toBeNull();
    expect(matchSymptom(it0, "mouth", "en")).toBe("");
  });
});
