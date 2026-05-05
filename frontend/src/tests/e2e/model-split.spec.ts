// src/tests/e2e/model-split.spec.ts
import { test, expect } from "@playwright/test";
import { continueAsGuest } from "./utils";

test("E2E_02: ModelSplitOverlay — trzy kolumny XGBoost / River / Ensemble z procentami", async ({ page }) => {
    await continueAsGuest(page);

    const textbox = page.getByRole("textbox").last();
    await textbox.fill("gorączka, kaszel, katar, ból głowy");
    await textbox.press("Enter");

    // Czekamy na Top-3, żeby przycisk info miał sens
    const topCard = page.getByText(/najbardziej prawdopodobne \(top-3\)/i).first();
    await topCard.waitFor({ timeout: 20_000 });

    // Klik w ikonkę "ⓘ" (TopDiagCard)
    const infoButton = page.getByRole("button", {
        name: /pokaż osobno xgboost i river/i,
    });
    await infoButton.click();

    // Szukamy overlay'a z trzema kolumnami
    await expect(page.getByText(/xgboost/i).first()).toBeVisible();
    await expect(page.getByText(/river/i).first()).toBeVisible();
    await expect(page.getByText(/ensemble/i).first()).toBeVisible();

    // W środku powinny być procenty (np. "45%")
    const percents = page.getByText(/\d+%/).first();
    await expect(percents).toBeVisible();
});
