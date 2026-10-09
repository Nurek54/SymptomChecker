import { test, expect } from "@playwright/test";
import { continueAsGuest } from "./utils";

test("E2E_02: ModelSplitOverlay — trzy kolumny XGBoost / River / Ensemble z procentami", async ({ page }) => {
    await continueAsGuest(page);

    const textbox = page.getByRole("textbox").last();
    await textbox.fill("gorączka, kaszel, katar, ból głowy");
    await textbox.press("Enter");

    // The info button only makes sense once Top-3 is shown
    const topCard = page.getByText(/najbardziej prawdopodobne \(top-3\)/i).first();
    await topCard.waitFor({ timeout: 20_000 });

    const infoButton = page.getByRole("button", {
        name: /pokaż osobno xgboost i river/i,
    });
    await infoButton.click();

    await expect(page.getByText(/xgboost/i).first()).toBeVisible();
    await expect(page.getByText(/river/i).first()).toBeVisible();
    await expect(page.getByText(/ensemble/i).first()).toBeVisible();

    const percents = page.getByText(/\d+%/).first();
    await expect(percents).toBeVisible();
});
