import { test, expect } from "@playwright/test";
import { continueAsGuest } from "./utils";

test("F1/A1: bez cookie wejście na /history pokazuje blokadę dostępu", async ({ page }) => {
    // Each Playwright test starts with empty storage (no cookie)
    await page.goto("/history");

    await expect(page.getByText(/historia rozmów/i)).toBeVisible();
    await expect(page.getByText(/musisz się zalogować/i)).toBeVisible();

    // Conversation list must not render
    await expect(page.getByText(/twoje rozmowy/i)).toHaveCount(0);
});

test("F1/R1/E2E_03: tryb gościa — czat działa, historia zablokowana", async ({ page }) => {
    await continueAsGuest(page);

    const textbox = page.getByRole("textbox").last();
    await textbox.fill("gorączka, kaszel, ból gardła");
    await textbox.press("Enter");

    const anyCard = page
        .getByText(/możliwe choroby|najbardziej prawdopodobne|zalecenia/i)
        .first();
    await expect(anyCard).toBeVisible({ timeout: 20_000 });

    await page.goto("/history");

    await expect(page.getByText(/historia rozmów/i)).toBeVisible();
    await expect(page.getByText(/musisz się zalogować/i)).toBeVisible();
});
