// src/tests/e2e/auth-guest.spec.ts
import { test, expect } from "@playwright/test";
import { continueAsGuest } from "./utils";

test("F1/A1: bez cookie wejście na /history pokazuje blokadę dostępu", async ({ page }) => {
    // Każdy test Playwright startuje z pustym storage (brak cookie)
    await page.goto("/history");

    await expect(page.getByText(/historia rozmów/i)).toBeVisible();
    await expect(page.getByText(/musisz się zalogować/i)).toBeVisible();

    // Nie widać panelu z listą rozmów ("Twoje rozmowy")
    await expect(page.getByText(/twoje rozmowy/i)).toHaveCount(0);
});

test("F1/R1/E2E_03: tryb gościa — czat działa, historia zablokowana", async ({ page }) => {
    await continueAsGuest(page);

    // 2) Wpisujemy objawy w pole czatu
    const textbox = page.getByRole("textbox").last();
    await textbox.fill("gorączka, kaszel, ból gardła");
    await textbox.press("Enter");

    // 3) Czekamy na jakąkolwiek odpowiedź asystenta lub Top-3
    const anyCard = page
        .getByText(/możliwe choroby|najbardziej prawdopodobne|zalecenia/i)
        .first();
    await expect(anyCard).toBeVisible({ timeout: 20_000 });

    // 4) Próba wejścia do historii jako gość → blokada
    await page.goto("/history");

    await expect(page.getByText(/historia rozmów/i)).toBeVisible();
    await expect(page.getByText(/musisz się zalogować/i)).toBeVisible();
});
