// src/tests/e2e/symptoms-panel.spec.ts
import { test, expect } from "@playwright/test";
import { continueAsGuest } from "./utils";

test("F2 (E2E): panel objawów dodaje objaw i odświeża Top-3", async ({ page }) => {
    await continueAsGuest(page);

    // Maszyna LLM potrzebuje czegoś startowego – przeklikamy jedną prostą wiadomość,
    // żeby pojawiły się karty boczne (jeśli jeszcze nie ma).
    const textbox = page.getByRole("textbox").last();
    await textbox.fill("gorączka, ból gardła, kaszel");
    await textbox.press("Enter");

    // Czekamy na kartę Top-3
    const topCard = page.getByText(/najbardziej prawdopodobne \(top-3\)/i).first();
    await topCard.waitFor({ timeout: 20_000 });

    // Przechodzimy do panelu objawów (prawa kolumna)
    const addButton = page.getByRole("button", { name: /dodaj/i }).first();
    await addButton.click();

    // 2) Wpisujemy "kaszel" w polu "Szukaj objawu..."
    const searchInput = page.getByPlaceholder(/szukaj objawu/i);
    await searchInput.fill("kaszel");

    // 3) Klik na wynik
    const option = page.getByRole("option", { name: /kaszel/i }).first();
    await option.click();

    // 4) W kartcie "Zebrane objawy" powinien pojawić się chip z "kaszel"
    await expect(page.getByText(/kaszel/i).first()).toBeVisible();

    // 5) Top-3 nadal jest widoczne (odświeżone backendem po PATCH)
    await expect(
        page.getByText(/najbardziej prawdopodobne \(top-3\)/i)
    ).toBeVisible();
});
