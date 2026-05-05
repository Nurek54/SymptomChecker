// src/tests/e2e/fullflow-logged.spec.ts
import { test, expect, Page } from "@playwright/test";

async function loginAsTestUser(page: Page) {
    await page.goto("/");

    // Bierzemy tylko pola z DIALOGU "Okno logowania", nie z tła
    const dialog = page.getByRole("dialog", { name: /okno logowania/i });
    await dialog.waitFor();

    const emailInput = dialog.getByPlaceholder(/przyklad\.pl/i);
    const passwordInput = dialog.getByPlaceholder(/min\. 8 znaków/i);

    const email = process.env.E2E_EMAIL ?? "test@example.com";
    const password = process.env.E2E_PASSWORD ?? "***REMOVED***";

    await emailInput.fill(email);
    await passwordInput.fill(password);

    // Klikamy przycisk "Zaloguj się" TYLKO z tego dialogu
    const loginButton = dialog.getByRole("button", { name: /zaloguj/i });
    await loginButton.click({ force: true });

    // Po udanym logowaniu powinno być widać główne pole czatu
    const textbox = page.getByRole("textbox").last();
    await textbox.waitFor();
}

test("E2E_01: pełny przepływ zalogowany użytkownik", async ({ page }) => {
    await loginAsTestUser(page);

    const textbox = page.getByRole("textbox").last();
    await textbox.fill("gorączka, kaszel, ból gardła");
    await textbox.press("Enter");

    // 1) Czekamy na odpowiedź asystenta + Top-3
    const topCard = page.getByText(/najbardziej prawdopodobne \(top-3\)/i).first();
    await topCard.waitFor({ timeout: 20_000 });

    // 2) Dodanie objawu przez panel
    const addButton = page.getByRole("button", { name: /dodaj/i }).first();
    await addButton.click();

    const searchInput = page.getByPlaceholder(/szukaj objawu/i);
    await searchInput.fill("katar");

    const option = page.getByRole("option", { name: /katar/i }).first();
    await option.click();

    // Chip "katar" w zebranych objawach
    await expect(page.getByText(/katar/i).first()).toBeVisible();

    // 3) Przejście do historii
    await page.goto("/history");

    // Dla zalogowanego użytkownika widzimy listę rozmów
    await expect(page.getByText(/twoje rozmowy/i)).toBeVisible();
});
