// src/tests/e2e/utils.ts
import type { Page } from "@playwright/test";

export async function continueAsGuest(page: Page) {
    await page.goto("/");

    // Czekamy na AuthSheet z przyciskiem "Kontynuuj jako gość" albo "Kontynuuj bez logowania"
    const guestButton = page.getByRole("button", {
        name: /kontynuuj (jako gość|bez logowania)/i,
    });
    await guestButton.waitFor();
    await guestButton.click();

    // Po kliknięciu powinien być widoczny ekran czatu (pole do wpisania objawów)
    const textbox = page.getByRole("textbox").last();
    await textbox.waitFor();
}
