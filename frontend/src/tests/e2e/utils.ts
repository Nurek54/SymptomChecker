import type { Page } from "@playwright/test";

export async function continueAsGuest(page: Page) {
    await page.goto("/");

    // AuthSheet labels the guest button in one of two ways
    const guestButton = page.getByRole("button", {
        name: /kontynuuj (jako gość|bez logowania)/i,
    });
    await guestButton.waitFor();
    await guestButton.click();

    const textbox = page.getByRole("textbox").last();
    await textbox.waitFor();
}
