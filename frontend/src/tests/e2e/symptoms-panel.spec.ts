import { test, expect } from "@playwright/test";
import { continueAsGuest } from "./utils";

test("F2 (E2E): panel objawów dodaje objaw i odświeża Top-3", async ({ page }) => {
    await continueAsGuest(page);

    // Send one message first so the side cards appear
    const textbox = page.getByRole("textbox").last();
    await textbox.fill("gorączka, ból gardła, kaszel");
    await textbox.press("Enter");

    const topCard = page.getByText(/najbardziej prawdopodobne \(top-3\)/i).first();
    await topCard.waitFor({ timeout: 20_000 });

    const addButton = page.getByRole("button", { name: /dodaj/i }).first();
    await addButton.click();

    const searchInput = page.getByPlaceholder(/szukaj objawu/i);
    await searchInput.fill("kaszel");

    const option = page.getByRole("option", { name: /kaszel/i }).first();
    await option.click();

    await expect(page.getByText(/kaszel/i).first()).toBeVisible();

    // Top-3 stays visible after the backend refresh that follows PATCH
    await expect(
        page.getByText(/najbardziej prawdopodobne \(top-3\)/i)
    ).toBeVisible();
});
