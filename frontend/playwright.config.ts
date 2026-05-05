// playwright.config.ts
import { defineConfig, devices } from "@playwright/test";

export default defineConfig({
    // Gdzie leżą testy e2e
    testDir: "./src/tests/e2e",

    timeout: 30_000,
    expect: {
        timeout: 5_000,
    },

    fullyParallel: true,
    forbidOnly: !!process.env.CI,
    retries: process.env.CI ? 2 : 0,
    workers: process.env.CI ? 1 : undefined,
    reporter: "list",

    use: {
        // Twój frontend Next.js
        baseURL: "http://localhost:3000",
        trace: "on-first-retry",
        screenshot: "only-on-failure",
        video: "retain-on-failure",
    },

    // Tu definiujemy projekt "chromium",
    // o który pyta `--project=chromium`
    projects: [
        {
            name: "chromium",
            use: {
                ...devices["Desktop Chrome"],
            },
        },
    ],

    // Playwright sam uruchomi `npm run dev` (Next) na potrzeby testów
    webServer: {
        command: "npm run dev",
        port: 3000,
        reuseExistingServer: !process.env.CI,
        timeout: 120_000,
    },
});
