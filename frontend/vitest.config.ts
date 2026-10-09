import { defineConfig } from "vitest/config";
import path from "path";

export default defineConfig({
    resolve: {
        alias: {
            "@": path.resolve(__dirname, "src"),
        },
    },
    test: {
        environment: "jsdom",
        globals: true,
        setupFiles: ["./src/tests/setupTests.ts"],
        // Unit tests only; e2e specs run under Playwright
        include: ["src/tests/unit/**/*.test.{ts,tsx}"],
        exclude: [
            "src/tests/e2e/**",
            "node_modules/**",
            ".next/**",
        ],
    },
});
