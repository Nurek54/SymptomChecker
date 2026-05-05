// vitest.config.ts
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
        // Bierzemy TYLKO testy jednostkowe, e2e ignorujemy
        include: ["src/tests/unit/**/*.test.{ts,tsx}"],
        exclude: [
            "src/tests/e2e/**",
            "node_modules/**",
            ".next/**",
        ],
    },
});
