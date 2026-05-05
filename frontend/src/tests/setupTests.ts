// src/tests/setupTests.ts
import "@testing-library/jest-dom/vitest";

// Sztuczne ENV, żeby Supabase / axios nie wybuchały w testach unit
process.env.NEXT_PUBLIC_SUPABASE_URL ??= "http://localhost:54321";
process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY ??= "test-anon-key";

process.env.NEXT_PUBLIC_API_BASE ??= "http://localhost:8000";
process.env.NEXT_PUBLIC_API_BASE_URL ??=
    process.env.NEXT_PUBLIC_API_BASE;
