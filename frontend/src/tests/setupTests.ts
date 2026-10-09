import "@testing-library/jest-dom/vitest";

// Dummy env so Supabase and axios clients can be created in unit tests
process.env.NEXT_PUBLIC_SUPABASE_URL ??= "http://localhost:54321";
process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY ??= "test-anon-key";

process.env.NEXT_PUBLIC_API_BASE ??= "http://localhost:8000";
process.env.NEXT_PUBLIC_API_BASE_URL ??=
    process.env.NEXT_PUBLIC_API_BASE;
