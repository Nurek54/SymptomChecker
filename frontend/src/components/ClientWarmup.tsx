"use client";
import { useEffect } from "react";
import { installSessionGuards, prewarmAuth } from "@/lib/api";
import { usePathname, useSearchParams } from "next/navigation";

export default function ClientWarmup() {
    const pathname = usePathname();
    const search = useSearchParams();

    useEffect(() => {
        installSessionGuards();      // focus/visibility/online
        void prewarmAuth(true);      // pierwszy mocniejszy warm
    }, []);

    // dogrzej przy każdej zmianie trasy / query
    useEffect(() => {
        void prewarmAuth();
    }, [pathname, search?.toString()]);

    return null;
}
