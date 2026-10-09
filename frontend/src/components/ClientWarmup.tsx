"use client";
import { useEffect } from "react";
import { installSessionGuards, prewarmAuth } from "@/lib/api";
import { usePathname, useSearchParams } from "next/navigation";

export default function ClientWarmup() {
    const pathname = usePathname();
    const search = useSearchParams();

    useEffect(() => {
        installSessionGuards();      // focus/visibility/online
        void prewarmAuth(true);      // stronger initial warm-up
    }, []);

    // Re-warm on every route or query change
    const searchKey = search?.toString() ?? "";
    useEffect(() => {
        void prewarmAuth();
    }, [pathname, searchKey]);

    return null;
}
