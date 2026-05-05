"use client";
import { useEffect } from "react";
import { useAuth } from "@/components/AuthProvider";

export default function AutoOpenAuth() {
  const { user, loading, isGuest, openAuth } = useAuth();
  useEffect(() => {
    if (!loading && !user && !isGuest) openAuth("login");
  }, [loading, user, isGuest, openAuth]);
  return null;
}
