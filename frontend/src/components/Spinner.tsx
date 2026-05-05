"use client";

import * as React from "react";

type Props = {
  /** Średnica w px (domyślnie 16 = h-4 w-4) */
  size?: number;
  /** Grubość pierścienia w px (domyślnie ~1/6 średnicy) */
  thickness?: number;
  className?: string;
};

export default function Spinner({ size = 16, thickness, className }: Props) {
  const ring = Math.max(2, thickness ?? Math.round(size / 6));
  const s = `${size}px`;
  const mask = `radial-gradient(farthest-side, transparent calc(100% - ${ring}px), #000 calc(100% - ${ring}px))`;

  const style: React.CSSProperties = {
    width: s,
    height: s,
    // gradient zgodny z Logo
    background:
      "conic-gradient(from 90deg, #34d399, #38bdf8 55%, #a78bfa 100%)",
    // „dziura” w środku, żeby powstał pierścień
    WebkitMask: mask,
    mask,
  };

  return (
    <span
      role="status"
      aria-label="Ładowanie"
      className={`inline-block animate-spin rounded-full shadow-[0_0_0_1px_rgba(0,0,0,0.04)] ${className ?? ""}`}
      style={style}
    />
  );
}
