"use client";

import * as React from "react";

type Props = {
  /** Diameter in px (default 16 = h-4 w-4) */
  size?: number;
  /** Ring thickness in px (default ~1/6 of the diameter) */
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
    // Gradient matches the logo
    background:
      "conic-gradient(from 90deg, #34d399, #38bdf8 55%, #a78bfa 100%)",
    // Mask out the center to turn the disc into a ring
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
