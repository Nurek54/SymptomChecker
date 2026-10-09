"use client";
import Image from "next/image";

export default function Logo({ className }: { className?: string }) {
  return (
    <Image
      src="/logo.png"
      alt="Logo aplikacji"
      width={64}
      height={64}
      className={["h-6 w-6 object-contain", className].filter(Boolean).join(" ")}
      quality={100}
      sizes="24px"
      priority
    />
  );
}
