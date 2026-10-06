import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  reactStrictMode: true,
  // swcMinify usunięte: w Next 15 minifikacja SWC jest zawsze włączona, a klucz jest nieznany
};

export default nextConfig;
