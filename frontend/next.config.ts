import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  reactStrictMode: true,
  // No swcMinify: Next 15 always minifies with SWC and rejects the key as unknown
};

export default nextConfig;
