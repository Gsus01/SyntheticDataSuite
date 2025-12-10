import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Enable standalone output for optimized Docker production builds
  // This creates a minimal bundle with only the necessary files
  output: "standalone",
};

export default nextConfig;
