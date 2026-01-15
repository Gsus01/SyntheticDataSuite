import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Enable standalone output for optimized Docker production builds
  // This creates a minimal bundle with only the necessary files
  output: "standalone",

  // Proxy /api requests to the backend server
  // This allows remote access without exposing port 8000
  async rewrites() {
    const backendUrl = process.env.BACKEND_URL || "http://localhost:8000";
    return [
      {
        source: "/api/:path*",
        destination: `${backendUrl}/:path*`,
      },
    ];
  },
};

export default nextConfig;
