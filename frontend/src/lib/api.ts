// API base URL for backend requests
// - Production/Ingress: "/api" (requests go through Ingress which routes to backend)
// - Local development: "http://localhost:8000" (direct to backend)
// Override with NEXT_PUBLIC_API_BASE_URL environment variable
export const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL || "/api";

/**
 * Build a URL for API requests that works with both relative and absolute base URLs.
 * 
 * - If API_BASE is absolute (http://...), uses it directly as URL base
 * - If API_BASE is relative (/api), uses window.location.origin as base
 * 
 * @param path - The API path (e.g., "/workflow/status")
 * @returns A URL object that can have searchParams added
 */
export function buildApiUrl(path: string): URL {
    const fullPath = `${API_BASE}${path}`;

    // Check if API_BASE is already absolute
    if (API_BASE.startsWith("http://") || API_BASE.startsWith("https://")) {
        return new URL(fullPath);
    }

    // For relative URLs, use the current origin as base
    // In SSR context, window might not exist, so we need a fallback
    const origin = typeof window !== "undefined"
        ? window.location.origin
        : "http://localhost:3000";

    return new URL(fullPath, origin);
}
