import { ThemeProvider } from "@/lib/theme-context";
import type { Metadata, Viewport } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import SponsorBar from "@/components/SponsorBar";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: {
    default: "SyntheticDataSuite",
    template: "%s | SyntheticDataSuite",
  },
  description:
    "Suite para diseño, ejecución y validación de workflows de generación de datos sintéticos.",
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  themeColor: "#0f172a",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={`${geistSans.variable} ${geistMono.variable} antialiased`}>
        <ThemeProvider>
          <div className="flex h-screen flex-col overflow-hidden">
            <main className="min-h-0 flex-1 overflow-hidden">{children}</main>
            <SponsorBar />
          </div>
        </ThemeProvider>
      </body>
    </html>
  );
}
