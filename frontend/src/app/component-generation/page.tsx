import ComponentGenerationPage from "@/components/ComponentGenerationPage";
import {
  IBM_Plex_Mono,
  IBM_Plex_Sans,
  Playfair_Display,
} from "next/font/google";

const editorialDisplay = Playfair_Display({
  subsets: ["latin"],
  variable: "--font-cg-display",
  weight: ["400", "700", "900"],
});

const editorialSans = IBM_Plex_Sans({
  subsets: ["latin"],
  variable: "--font-cg-sans",
  weight: ["300", "400", "500", "700"],
});

const editorialMono = IBM_Plex_Mono({
  subsets: ["latin"],
  variable: "--font-cg-mono",
  weight: ["400", "500", "600"],
});

export default function ComponentGenerationRoute() {
  return (
    <div
      className={`${editorialDisplay.variable} ${editorialSans.variable} ${editorialMono.variable} cg-route`}
    >
      <ComponentGenerationPage />
    </div>
  );
}
