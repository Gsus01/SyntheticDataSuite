import ComponentGenerationPage from "@/components/ComponentGenerationPage";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Component Generation",
  description:
    "Generación automática de componentes con revisión human-in-the-loop, timeline de eventos y consola en vivo.",
};

export default function ComponentGenerationRoute() {
  return <ComponentGenerationPage />;
}
