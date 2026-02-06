// tailwind.config.ts
import type { Config } from 'tailwindcss'

const config: Config = {
    content: [
        "./src/**/*.{ts,tsx,js,jsx}",
        "./pages/**/*.{ts,tsx,js,jsx}",
        "./app/**/*.{ts,tsx,js,jsx}"
    ],
    darkMode: 'class', // usa la clase "dark" para activar el modo oscuro
    theme: {
        extend: {
            // Mapear colores a CSS variables para que el toggle cambie los valores en tiempo real
            colors: {
                // Ejemplos de tokens basados en CSS variables
                page: 'rgb(var(--bg))',
                surface: 'rgb(var(--surface))',
                text: 'rgb(var(--text))',
                border: 'rgb(var(--border))',
                ring: 'rgb(var(--ring))',
                primary: 'rgb(var(--primary))',
                secondary: 'rgb(var(--secondary))',
                // Puedes añadir más tokens si migras partes de la UI con estos nombres
            }
        }
    },
    plugins: [],
}
export default config
