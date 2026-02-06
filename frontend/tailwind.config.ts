// tailwind.config.ts
import type { Config } from 'tailwindcss'

const config: Config = {
    content: [
        "./src/**/*.{ts,tsx,js,jsx}",
        "./pages/**/*.{ts,tsx,js,jsx}",
        "./app/**/*.{ts,tsx,js,jsx}"
    ],
    darkMode: 'class',
    theme: {
        extend: {
            colors: {
                page: 'rgb(var(--bg))',
                surface: 'rgb(var(--surface))',
                text: 'rgb(var(--text))',
                border: 'rgb(var(--border))',
                ring: 'rgb(var(--ring))',
                primary: 'rgb(var(--primary))',
                secondary: 'rgb(var(--secondary))',
            }
        }
    },
    plugins: [],
}
export default config
