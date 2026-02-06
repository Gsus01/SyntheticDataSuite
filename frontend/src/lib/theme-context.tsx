// src/lib/theme-context.tsx
"use client";
import React, { createContext, useContext, useEffect, useState } from "react";

type ThemeContextType = {
    isDark: boolean;
    toggleTheme: () => void;
    setTheme: (dark: boolean) => void;
};

const ThemeContext = createContext<ThemeContextType | undefined>(undefined);

export function useTheme() {
    const context = useContext(ThemeContext);
    if (!context) throw new Error("useTheme must be used within ThemeProvider");
    return context;
}

export function ThemeProvider({ children }: { children: React.ReactNode }) {
    const [isDark, setIsDark] = useState(() => {
        // Check localStorage first, fallback to system preference
        if (typeof window !== 'undefined') {
            const saved = localStorage.getItem('theme');
            if (saved) return saved === 'dark';
            return window.matchMedia('(prefers-color-scheme: dark)').matches;
        }
        return false;
    });

    useEffect(() => {
        localStorage.setItem('theme', isDark ? 'dark' : 'light');
        if (typeof document !== 'undefined') {
            const root = document.documentElement;
            root.classList.toggle('dark', isDark);
            root.classList.toggle('light', !isDark);
        }
    }, [isDark]);

    return (
        <ThemeContext.Provider value={{
            isDark,
            toggleTheme: () => setIsDark(prev => !prev),
            setTheme: setIsDark
        }}>
            {children}
        </ThemeContext.Provider>
    );
}
