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
    const [isDark, setIsDark] = useState(false);
    const [isThemeReady, setIsThemeReady] = useState(false);

    useEffect(() => {
        const saved = localStorage.getItem('theme');
        const initialTheme = saved
            ? saved === 'dark'
            : window.matchMedia('(prefers-color-scheme: dark)').matches;
        setIsDark(initialTheme);
        setIsThemeReady(true);
    }, []);

    useEffect(() => {
        if (!isThemeReady) return;

        localStorage.setItem('theme', isDark ? 'dark' : 'light');
        const root = document.documentElement;
        root.classList.toggle('dark', isDark);
        root.classList.toggle('light', !isDark);
    }, [isDark, isThemeReady]);

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
