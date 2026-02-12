"use client";

import { useEffect } from "react";

export default function ComponentGenerationError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    console.error("component-generation route error", error);
  }, [error]);

  return (
    <div className="min-h-screen bg-slate-100 p-6 text-slate-900 dark:bg-slate-950 dark:text-slate-100">
      <div className="mx-auto mt-10 max-w-2xl rounded-2xl border border-slate-200 bg-white p-6 shadow-lg dark:border-slate-700 dark:bg-slate-900">
        <h2 className="text-lg font-semibold">Error en component generation</h2>
        <p className="mt-3 text-sm text-slate-600 dark:text-slate-300">
          Ocurrió un error inesperado al renderizar esta ruta.
        </p>
        <pre className="mt-4 max-h-64 overflow-auto rounded border border-slate-200 bg-slate-50 p-3 font-mono text-xs text-slate-700 dark:border-slate-700 dark:bg-slate-950/60 dark:text-slate-300">
          {error.message}
        </pre>
        <button
          type="button"
          onClick={reset}
          className="mt-5 inline-flex h-9 items-center justify-center rounded-md border border-indigo-600 bg-indigo-600 px-4 text-sm font-semibold text-white transition hover:bg-indigo-500 dark:border-indigo-500 dark:bg-indigo-500 dark:hover:bg-indigo-400"
        >
          Reintentar
        </button>
      </div>
    </div>
  );
}
