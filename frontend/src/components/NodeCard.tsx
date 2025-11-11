"use client";

import React from "react";
import type { WorkflowNodeRuntimeStatus } from "@/types/flow";

type Variant = "input" | "default" | "output";
export type NodeTone = "input" | "preprocessing" | "training" | "generation" | "output" | "other";

export type NodeCardProps = {
  label: string;
  variant: Variant;
  tone?: NodeTone; // visual color theme, independent of handles
  compact?: boolean;
  selected?: boolean;
  status?: WorkflowNodeRuntimeStatus;
};

const variantClasses: Record<Variant, string> = {
  input:
    "border-blue-300 bg-blue-50/60 text-blue-800 dark:border-blue-400 dark:bg-blue-900/30 dark:text-blue-100",
  default:
    "border-gray-300 bg-white text-gray-800 dark:border-gray-600 dark:bg-gray-800 dark:text-gray-100",
  output:
    "border-emerald-300 bg-emerald-50/60 text-emerald-800 dark:border-emerald-400 dark:bg-emerald-900/30 dark:text-emerald-100",
};

const toneClasses: Record<NodeTone, string> = {
  input:
    "border-blue-300 bg-blue-50/60 text-blue-800 dark:border-blue-400 dark:bg-blue-900/30 dark:text-blue-100",
  preprocessing:
    "border-indigo-300 bg-indigo-50/60 text-indigo-800 dark:border-indigo-400 dark:bg-indigo-900/30 dark:text-indigo-100",
  training:
    "border-amber-300 bg-amber-50/60 text-amber-800 dark:border-amber-400 dark:bg-amber-900/30 dark:text-amber-100",
  generation:
    "border-emerald-300 bg-emerald-50/60 text-emerald-800 dark:border-emerald-400 dark:bg-emerald-900/30 dark:text-emerald-100",
  output:
    "border-emerald-300 bg-emerald-50/60 text-emerald-800 dark:border-emerald-400 dark:bg-emerald-900/30 dark:text-emerald-100",
  other:
    "border-gray-300 bg-gray-50/60 text-gray-800 dark:border-gray-500 dark:bg-gray-800/30 dark:text-gray-100",
};

const statusThemes: Record<
  string,
  {
    dotClass: string;
    label: string;
  }
> = {
  pending: {
    dotClass: "bg-slate-400/70 shadow-sm",
    label: "Pendiente",
  },
  running: {
    dotClass: "bg-amber-400 shadow-md shadow-amber-400/50 animate-pulse",
    label: "En ejecución",
  },
  succeeded: {
    dotClass: "bg-emerald-500 shadow-md shadow-emerald-400/50",
    label: "Completado",
  },
  failed: {
    dotClass: "bg-rose-500 shadow-md shadow-rose-400/50",
    label: "Falló",
  },
  error: {
    dotClass: "bg-rose-500 shadow-md shadow-rose-400/50",
    label: "Error",
  },
  terminated: {
    dotClass: "bg-rose-500 shadow-md shadow-rose-400/50",
    label: "Cancelado",
  },
  skipped: {
    dotClass: "bg-slate-400/70 shadow-sm",
    label: "Omitido",
  },
  omitted: {
    dotClass: "bg-slate-400/70 shadow-sm",
    label: "Omitido",
  },
};

function resolveStatusTheme(status?: WorkflowNodeRuntimeStatus) {
  if (!status?.phase) {
    return null;
  }
  const key = status.phase.toLowerCase();
  const theme = statusThemes[key] ?? {
    dotClass: "bg-slate-400/70 shadow-sm",
    label: status.phase,
  };
  return {
    ...theme,
    phase: status.phase,
    message: status.message,
    tooltip: status.message ? `${theme.label}: ${status.message}` : theme.label,
  };
}

export default function NodeCard({ label, variant, tone, compact, selected, status }: NodeCardProps) {
  const colorClass = tone ? toneClasses[tone] ?? variantClasses[variant] : variantClasses[variant];
  const statusPresentation = resolveStatusTheme(status);
  const tooltip = statusPresentation?.tooltip;
  return (
    <div
      className={
        "select-none rounded-md border shadow-sm " +
        colorClass +
        " " +
        (compact
          ? "px-3 py-2 text-xs"
          : "px-4 py-2 text-sm sm:text-base") +
        (selected ? " ring-2 ring-blue-400/60" : "")
      }
    >
      <div className="flex items-center justify-between gap-3">
        <div className="min-w-0 truncate font-medium">{label}</div>
        {statusPresentation ? (
          <span
            className="relative flex h-3 w-3 shrink-0 items-center justify-center"
            title={tooltip}
            aria-label={statusPresentation.tooltip}
          >
            <span className={"h-3 w-3 rounded-full " + statusPresentation.dotClass} />
          </span>
        ) : null}
      </div>
    </div>
  );
}
