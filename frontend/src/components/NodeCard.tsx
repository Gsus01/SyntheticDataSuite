"use client";

import React from "react";

type Variant = "input" | "default" | "output";
export type NodeTone = "input" | "preprocessing" | "training" | "generation" | "output" | "other";

export type NodeCardProps = {
  label: string;
  variant: Variant;
  tone?: NodeTone; // visual color theme, independent of handles
  compact?: boolean;
  selected?: boolean;
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

export default function NodeCard({ label, variant, tone, compact, selected }: NodeCardProps) {
  const colorClass = tone ? toneClasses[tone] ?? variantClasses[variant] : variantClasses[variant];
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
      <div className="font-medium">{label}</div>
    </div>
  );
}
