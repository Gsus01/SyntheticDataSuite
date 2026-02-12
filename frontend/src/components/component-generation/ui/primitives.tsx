import React from "react";

type ClassValue = string | false | null | undefined;

function cn(...values: ClassValue[]): string {
  return values.filter(Boolean).join(" ");
}

const BASE_CONTROL_CLASS =
  "rounded-md border border-slate-300 bg-white px-2.5 py-1.5 text-xs text-slate-900 shadow-sm transition focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500/50 focus-visible:ring-offset-1 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-100";

type ControlSizingProps = {
  fullWidth?: boolean;
};

export function Input(
  props: React.InputHTMLAttributes<HTMLInputElement> & ControlSizingProps
) {
  const { className, fullWidth = true, ...rest } = props;
  return <input {...rest} className={cn(BASE_CONTROL_CLASS, fullWidth && "w-full", className)} />;
}

export function Select(
  props: React.SelectHTMLAttributes<HTMLSelectElement> & ControlSizingProps
) {
  const { className, fullWidth = true, ...rest } = props;
  return <select {...rest} className={cn(BASE_CONTROL_CLASS, fullWidth && "w-full", className)} />;
}

export function Textarea(
  props: React.TextareaHTMLAttributes<HTMLTextAreaElement> & ControlSizingProps
) {
  const { className, fullWidth = true, ...rest } = props;
  return <textarea {...rest} className={cn(BASE_CONTROL_CLASS, fullWidth && "w-full", className)} />;
}

export function Label({
  className,
  ...props
}: React.LabelHTMLAttributes<HTMLLabelElement>) {
  return (
    <label
      {...props}
      className={cn("mb-1 block text-[11px] font-medium text-slate-600 dark:text-slate-300", className)}
    />
  );
}

export type ButtonVariant =
  | "primary"
  | "secondary"
  | "danger"
  | "warning"
  | "success"
  | "ghost";

export type ButtonSize = "sm" | "md";

const BUTTON_VARIANTS: Record<ButtonVariant, string> = {
  primary:
    "border-indigo-600 bg-indigo-600 text-white hover:bg-indigo-500 dark:border-indigo-500 dark:bg-indigo-500 dark:hover:bg-indigo-400",
  secondary:
    "border-slate-300 bg-white text-slate-700 hover:bg-slate-100 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-200 dark:hover:bg-slate-800",
  danger:
    "border-red-600 bg-red-600 text-white hover:bg-red-500 dark:border-red-500 dark:bg-red-500 dark:hover:bg-red-400",
  warning:
    "border-amber-500 bg-amber-500 text-slate-950 hover:bg-amber-400 dark:border-amber-400 dark:bg-amber-400 dark:hover:bg-amber-300",
  success:
    "border-emerald-600 bg-emerald-600 text-white hover:bg-emerald-500 dark:border-emerald-500 dark:bg-emerald-500 dark:hover:bg-emerald-400",
  ghost:
    "border-transparent bg-transparent text-slate-600 hover:bg-slate-200 dark:text-slate-300 dark:hover:bg-slate-800",
};

const BUTTON_SIZES: Record<ButtonSize, string> = {
  sm: "h-8 px-2.5 text-[11px]",
  md: "h-9 px-3 text-xs",
};

type ButtonProps = React.ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: ButtonVariant;
  size?: ButtonSize;
};

export function Button({
  className,
  variant = "secondary",
  size = "md",
  ...props
}: ButtonProps) {
  return (
    <button
      {...props}
      className={cn(
        "inline-flex cursor-pointer items-center justify-center gap-1.5 rounded-md border font-semibold transition disabled:cursor-not-allowed disabled:opacity-50",
        BUTTON_VARIANTS[variant],
        BUTTON_SIZES[size],
        className
      )}
    />
  );
}

export type BadgeVariant =
  | "neutral"
  | "running"
  | "waiting"
  | "success"
  | "danger";

const BADGE_VARIANTS: Record<BadgeVariant, string> = {
  neutral:
    "border-slate-300 bg-slate-100 text-slate-700 dark:border-slate-700 dark:bg-slate-800 dark:text-slate-200",
  running:
    "border-sky-300 bg-sky-100 text-sky-700 dark:border-sky-700 dark:bg-sky-900/50 dark:text-sky-300",
  waiting:
    "border-amber-300 bg-amber-100 text-amber-700 dark:border-amber-700 dark:bg-amber-900/40 dark:text-amber-300",
  success:
    "border-emerald-300 bg-emerald-100 text-emerald-700 dark:border-emerald-700 dark:bg-emerald-900/40 dark:text-emerald-300",
  danger:
    "border-red-300 bg-red-100 text-red-700 dark:border-red-700 dark:bg-red-900/40 dark:text-red-300",
};

type BadgeProps = React.HTMLAttributes<HTMLSpanElement> & {
  variant?: BadgeVariant;
};

export function Badge({ className, variant = "neutral", ...props }: BadgeProps) {
  return (
    <span
      {...props}
      className={cn(
        "inline-flex rounded-full border px-2.5 py-1 text-[10px] font-semibold uppercase tracking-wide",
        BADGE_VARIANTS[variant],
        className
      )}
    />
  );
}

export function Panel({
  className,
  ...props
}: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      {...props}
      className={cn(
        "rounded-xl border border-slate-200 bg-white p-3 shadow-sm dark:border-slate-700 dark:bg-slate-900",
        className
      )}
    />
  );
}

export function SectionTitle({
  className,
  ...props
}: React.HTMLAttributes<HTMLHeadingElement>) {
  return (
    <h3
      {...props}
      className={cn(
        "text-xs font-semibold uppercase tracking-wide text-slate-600 dark:text-slate-300",
        className
      )}
    />
  );
}

export function CodeBlock({
  className,
  ...props
}: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      {...props}
      className={cn(
        "rounded-md border border-slate-200 bg-slate-50 p-2 font-mono text-[11px] text-slate-700 dark:border-slate-700 dark:bg-slate-950/60 dark:text-slate-300",
        className
      )}
    />
  );
}

export { cn };
