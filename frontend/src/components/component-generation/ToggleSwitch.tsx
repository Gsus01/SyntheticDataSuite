import type { ToggleSwitchProps } from "./types";
import { cn } from "./ui/primitives";

export function ToggleSwitch({
  label,
  checked,
  onChange,
  className,
  labelClassName,
}: ToggleSwitchProps) {
  return (
    <label
      className={cn(
        "flex cursor-pointer items-center justify-between gap-3 text-slate-700 dark:text-slate-200",
        className
      )}
    >
      <span className={cn("text-[12px]", labelClassName)}>{label}</span>
      <span className="relative inline-flex items-center">
        <input
          type="checkbox"
          checked={checked}
          onChange={(event) => onChange(event.target.checked)}
          className="peer sr-only"
        />
        <span className="h-5 w-10 rounded-full border border-slate-300 bg-slate-200 transition-colors duration-200 peer-checked:border-indigo-600 peer-checked:bg-indigo-600 peer-focus-visible:ring-2 peer-focus-visible:ring-indigo-500/40 peer-focus-visible:ring-offset-2 dark:border-slate-700 dark:bg-slate-800 dark:peer-checked:border-indigo-500 dark:peer-checked:bg-indigo-500" />
        <span className="pointer-events-none absolute left-0.5 top-0.5 size-4 rounded-full bg-white shadow transition-transform duration-200 peer-checked:translate-x-5 dark:bg-slate-100" />
      </span>
    </label>
  );
}
