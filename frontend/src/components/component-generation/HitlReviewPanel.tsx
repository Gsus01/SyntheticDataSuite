import type { ReactNode } from "react";

import type { PlanPortView, PlanViewModel } from "./types";
import { Button, Label, Textarea, cn } from "./ui/primitives";

type HitlReviewPanelProps = {
  planViewModel: PlanViewModel;
  feedback: string;
  onFeedbackChange: (value: string) => void;
  onApprove: () => void;
  onRequestChanges: () => void;
  decisionLoading: boolean;
};

type ComponentTone = {
  shell: string;
  summary: string;
  chip: string;
};

const COMPONENT_TONES: ComponentTone[] = [
  {
    shell:
      "border-sky-300 bg-sky-50/50 dark:border-sky-700 dark:bg-sky-900/20",
    summary:
      "border-sky-200 bg-sky-50/70 dark:border-sky-800 dark:bg-sky-900/30",
    chip: "border-sky-300 bg-sky-100 text-sky-700 dark:border-sky-700 dark:bg-sky-900/40 dark:text-sky-300",
  },
  {
    shell:
      "border-emerald-300 bg-emerald-50/50 dark:border-emerald-700 dark:bg-emerald-900/20",
    summary:
      "border-emerald-200 bg-emerald-50/70 dark:border-emerald-800 dark:bg-emerald-900/30",
    chip:
      "border-emerald-300 bg-emerald-100 text-emerald-700 dark:border-emerald-700 dark:bg-emerald-900/40 dark:text-emerald-300",
  },
  {
    shell:
      "border-violet-300 bg-violet-50/50 dark:border-violet-700 dark:bg-violet-900/20",
    summary:
      "border-violet-200 bg-violet-50/70 dark:border-violet-800 dark:bg-violet-900/30",
    chip: "border-violet-300 bg-violet-100 text-violet-700 dark:border-violet-700 dark:bg-violet-900/40 dark:text-violet-300",
  },
  {
    shell:
      "border-rose-300 bg-rose-50/50 dark:border-rose-700 dark:bg-rose-900/20",
    summary:
      "border-rose-200 bg-rose-50/70 dark:border-rose-800 dark:bg-rose-900/30",
    chip: "border-rose-300 bg-rose-100 text-rose-700 dark:border-rose-700 dark:bg-rose-900/40 dark:text-rose-300",
  },
  {
    shell:
      "border-cyan-300 bg-cyan-50/50 dark:border-cyan-700 dark:bg-cyan-900/20",
    summary:
      "border-cyan-200 bg-cyan-50/70 dark:border-cyan-800 dark:bg-cyan-900/30",
    chip: "border-cyan-300 bg-cyan-100 text-cyan-700 dark:border-cyan-700 dark:bg-cyan-900/40 dark:text-cyan-300",
  },
];

function SectionCard({
  title,
  children,
  className,
}: {
  title: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <div
      className={cn(
        "rounded-lg border border-slate-200 bg-slate-50/80 p-3 dark:border-slate-700 dark:bg-slate-900/40",
        className
      )}
    >
      <div className="text-[10px] font-semibold uppercase tracking-[0.12em] text-slate-600 dark:text-slate-300">
        {title}
      </div>
      <div className="mt-1.5">{children}</div>
    </div>
  );
}

function FieldList({
  items,
  emptyText,
}: {
  items: Array<{ key: string; value: string }>;
  emptyText: string;
}) {
  if (items.length === 0) {
    return (
      <p className="text-[11px] text-slate-500 dark:text-slate-400">{emptyText}</p>
    );
  }

  return (
    <div className="space-y-1">
      {items.map((field) => (
        <div
          key={field.key}
          className="rounded border border-slate-200 bg-white px-2 py-1 text-[11px] text-slate-700 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-300"
        >
          <span className="font-semibold text-slate-600 dark:text-slate-200">
            {field.key}:
          </span>{" "}
          {field.value}
        </div>
      ))}
    </div>
  );
}

function PortCard({
  port,
  kind,
}: {
  port: PlanPortView;
  kind: "input" | "output";
}) {
  const isInput = kind === "input";

  return (
    <div
      className={cn(
        "rounded-md border px-2.5 py-2",
        isInput
          ? "border-sky-200 bg-sky-50/70 dark:border-sky-800 dark:bg-sky-900/25"
          : "border-emerald-200 bg-emerald-50/70 dark:border-emerald-800 dark:bg-emerald-900/25"
      )}
    >
      <div className="flex flex-wrap items-center justify-between gap-2">
        <span className="text-[11px] font-semibold text-slate-800 dark:text-slate-100">
          {port.name}
        </span>
        <span
          className={cn(
            "rounded-full border px-2 py-0.5 text-[10px] font-medium uppercase tracking-wide",
            isInput
              ? "border-sky-300 bg-sky-100 text-sky-700 dark:border-sky-700 dark:bg-sky-900/45 dark:text-sky-300"
              : "border-emerald-300 bg-emerald-100 text-emerald-700 dark:border-emerald-700 dark:bg-emerald-900/45 dark:text-emerald-300"
          )}
        >
          {port.role}
        </span>
      </div>
      <div className="mt-1 rounded border border-black/10 bg-white/70 px-2 py-1 font-mono text-[11px] text-slate-700 dark:border-white/10 dark:bg-slate-950/40 dark:text-slate-200">
        {port.path}
      </div>
      {port.extraFields.length > 0 ? (
        <div className="mt-1.5 border-t border-black/10 pt-1.5 dark:border-white/10">
          <FieldList items={port.extraFields} emptyText="" />
        </div>
      ) : null}
    </div>
  );
}

function PortGroup({
  kind,
  ports,
}: {
  kind: "input" | "output";
  ports: PlanPortView[];
}) {
  const isInput = kind === "input";
  const label = isInput ? "Inputs" : "Outputs";

  return (
    <SectionCard
      title={
        <div className="flex items-center justify-between gap-2">
          <span>{label}</span>
          <span
            className={cn(
              "rounded-full border px-1.5 py-0.5 text-[10px] font-semibold",
              isInput
                ? "border-sky-300 bg-sky-100 text-sky-700 dark:border-sky-700 dark:bg-sky-900/40 dark:text-sky-300"
                : "border-emerald-300 bg-emerald-100 text-emerald-700 dark:border-emerald-700 dark:bg-emerald-900/40 dark:text-emerald-300"
            )}
          >
            {ports.length}
          </span>
        </div>
      }
      className={
        isInput
          ? "border-sky-200 bg-sky-50/40 dark:border-sky-800 dark:bg-sky-900/15"
          : "border-emerald-200 bg-emerald-50/40 dark:border-emerald-800 dark:bg-emerald-900/15"
      }
    >
      {ports.length === 0 ? (
        <p className="text-[11px] text-slate-500 dark:text-slate-400">
          {isInput ? "Sin inputs." : "Sin outputs."}
        </p>
      ) : (
        <div className="space-y-2">
          {ports.map((port, index) => (
            <PortCard key={`${kind}-${index}-${port.name}`} port={port} kind={kind} />
          ))}
        </div>
      )}
    </SectionCard>
  );
}

export function HitlReviewPanel({
  planViewModel,
  feedback,
  onFeedbackChange,
  onApprove,
  onRequestChanges,
  decisionLoading,
}: HitlReviewPanelProps) {
  return (
    <div className="min-h-0 flex-1 overflow-y-auto p-4 [scrollbar-gutter:stable]">
      <div className="rounded-xl border border-amber-300 bg-amber-50 p-4 shadow-sm dark:border-amber-700 dark:bg-amber-900/20">
        <div className="flex flex-wrap items-start justify-between gap-2">
          <div>
            <h3 className="text-[13px] font-semibold uppercase tracking-[0.12em] text-amber-700 dark:text-amber-300">
              Revisión HITL requerida
            </h3>
            <p className="mt-1 text-[12px] text-slate-600 dark:text-slate-300">
              El plan requiere decisión manual. Revisa campos, aprueba o solicita
              cambios.
            </p>
          </div>
          <span className="rounded-full border border-amber-400 bg-amber-100 px-2.5 py-1 text-[10px] font-semibold uppercase tracking-[0.12em] text-amber-700 dark:border-amber-700 dark:bg-amber-900/40 dark:text-amber-300">
            Waiting approval
          </span>
        </div>

        <div className="mt-4 space-y-3">
          <SectionCard title="Rationale">
            <p className="whitespace-pre-wrap text-[12px] leading-relaxed text-slate-700 dark:text-slate-200">
              {planViewModel.rationale}
            </p>
          </SectionCard>

          <SectionCard title="Assumptions">
            {planViewModel.assumptions.length === 0 ? (
              <p className="text-[12px] text-slate-500 dark:text-slate-400">
                Sin assumptions.
              </p>
            ) : (
              <ul className="space-y-1.5 text-[12px] text-slate-700 dark:text-slate-200">
                {planViewModel.assumptions.map((assumption, index) => (
                  <li
                    key={`assumption-${index}`}
                    className="rounded-md border border-slate-200 bg-white px-2.5 py-1.5 dark:border-slate-700 dark:bg-slate-900"
                  >
                    {assumption}
                  </li>
                ))}
              </ul>
            )}
          </SectionCard>

          <SectionCard title={`Components (${planViewModel.components.length})`}>
            {planViewModel.components.length === 0 ? (
              <p className="text-[12px] text-slate-500 dark:text-slate-400">
                El plan no incluye componentes.
              </p>
            ) : (
              <div className="space-y-3">
                {planViewModel.components.map((component, index) => {
                  const tone = COMPONENT_TONES[index % COMPONENT_TONES.length];
                  return (
                    <details
                      key={`${component.name}-${index}`}
                      className={cn("group rounded-xl border p-3", tone.shell)}
                    >
                      <summary className="cursor-pointer list-none">
                        <div className="flex flex-wrap items-center justify-between gap-2">
                          <div>
                            <div className="flex items-center gap-2 text-[13px] font-semibold text-slate-800 dark:text-slate-100">
                              <span
                                className={cn(
                                  "inline-flex min-w-7 items-center justify-center rounded-full border px-1.5 py-0.5 text-[10px] font-semibold",
                                  tone.chip
                                )}
                              >
                                #{index + 1}
                              </span>
                              <span>{component.title}</span>
                            </div>
                            <div className="mt-0.5 flex flex-wrap gap-1.5">
                              <span
                                className={cn(
                                  "rounded-full border px-2 py-0.5 text-[10px] font-medium",
                                  tone.chip
                                )}
                              >
                                {component.inputs.length} inputs
                              </span>
                              <span
                                className={cn(
                                  "rounded-full border px-2 py-0.5 text-[10px] font-medium",
                                  tone.chip
                                )}
                              >
                                {component.outputs.length} outputs
                              </span>
                              <span
                                className={cn(
                                  "rounded-full border px-2 py-0.5 text-[10px] font-medium",
                                  tone.chip
                                )}
                              >
                                {component.parameters.length} params
                              </span>
                            </div>
                          </div>
                          <span className="inline-flex size-6 items-center justify-center rounded-full border border-slate-300 bg-slate-100 text-slate-600 dark:border-slate-700 dark:bg-slate-800 dark:text-slate-300">
                            <svg
                              viewBox="0 0 20 20"
                              fill="currentColor"
                              className="size-4 transition-transform group-open:rotate-180"
                              aria-hidden="true"
                            >
                              <path
                                fillRule="evenodd"
                                d="M5.23 7.21a.75.75 0 0 1 1.06.02L10 11.156l3.71-3.925a.75.75 0 1 1 1.08 1.04l-4.25 4.5a.75.75 0 0 1-1.08 0l-4.25-4.5a.75.75 0 0 1 .02-1.06Z"
                                clipRule="evenodd"
                              />
                            </svg>
                          </span>
                        </div>
                      </summary>

                      <SectionCard title="Resumen del componente" className={tone.summary}>
                        <div className="space-y-2 text-[12px]">
                          <div className="flex flex-wrap items-center gap-2">
                            <span className="rounded border border-slate-300 bg-white px-2 py-0.5 font-mono text-[11px] text-slate-800 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-100">
                              {component.name}
                            </span>
                            <span className="rounded border border-slate-300 bg-white px-2 py-0.5 text-[11px] font-medium text-slate-700 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-200">
                              {component.type}
                            </span>
                          </div>
                          <p className="whitespace-pre-wrap leading-relaxed text-slate-700 dark:text-slate-200">
                            {component.description}
                          </p>
                        </div>
                      </SectionCard>

                      <div className="mt-3 grid grid-cols-1 gap-3 xl:grid-cols-2">
                        <PortGroup kind="input" ports={component.inputs} />
                        <PortGroup kind="output" ports={component.outputs} />
                      </div>

                      <div className="mt-3 grid grid-cols-1 gap-3 xl:grid-cols-2">
                        <SectionCard title={`Parameters defaults (${component.parameters.length})`}>
                          {component.parameters.length === 0 ? (
                            <p className="text-[11px] text-slate-500 dark:text-slate-400">
                              Sin parámetros.
                            </p>
                          ) : (
                            <div className="space-y-1">
                              {component.parameters.map((param) => (
                                <div
                                  key={`param-${index}-${param.key}`}
                                  className="rounded border border-slate-200 bg-white px-2 py-1 dark:border-slate-700 dark:bg-slate-900"
                                >
                                  <div className="font-mono text-[11px] text-slate-800 dark:text-slate-100">
                                    {param.key}
                                  </div>
                                  <pre className="mt-0.5 whitespace-pre-wrap break-words font-mono text-[10px] text-slate-600 dark:text-slate-300">
                                    {param.value}
                                  </pre>
                                </div>
                              ))}
                            </div>
                          )}
                        </SectionCard>

                        <SectionCard title={`Notes (${component.notes.length})`}>
                          {component.notes.length === 0 ? (
                            <p className="text-[11px] text-slate-500 dark:text-slate-400">
                              Sin notas.
                            </p>
                          ) : (
                            <ul className="space-y-1">
                              {component.notes.map((note, noteIndex) => (
                                <li
                                  key={`note-${index}-${noteIndex}`}
                                  className="rounded border border-slate-200 bg-white px-2 py-1 text-[11px] text-slate-700 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-300"
                                >
                                  {note}
                                </li>
                              ))}
                            </ul>
                          )}
                        </SectionCard>
                      </div>

                      {component.extraFields.length > 0 ? (
                        <div className="mt-3">
                          <SectionCard title="Extra fields">
                            <FieldList items={component.extraFields} emptyText="" />
                          </SectionCard>
                        </div>
                      ) : null}
                    </details>
                  );
                })}
              </div>
            )}
          </SectionCard>

          {planViewModel.extraFields.length > 0 ? (
            <SectionCard title="Plan extra fields">
              <FieldList items={planViewModel.extraFields} emptyText="" />
            </SectionCard>
          ) : null}
        </div>

        <label className="mt-4 block">
          <Label>Feedback para revisar plan</Label>
          <Textarea
            value={feedback}
            onChange={(event) => onFeedbackChange(event.target.value)}
            rows={5}
            placeholder="Qué debería cambiar el agente..."
          />
        </label>

        <div className="mt-3 flex gap-2">
          <Button type="button" onClick={onApprove} disabled={decisionLoading} variant="success">
            Aprobar plan
          </Button>
          <Button
            type="button"
            onClick={onRequestChanges}
            disabled={decisionLoading}
            variant="warning"
          >
            Solicitar cambios
          </Button>
        </div>
      </div>
    </div>
  );
}
