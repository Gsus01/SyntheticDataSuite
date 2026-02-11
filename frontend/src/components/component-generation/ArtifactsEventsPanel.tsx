import React from "react";

import type {
  ComponentGenerationRunEvent,
  ComponentGenerationRunSnapshot,
} from "@/lib/component-generation";
import {
  buildComponentGenerationFileDownloadUrl,
  previewComponentGenerationFile,
} from "@/lib/component-generation";

import { FilePreviewModal } from "./ui/FilePreviewModal";
import { Badge, CodeBlock, Panel, SectionTitle, cn } from "./ui/primitives";
import { eventSummary, formatTimestamp, runStatusBadge } from "./utils";

type GeneratedFileKind =
  | "image"
  | "json"
  | "python"
  | "typescript"
  | "javascript"
  | "markdown"
  | "yaml"
  | "toml"
  | "ini"
  | "text"
  | "other";

type GeneratedFilePreview = {
  componentName: string;
  filename: string;
  path: string;
  kind: GeneratedFileKind;
};

type FilePreviewState = {
  loading: boolean;
  error: string | null;
  isImage: boolean;
  isBinary: boolean;
  content: string;
  truncated: boolean;
  contentType?: string | null;
  languageHint?: string | null;
  size?: number | null;
};

type ArtifactsEventsPanelProps = {
  run: ComponentGenerationRunSnapshot | null;
  events: ComponentGenerationRunEvent[];
  streamConnected: boolean;
};

const IMAGE_EXTENSIONS = new Set([
  ".png",
  ".jpg",
  ".jpeg",
  ".gif",
  ".webp",
  ".svg",
  ".bmp",
  ".tiff",
  ".tif",
]);

const KIND_LABEL: Record<GeneratedFileKind, string> = {
  image: "image",
  json: "json",
  python: "python",
  typescript: "ts",
  javascript: "js",
  markdown: "md",
  yaml: "yaml",
  toml: "toml",
  ini: "ini",
  text: "text",
  other: "file",
};

function getActiveStepLabel(run: ComponentGenerationRunSnapshot | null): string {
  if (!run) return "—";
  const waiting = Object.entries(run.nodeStates).find(
    ([, node]) => node.state === "waiting_decision"
  );
  if (waiting) return waiting[0];

  const running = Object.entries(run.nodeStates).find(
    ([, node]) => node.state === "running"
  );
  if (running) return running[0];
  return "sin paso activo";
}

function kindBadgeClass(kind: GeneratedFileKind): string {
  switch (kind) {
    case "image":
      return "border-cyan-300 bg-cyan-100 text-cyan-700 dark:border-cyan-700 dark:bg-cyan-900/40 dark:text-cyan-300";
    case "json":
      return "border-indigo-300 bg-indigo-100 text-indigo-700 dark:border-indigo-700 dark:bg-indigo-900/40 dark:text-indigo-300";
    case "python":
      return "border-sky-300 bg-sky-100 text-sky-700 dark:border-sky-700 dark:bg-sky-900/40 dark:text-sky-300";
    case "typescript":
    case "javascript":
      return "border-emerald-300 bg-emerald-100 text-emerald-700 dark:border-emerald-700 dark:bg-emerald-900/40 dark:text-emerald-300";
    case "markdown":
      return "border-violet-300 bg-violet-100 text-violet-700 dark:border-violet-700 dark:bg-violet-900/40 dark:text-violet-300";
    case "yaml":
    case "toml":
    case "ini":
      return "border-amber-300 bg-amber-100 text-amber-700 dark:border-amber-700 dark:bg-amber-900/40 dark:text-amber-300";
    case "text":
      return "border-slate-300 bg-slate-100 text-slate-700 dark:border-slate-700 dark:bg-slate-800 dark:text-slate-300";
    default:
      return "border-slate-300 bg-slate-100 text-slate-700 dark:border-slate-700 dark:bg-slate-800 dark:text-slate-300";
  }
}

function detectGeneratedFileKind(filename: string, path: string): GeneratedFileKind {
  const lowerFilename = filename.toLowerCase();
  const lowerPath = path.toLowerCase();
  const extension = lowerFilename.includes(".")
    ? lowerFilename.slice(lowerFilename.lastIndexOf("."))
    : lowerPath.includes(".")
      ? lowerPath.slice(lowerPath.lastIndexOf("."))
      : "";

  if (IMAGE_EXTENSIONS.has(extension)) return "image";
  if (lowerFilename === "dockerfile") return "other";
  if (extension === ".json" || extension === ".ipynb") return "json";
  if (extension === ".py") return "python";
  if (extension === ".ts" || extension === ".tsx") return "typescript";
  if (extension === ".js" || extension === ".jsx") return "javascript";
  if (extension === ".md") return "markdown";
  if (extension === ".yaml" || extension === ".yml") return "yaml";
  if (extension === ".toml") return "toml";
  if (extension === ".ini" || extension === ".cfg") return "ini";
  if (extension === ".txt" || extension === ".csv") return "text";
  return "other";
}

function flattenGeneratedFiles(
  generatedIndex: Record<string, Record<string, string>>
): GeneratedFilePreview[] {
  const items: GeneratedFilePreview[] = [];
  for (const [componentName, fileMap] of Object.entries(generatedIndex)) {
    for (const [filename, path] of Object.entries(fileMap || {})) {
      items.push({
        componentName,
        filename,
        path,
        kind: detectGeneratedFileKind(filename, path),
      });
    }
  }
  return items;
}

const INITIAL_PREVIEW_STATE: FilePreviewState = {
  loading: false,
  error: null,
  isImage: false,
  isBinary: false,
  content: "",
  truncated: false,
  contentType: null,
  languageHint: null,
  size: null,
};

export function ArtifactsEventsPanel({
  run,
  events,
  streamConnected,
}: ArtifactsEventsPanelProps) {
  const generatedCount = Object.keys(run?.generatedIndex || {}).length;
  const flattenedFiles = flattenGeneratedFiles(run?.generatedIndex || {});
  const lastEvent = events.length > 0 ? events[events.length - 1] : null;

  const [previewTarget, setPreviewTarget] = React.useState<GeneratedFilePreview | null>(null);
  const [previewState, setPreviewState] = React.useState<FilePreviewState>(
    INITIAL_PREVIEW_STATE
  );
  const previewRequestRef = React.useRef(0);

  const closePreviewModal = React.useCallback(() => {
    previewRequestRef.current += 1;
    setPreviewTarget(null);
    setPreviewState(INITIAL_PREVIEW_STATE);
  }, []);

  React.useEffect(() => {
    closePreviewModal();
  }, [run?.runId, closePreviewModal]);

  const openFilePreview = React.useCallback(
    async (file: GeneratedFilePreview) => {
      if (!run?.runId) return;

      const requestId = previewRequestRef.current + 1;
      previewRequestRef.current = requestId;
      setPreviewTarget(file);

      if (file.kind === "image") {
        setPreviewState({
          ...INITIAL_PREVIEW_STATE,
          isImage: true,
          languageHint: "image",
        });
        return;
      }

      setPreviewState({
        ...INITIAL_PREVIEW_STATE,
        loading: true,
      });

      try {
        const preview = await previewComponentGenerationFile(run.runId, file.path);
        if (previewRequestRef.current !== requestId) return;

        setPreviewState({
          loading: false,
          error: null,
          isImage:
            Boolean(preview.contentType) &&
            (preview.contentType || "").toLowerCase().startsWith("image/"),
          isBinary: preview.isBinary,
          content: preview.content || "",
          truncated: Boolean(preview.truncated),
          contentType: preview.contentType ?? null,
          languageHint: preview.languageHint ?? null,
          size: typeof preview.size === "number" ? preview.size : null,
        });
      } catch (error) {
        if (previewRequestRef.current !== requestId) return;
        setPreviewState({
          ...INITIAL_PREVIEW_STATE,
          error: error instanceof Error ? error.message : "No se pudo cargar el preview.",
        });
      }
    },
    [run?.runId]
  );

  return (
    <Panel>
      <SectionTitle>Actividad y artefactos</SectionTitle>
      {run ? (
        <div className="mt-2 space-y-3 text-xs">
          <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
            <div className="rounded-md border border-slate-200 bg-slate-50/80 p-2 dark:border-slate-700 dark:bg-slate-900/40">
              <div className="text-[10px] uppercase tracking-wide text-slate-500 dark:text-slate-400">
                Estado de ejecución
              </div>
              <div className="mt-1 flex items-center gap-2">
                <Badge variant={runStatusBadge(run.status)}>
                  {run.status.replace("_", " ")}
                </Badge>
                <Badge variant={streamConnected ? "success" : "neutral"}>
                  {streamConnected ? "stream online" : "stream offline"}
                </Badge>
              </div>
            </div>

            <div className="rounded-md border border-slate-200 bg-slate-50/80 p-2 dark:border-slate-700 dark:bg-slate-900/40">
              <div className="text-[10px] uppercase tracking-wide text-slate-500 dark:text-slate-400">
                Paso activo
              </div>
              <div className="mt-1 font-mono text-[12px] font-semibold text-slate-800 dark:text-slate-100">
                {getActiveStepLabel(run)}
              </div>
            </div>

            <div className="rounded-md border border-slate-200 bg-slate-50/80 p-2 dark:border-slate-700 dark:bg-slate-900/40">
              <div className="text-[10px] uppercase tracking-wide text-slate-500 dark:text-slate-400">
                Último evento
              </div>
              {lastEvent ? (
                <>
                  <div className="mt-1 truncate text-[12px] font-semibold text-slate-800 dark:text-slate-100">
                    {eventSummary(lastEvent)}
                  </div>
                  <div className="text-[11px] text-slate-500 dark:text-slate-400">
                    {formatTimestamp(lastEvent.timestamp)}
                  </div>
                </>
              ) : (
                <div className="mt-1 text-[11px] text-slate-500 dark:text-slate-400">
                  Sin eventos todavía.
                </div>
              )}
            </div>

            <div className="rounded-md border border-slate-200 bg-slate-50/80 p-2 dark:border-slate-700 dark:bg-slate-900/40">
              <div className="text-[10px] uppercase tracking-wide text-slate-500 dark:text-slate-400">
                Resumen de salida
              </div>
              <div className="mt-1 text-[12px] font-semibold text-slate-800 dark:text-slate-100">
                {generatedCount} componente(s) · review{" "}
                <span className="font-mono">{run.reviewStatus || "—"}</span>
              </div>
            </div>
          </div>

          <div className="rounded-md border border-slate-200 bg-slate-50/80 p-2 dark:border-slate-700 dark:bg-slate-900/40">
            <div className="mb-1 flex items-center justify-between gap-2">
              <div className="text-[10px] uppercase tracking-wide text-slate-500 dark:text-slate-400">
                Ficheros generados ({flattenedFiles.length})
              </div>
              <div className="text-[10px] text-slate-500 dark:text-slate-400">
                click para previsualizar
              </div>
            </div>
            {flattenedFiles.length === 0 ? (
              <p className="text-[11px] text-slate-500 dark:text-slate-400">
                Todavía no hay ficheros generados.
              </p>
            ) : (
              <div className="max-h-40 space-y-1 overflow-y-auto [scrollbar-gutter:stable]">
                {flattenedFiles.slice(0, 24).map((item) => (
                  <button
                    key={`${item.componentName}-${item.filename}-${item.path}`}
                    type="button"
                    onClick={() => void openFilePreview(item)}
                    className="w-full cursor-pointer rounded border border-slate-200 bg-white px-2 py-1 text-left transition hover:border-indigo-300 hover:bg-indigo-50 dark:border-slate-700 dark:bg-slate-900 dark:hover:border-indigo-700 dark:hover:bg-indigo-900/20"
                  >
                    <div className="flex flex-wrap items-center gap-1.5">
                      <span className="rounded border border-slate-300 bg-slate-100 px-1.5 py-0.5 text-[10px] uppercase text-slate-700 dark:border-slate-700 dark:bg-slate-800 dark:text-slate-200">
                        {item.componentName}
                      </span>
                      <span className="text-[11px] font-semibold text-slate-800 dark:text-slate-100">
                        {item.filename}
                      </span>
                      <span
                        className={cn(
                          "rounded-full border px-1.5 py-0.5 text-[9px] font-semibold uppercase",
                          kindBadgeClass(item.kind)
                        )}
                      >
                        {KIND_LABEL[item.kind]}
                      </span>
                    </div>
                    <div className="mt-0.5 break-all font-mono text-[10px] text-slate-500 dark:text-slate-400">
                      {item.path}
                    </div>
                  </button>
                ))}
              </div>
            )}
          </div>

          <details className="rounded-md border border-slate-200 bg-slate-50/80 p-2 dark:border-slate-700 dark:bg-slate-900/40">
            <summary className="cursor-pointer text-[10px] font-semibold uppercase tracking-wide text-slate-600 dark:text-slate-300">
              Reporte de revisión
            </summary>
            <CodeBlock className="mt-2 max-h-28 overflow-y-auto [scrollbar-gutter:stable]">
              {(run.reviewReport || "(sin reporte)").trim() || "(sin reporte)"}
            </CodeBlock>
          </details>

          <details className="rounded-md border border-red-200 bg-red-50/70 p-2 dark:border-red-800 dark:bg-red-900/20">
            <summary className="cursor-pointer text-[10px] font-semibold uppercase tracking-wide text-red-700 dark:text-red-300">
              Errores
            </summary>
            <CodeBlock className="mt-2 max-h-20 overflow-y-auto text-red-700 [scrollbar-gutter:stable] dark:text-red-300">
              {run.error || "(sin error)"}
            </CodeBlock>
          </details>

          {previewTarget ? (
            <FilePreviewModal
              isOpen={Boolean(previewTarget)}
              fileName={previewTarget.filename}
              filePath={previewTarget.path}
              contentType={previewState.contentType}
              languageHint={previewState.languageHint}
              size={previewState.size}
              downloadUrl={buildComponentGenerationFileDownloadUrl(
                run.runId,
                previewTarget.path
              )}
              loading={previewState.loading}
              error={previewState.error}
              isImage={previewState.isImage || previewTarget.kind === "image"}
              isBinary={previewState.isBinary}
              content={previewState.content}
              truncated={previewState.truncated}
              onClose={closePreviewModal}
            />
          ) : null}
        </div>
      ) : (
        <p className="mt-2 text-xs text-slate-500 dark:text-slate-400">
          Sin artefactos todavía.
        </p>
      )}
    </Panel>
  );
}
