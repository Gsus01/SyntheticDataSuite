import React from "react";

import {
  buildComponentGenerationFileDownloadUrl,
  previewComponentGenerationFile,
  type ComponentGenerationRunSnapshot,
} from "@/lib/component-generation";

import type { IntegrationSummaryFile, IntegrationSummaryView } from "./types";
import { Badge, Button, SectionTitle, cn } from "./ui/primitives";
import { FilePreviewModal } from "./ui/FilePreviewModal";

type IntegrationConfirmPanelProps = {
  run: ComponentGenerationRunSnapshot;
  summary: IntegrationSummaryView;
  decisionLoading: boolean;
  onApprove: () => void;
  onReject: () => void;
};

type PreviewState = {
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

type PreviewTarget = {
  componentName: string;
  file: IntegrationSummaryFile;
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

const INITIAL_PREVIEW_STATE: PreviewState = {
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

function isImageFile(path: string): boolean {
  const lower = path.toLowerCase();
  const extension = lower.includes(".") ? lower.slice(lower.lastIndexOf(".")) : "";
  return IMAGE_EXTENSIONS.has(extension);
}

export function IntegrationConfirmPanel({
  run,
  summary,
  decisionLoading,
  onApprove,
  onReject,
}: IntegrationConfirmPanelProps) {
  const [previewTarget, setPreviewTarget] = React.useState<PreviewTarget | null>(null);
  const [previewState, setPreviewState] = React.useState<PreviewState>(INITIAL_PREVIEW_STATE);
  const previewRequestRef = React.useRef(0);

  React.useEffect(() => {
    previewRequestRef.current += 1;
    setPreviewTarget(null);
    setPreviewState(INITIAL_PREVIEW_STATE);
  }, [run.runId]);

  const closePreview = React.useCallback(() => {
    previewRequestRef.current += 1;
    setPreviewTarget(null);
    setPreviewState(INITIAL_PREVIEW_STATE);
  }, []);

  const openPreview = React.useCallback(
    async (componentName: string, file: IntegrationSummaryFile) => {
      const requestId = previewRequestRef.current + 1;
      previewRequestRef.current = requestId;
      setPreviewTarget({ componentName, file });

      if (isImageFile(file.path)) {
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
    [run.runId]
  );

  return (
    <div className="min-h-0 flex-1 overflow-y-auto p-4 [scrollbar-gutter:stable]">
      <div className="rounded-xl border border-emerald-300 bg-emerald-50 p-4 shadow-sm dark:border-emerald-700 dark:bg-emerald-900/20">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <h3 className="text-[13px] font-semibold uppercase tracking-[0.12em] text-emerald-700 dark:text-emerald-300">
              Confirmación Final De Integración
            </h3>
            <p className="mt-1 text-[12px] text-slate-600 dark:text-slate-300">
              Revisa componentes y archivos generados. Al confirmar se compilarán las
              imágenes Docker y se registrarán los ComponentSpec en la plataforma.
            </p>
          </div>
          <Badge variant="waiting">Waiting approval</Badge>
        </div>

        <div className="mt-4 rounded-lg border border-slate-200 bg-white/70 p-3 text-[12px] dark:border-slate-700 dark:bg-slate-900/30">
          <div className="font-semibold text-slate-800 dark:text-slate-100">
            {summary.componentCount} componente(s) listo(s) para integración
          </div>
          <div className="mt-1 text-slate-600 dark:text-slate-300">
            Run ID: <span className="font-mono">{run.runId}</span>
          </div>
        </div>

        <div className="mt-4 space-y-3">
          {summary.components.length === 0 ? (
            <div className="rounded-md border border-slate-200 bg-white/60 p-3 text-[12px] text-slate-500 dark:border-slate-700 dark:bg-slate-900/30 dark:text-slate-400">
              No hay componentes para integrar.
            </div>
          ) : (
            summary.components.map((component) => (
              <section
                key={`${component.name}-${component.version}`}
                className="rounded-lg border border-slate-200 bg-white/80 p-3 dark:border-slate-700 dark:bg-slate-900/30"
              >
                <div className="flex flex-wrap items-center gap-2">
                  <SectionTitle className="m-0 text-[12px]">{component.title}</SectionTitle>
                  <span className="rounded-full border border-slate-300 bg-slate-100 px-2 py-0.5 font-mono text-[10px] text-slate-700 dark:border-slate-700 dark:bg-slate-800 dark:text-slate-200">
                    {component.name}
                  </span>
                  <span className="rounded-full border border-slate-300 bg-slate-100 px-2 py-0.5 text-[10px] text-slate-700 dark:border-slate-700 dark:bg-slate-800 dark:text-slate-200">
                    {component.type}
                  </span>
                  <span className="rounded-full border border-slate-300 bg-slate-100 px-2 py-0.5 text-[10px] text-slate-700 dark:border-slate-700 dark:bg-slate-800 dark:text-slate-200">
                    {component.version}
                  </span>
                </div>
                <div className="mt-1 text-[11px] text-slate-600 dark:text-slate-300">
                  Imagen: <span className="font-mono">{component.image}</span>
                </div>
                {component.description ? (
                  <p className="mt-1 text-[11px] text-slate-600 dark:text-slate-300">
                    {component.description}
                  </p>
                ) : null}

                <div className="mt-2">
                  <div className="mb-1 text-[10px] font-semibold uppercase tracking-wide text-slate-500 dark:text-slate-400">
                    Ficheros ({component.files.length})
                  </div>
                  <div className="max-h-36 space-y-1 overflow-y-auto [scrollbar-gutter:stable]">
                    {component.files.map((file) => (
                      <button
                        key={`${component.name}-${file.name}-${file.path}`}
                        type="button"
                        onClick={() => void openPreview(component.name, file)}
                        className={cn(
                          "w-full rounded border border-slate-200 bg-white px-2 py-1 text-left text-[11px] transition hover:border-indigo-300 hover:bg-indigo-50",
                          "dark:border-slate-700 dark:bg-slate-900 dark:hover:border-indigo-700 dark:hover:bg-indigo-900/20"
                        )}
                      >
                        <div className="font-semibold text-slate-800 dark:text-slate-100">
                          {file.name}
                        </div>
                        <div className="mt-0.5 break-all font-mono text-[10px] text-slate-500 dark:text-slate-400">
                          {file.path}
                        </div>
                      </button>
                    ))}
                  </div>
                </div>
              </section>
            ))
          )}
        </div>

        <div className="mt-4 flex flex-wrap items-center justify-end gap-2 border-t border-emerald-200 pt-3 dark:border-emerald-800">
          <Button
            type="button"
            onClick={onReject}
            disabled={decisionLoading}
            variant="secondary"
            size="sm"
          >
            {decisionLoading ? "Procesando..." : "No subir ahora"}
          </Button>
          <Button
            type="button"
            onClick={onApprove}
            disabled={decisionLoading}
            variant="primary"
            size="sm"
          >
            {decisionLoading ? "Integrando..." : "Confirmar y subir a la plataforma"}
          </Button>
        </div>
      </div>

      {previewTarget ? (
        <FilePreviewModal
          isOpen={Boolean(previewTarget)}
          fileName={previewTarget.file.name}
          filePath={previewTarget.file.path}
          contentType={previewState.contentType}
          languageHint={previewState.languageHint}
          size={previewState.size}
          downloadUrl={buildComponentGenerationFileDownloadUrl(
            run.runId,
            previewTarget.file.path
          )}
          loading={previewState.loading}
          error={previewState.error}
          isImage={previewState.isImage || isImageFile(previewTarget.file.path)}
          isBinary={previewState.isBinary}
          content={previewState.content}
          truncated={previewState.truncated}
          onClose={closePreview}
        />
      ) : null}
    </div>
  );
}
