import React from "react";
import { createPortal } from "react-dom";

import { formatBytes } from "../utils";
import { Button, cn } from "./primitives";

type FilePreviewModalProps = {
  isOpen: boolean;
  fileName: string;
  filePath: string;
  contentType?: string | null;
  languageHint?: string | null;
  size?: number | null;
  downloadUrl: string;
  loading: boolean;
  error: string | null;
  isImage: boolean;
  isBinary: boolean;
  content: string;
  truncated: boolean;
  onClose: () => void;
};

type HighlightRule = {
  regex: RegExp;
  className: string;
};

function codeToneClass(languageHint?: string | null): string {
  switch ((languageHint || "").toLowerCase()) {
    case "python":
      return "text-sky-100";
    case "typescript":
    case "tsx":
    case "javascript":
    case "jsx":
      return "text-emerald-100";
    case "json":
      return "text-slate-100";
    case "yaml":
    case "toml":
    case "ini":
      return "text-amber-100";
    case "markdown":
      return "text-violet-100";
    default:
      return "text-slate-100";
  }
}

function languageBadgeClass(languageHint?: string | null): string {
  switch ((languageHint || "").toLowerCase()) {
    case "python":
      return "border-sky-300 bg-sky-100 text-sky-700 dark:border-sky-700 dark:bg-sky-900/40 dark:text-sky-300";
    case "typescript":
    case "tsx":
    case "javascript":
    case "jsx":
      return "border-emerald-300 bg-emerald-100 text-emerald-700 dark:border-emerald-700 dark:bg-emerald-900/40 dark:text-emerald-300";
    case "json":
      return "border-indigo-300 bg-indigo-100 text-indigo-700 dark:border-indigo-700 dark:bg-indigo-900/40 dark:text-indigo-300";
    case "yaml":
    case "toml":
    case "ini":
      return "border-amber-300 bg-amber-100 text-amber-700 dark:border-amber-700 dark:bg-amber-900/40 dark:text-amber-300";
    case "markdown":
      return "border-violet-300 bg-violet-100 text-violet-700 dark:border-violet-700 dark:bg-violet-900/40 dark:text-violet-300";
    default:
      return "border-slate-300 bg-slate-100 text-slate-700 dark:border-slate-700 dark:bg-slate-800 dark:text-slate-300";
  }
}

const PYTHON_RULES: HighlightRule[] = [
  { regex: /#.*$/g, className: "text-slate-500" },
  {
    regex: /"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'/g,
    className: "text-emerald-300",
  },
  {
    regex:
      /\b(?:def|class|if|elif|else|for|while|try|except|finally|return|import|from|as|with|pass|break|continue|lambda|yield|in|is|and|or|not|True|False|None)\b/g,
    className: "text-violet-300",
  },
  { regex: /\b\d+(?:\.\d+)?\b/g, className: "text-cyan-300" },
  { regex: /@[A-Za-z_]\w*/g, className: "text-amber-300" },
];

const JS_TS_RULES: HighlightRule[] = [
  { regex: /\/\/.*$/g, className: "text-slate-500" },
  { regex: /\/\*.*\*\//g, className: "text-slate-500" },
  {
    regex: /"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|`(?:\\.|[^`\\])*`/g,
    className: "text-emerald-300",
  },
  {
    regex:
      /\b(?:const|let|var|function|return|if|else|for|while|switch|case|break|continue|try|catch|finally|class|extends|new|import|from|export|default|async|await|type|interface|implements|public|private|protected|readonly|enum|as|in|of|true|false|null|undefined)\b/g,
    className: "text-violet-300",
  },
  { regex: /\b\d+(?:\.\d+)?\b/g, className: "text-cyan-300" },
];

const YAML_TOML_INI_RULES: HighlightRule[] = [
  { regex: /#.*$/g, className: "text-slate-500" },
  { regex: /;.*$/g, className: "text-slate-500" },
  {
    regex: /"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'/g,
    className: "text-emerald-300",
  },
  { regex: /^\s*[\w.-]+(?=\s*[:=])/g, className: "text-sky-300" },
  { regex: /\b(?:true|false|null|yes|no|on|off)\b/gi, className: "text-violet-300" },
  { regex: /\b\d+(?:\.\d+)?\b/g, className: "text-cyan-300" },
];

const MARKDOWN_RULES: HighlightRule[] = [
  { regex: /^\s{0,3}#{1,6}\s.+$/g, className: "text-sky-300" },
  { regex: /^\s*[-*+]\s.+$/g, className: "text-violet-300" },
  { regex: /^\s*\d+\.\s.+$/g, className: "text-violet-300" },
  { regex: /`[^`]+`/g, className: "text-emerald-300" },
  { regex: /\[[^\]]+\]\([^)]+\)/g, className: "text-cyan-300" },
  { regex: /\*\*[^*]+\*\*|\*[^*]+\*|_[^_]+_/g, className: "text-amber-300" },
];

function renderLineWithRules(line: string, rules: HighlightRule[]): React.ReactNode[] {
  if (!line) {
    return [<span key="empty-line">{line}</span>];
  }

  const nodes: React.ReactNode[] = [];
  let cursor = 0;
  let tokenId = 0;

  while (cursor < line.length) {
    let earliest:
      | { start: number; end: number; className: string; text: string }
      | null = null;

    for (const rule of rules) {
      rule.regex.lastIndex = cursor;
      const match = rule.regex.exec(line);
      if (!match) continue;

      const start = match.index;
      const text = match[0];
      const end = start + text.length;
      if (text.length === 0) continue;

      if (!earliest || start < earliest.start) {
        earliest = { start, end, className: rule.className, text };
      }
    }

    if (!earliest) {
      nodes.push(
        <span key={`plain-tail-${cursor}`}>{line.slice(cursor)}</span>
      );
      break;
    }

    if (earliest.start > cursor) {
      nodes.push(
        <span key={`plain-${tokenId}-${cursor}`}>
          {line.slice(cursor, earliest.start)}
        </span>
      );
    }

    nodes.push(
      <span key={`token-${tokenId}-${earliest.start}`} className={earliest.className}>
        {earliest.text}
      </span>
    );
    tokenId += 1;
    cursor = earliest.end;
  }

  return nodes;
}

function renderJsonLine(line: string): React.ReactNode[] {
  const tokenRegex =
    /("(\\u[\da-fA-F]{4}|\\[^u]|[^\\"])*"(\s*:)?|\btrue\b|\bfalse\b|\bnull\b|-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)/g;
  const nodes: React.ReactNode[] = [];
  let lastIndex = 0;
  let match = tokenRegex.exec(line);
  let tokenIndex = 0;

  while (match) {
    if (match.index > lastIndex) {
      nodes.push(
        <span key={`plain-${tokenIndex}-${lastIndex}`}>
          {line.slice(lastIndex, match.index)}
        </span>
      );
    }

    const token = match[0];
    let className = "text-emerald-300";
    if (token.startsWith('"')) {
      className = token.endsWith(":") ? "text-sky-300" : "text-emerald-300";
    } else if (token === "true" || token === "false") {
      className = "text-violet-300";
    } else if (token === "null") {
      className = "text-amber-300";
    } else {
      className = "text-cyan-300";
    }

    nodes.push(
      <span key={`token-${tokenIndex}-${match.index}`} className={className}>
        {token}
      </span>
    );
    tokenIndex += 1;
    lastIndex = match.index + token.length;
    match = tokenRegex.exec(line);
  }

  if (lastIndex < line.length) {
    nodes.push(<span key={`tail-${lastIndex}`}>{line.slice(lastIndex)}</span>);
  }
  if (nodes.length === 0) {
    nodes.push(<span key="empty">{line}</span>);
  }
  return nodes;
}

function renderHighlightedLine(
  line: string,
  languageHint?: string | null
): React.ReactNode[] {
  const language = (languageHint || "").toLowerCase();

  if (language === "json") {
    return renderJsonLine(line);
  }

  if (language === "python") {
    return renderLineWithRules(line, PYTHON_RULES);
  }

  if (
    language === "typescript" ||
    language === "tsx" ||
    language === "javascript" ||
    language === "jsx"
  ) {
    return renderLineWithRules(line, JS_TS_RULES);
  }

  if (language === "yaml" || language === "toml" || language === "ini") {
    return renderLineWithRules(line, YAML_TOML_INI_RULES);
  }

  if (language === "markdown") {
    return renderLineWithRules(line, MARKDOWN_RULES);
  }

  return [<span key="plain-default">{line}</span>];
}

function TextPreview({
  content,
  languageHint,
}: {
  content: string;
  languageHint?: string | null;
}) {
  let normalizedContent = content;
  if ((languageHint || "").toLowerCase() === "json") {
    try {
      normalizedContent = JSON.stringify(JSON.parse(content), null, 2);
    } catch {
      // Keep raw text if it is not valid JSON.
    }
  }
  const lines = normalizedContent.split("\n");

  return (
    <pre
      className={cn(
        "overflow-auto p-3 font-mono text-[12px] leading-relaxed",
        codeToneClass(languageHint)
      )}
    >
      {lines.map((line, index) => (
        <div key={`line-${index}`}>{renderHighlightedLine(line, languageHint)}</div>
      ))}
    </pre>
  );
}

export function FilePreviewModal({
  isOpen,
  fileName,
  filePath,
  contentType,
  languageHint,
  size,
  downloadUrl,
  loading,
  error,
  isImage,
  isBinary,
  content,
  truncated,
  onClose,
}: FilePreviewModalProps) {
  React.useEffect(() => {
    if (!isOpen) return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        onClose();
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [isOpen, onClose]);

  if (!isOpen || typeof document === "undefined") {
    return null;
  }

  return createPortal(
    <div
      className="fixed inset-0 z-[120] flex items-center justify-center bg-slate-950/70 px-3 py-4 backdrop-blur-sm"
      role="dialog"
      aria-modal="true"
      aria-label={`Vista previa de ${fileName}`}
      onClick={onClose}
    >
      <div
        className="flex h-[min(86vh,56rem)] w-[min(95vw,72rem)] flex-col overflow-hidden rounded-xl border border-slate-300 bg-white shadow-2xl dark:border-slate-700 dark:bg-slate-900"
        onClick={(event) => event.stopPropagation()}
      >
        <header className="border-b border-slate-200 px-3 py-2 dark:border-slate-700">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div className="min-w-0">
              <div className="truncate text-sm font-semibold text-slate-900 dark:text-slate-100">
                {fileName}
              </div>
              <div className="truncate font-mono text-[10px] text-slate-500 dark:text-slate-400">
                {filePath}
              </div>
            </div>
            <div className="flex items-center gap-1.5">
              {languageHint ? (
                <span
                  className={cn(
                    "rounded-full border px-2 py-0.5 text-[10px] font-semibold uppercase",
                    languageBadgeClass(languageHint)
                  )}
                >
                  {languageHint}
                </span>
              ) : null}
              {contentType ? (
                <span className="rounded-full border border-slate-300 bg-slate-100 px-2 py-0.5 text-[10px] font-semibold text-slate-600 dark:border-slate-700 dark:bg-slate-800 dark:text-slate-300">
                  {contentType}
                </span>
              ) : null}
              {typeof size === "number" ? (
                <span className="rounded-full border border-slate-300 bg-slate-100 px-2 py-0.5 text-[10px] font-semibold text-slate-600 dark:border-slate-700 dark:bg-slate-800 dark:text-slate-300">
                  {formatBytes(size)}
                </span>
              ) : null}
            </div>
          </div>
        </header>

        <div className="min-h-0 flex-1 overflow-auto bg-slate-950">
          {loading ? (
            <div className="flex h-full items-center justify-center text-sm text-slate-300">
              Cargando preview...
            </div>
          ) : error ? (
            <div className="p-4 text-sm text-red-300">{error}</div>
          ) : isImage ? (
            <div className="flex h-full items-center justify-center p-3">
              {/* Runtime URL preview, no Next/Image optimization here. */}
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={downloadUrl}
                alt={`Preview de ${fileName}`}
                className="max-h-full max-w-full rounded border border-slate-700 bg-slate-900 object-contain"
              />
            </div>
          ) : isBinary ? (
            <div className="p-4 text-sm text-slate-300">
              Este fichero no es previsualizable como texto.
            </div>
          ) : (
            <TextPreview content={content} languageHint={languageHint} />
          )}
        </div>

        <footer className="flex items-center justify-between gap-2 border-t border-slate-200 px-3 py-2 dark:border-slate-700">
          <div className="text-[11px] text-slate-500 dark:text-slate-400">
            {truncated ? "Contenido truncado en preview." : "Preview completa."}
          </div>
          <div className="flex items-center gap-2">
            <a
              href={downloadUrl}
              className="inline-flex h-8 cursor-pointer items-center justify-center rounded-md border border-slate-300 bg-white px-2.5 text-[11px] font-semibold text-slate-700 transition hover:bg-slate-100 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-200 dark:hover:bg-slate-800"
              download
            >
              Descargar
            </a>
            <Button type="button" variant="secondary" size="sm" onClick={onClose}>
              Cerrar
            </Button>
          </div>
        </footer>
      </div>
    </div>,
    document.body
  );
}
