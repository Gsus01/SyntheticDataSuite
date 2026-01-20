import { API_BASE } from "@/lib/api";
import { NODE_TYPES, type NodeTypeId } from "@/lib/flow-const";
import type { FlowNodePorts, NodeArtifactPort } from "@/types/flow";

export type CatalogArtifact = {
  name: string;
  path?: string | null;
  role?: string | null;
};

export type CatalogArtifacts = {
  inputs?: CatalogArtifact[];
  outputs?: CatalogArtifact[];
};

export type CatalogNodeTemplate = {
  name: string;
  type: string; // preprocessing | training | generation | input | output | ...
  parameters: string[];
  artifacts: CatalogArtifacts;
  limits?: Record<string, unknown>;
  version?: string;
  parameter_defaults?: Record<string, unknown>;
};

const CONFIG_EXTENSIONS = [".yaml", ".yml", ".json"];

export function isConfigArtifact(spec: CatalogArtifact): boolean {
  if (spec.role && spec.role.toLowerCase() === "config") return true;
  const name = (spec.name || "").toLowerCase();
  const path = (spec.path || "").toLowerCase();
  if (name === "processed-data") return false;
  if (name.includes("config")) return true;
  if (path.includes("/config/")) return true;
  return CONFIG_EXTENSIONS.some((ext) => path.endsWith(ext));
}

function normalizePorts(list?: CatalogArtifact[]): NodeArtifactPort[] {
  if (!Array.isArray(list)) return [];
  return list
    .filter((item): item is CatalogArtifact => Boolean(item?.name))
    .map((item) => ({
      name: item.name,
      path: item.path ?? null,
    }));
}

export function computeConnectablePorts(artifacts?: CatalogArtifacts | null): FlowNodePorts {
  const inputs = normalizePorts(artifacts?.inputs).filter((spec) => !isConfigArtifact(spec));
  const outputs = normalizePorts(artifacts?.outputs);
  return { inputs, outputs };
}

export function inferNodeType(template: CatalogNodeTemplate): NodeTypeId {
  const inputs = template.artifacts?.inputs?.length ?? 0;
  const outputs = template.artifacts?.outputs?.length ?? 0;
  if (inputs === 0 && outputs > 0) return NODE_TYPES.nodeInput;
  if (inputs > 0 && outputs === 0) return NODE_TYPES.nodeOutput;
  return NODE_TYPES.nodeDefault;
}

export function buildTemplateIndex(
  templates: CatalogNodeTemplate[]
): Record<string, CatalogNodeTemplate> {
  return templates.reduce<Record<string, CatalogNodeTemplate>>((acc, template) => {
    acc[template.name] = template;
    return acc;
  }, {});
}

export async function fetchNodeTemplates(): Promise<CatalogNodeTemplate[]> {
  const response = await fetch(`${API_BASE}/workflow-templates`);
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}`);
  }
  return (await response.json()) as CatalogNodeTemplate[];
}

