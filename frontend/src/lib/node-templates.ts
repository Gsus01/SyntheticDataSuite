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

export type DeleteCatalogComponentResponse = {
  name: string;
  version?: string | null;
  deleted: boolean;
  componentDeleted: boolean;
  activeVersion?: string | null;
};

export type CatalogComponentSummary = {
  name: string;
  activeVersion?: string | null;
  createdAt?: string | null;
  updatedAt?: string | null;
};

export type CatalogComponentVersion = {
  name: string;
  version: string;
  spec: Record<string, unknown>;
  createdAt?: string | null;
};

const CONFIG_EXTENSIONS = [".yaml", ".yml", ".json"];

export function isConfigArtifact(spec: CatalogArtifact): boolean {
  if (spec.role) {
    const role = spec.role.toLowerCase();
    if (role === "config") return true;
    return false;
  }
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
  const rawInputs = Array.isArray(artifacts?.inputs) ? artifacts?.inputs : [];
  const filteredInputs = rawInputs.filter((spec) => !isConfigArtifact(spec));
  const inputs = normalizePorts(filteredInputs);
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

export async function deleteCatalogComponent(
  templateName: string
): Promise<DeleteCatalogComponentResponse> {
  const response = await fetch(`${API_BASE}/components/${encodeURIComponent(templateName)}`, {
    method: "DELETE",
  });

  if (!response.ok) {
    const message = await response.text();
    throw new Error(message || `HTTP ${response.status}`);
  }

  return (await response.json()) as DeleteCatalogComponentResponse;
}

export async function listCatalogComponents(): Promise<CatalogComponentSummary[]> {
  const response = await fetch(`${API_BASE}/components`);
  if (!response.ok) {
    const message = await response.text();
    throw new Error(message || `HTTP ${response.status}`);
  }
  return (await response.json()) as CatalogComponentSummary[];
}

export async function getCatalogComponentVersions(
  templateName: string
): Promise<CatalogComponentVersion[]> {
  const response = await fetch(`${API_BASE}/components/${encodeURIComponent(templateName)}`);
  if (!response.ok) {
    const message = await response.text();
    throw new Error(message || `HTTP ${response.status}`);
  }
  return (await response.json()) as CatalogComponentVersion[];
}

export async function registerCatalogComponent(
  spec: Record<string, unknown>,
  activate = true
): Promise<CatalogComponentSummary> {
  const response = await fetch(`${API_BASE}/components`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      spec,
      activate,
    }),
  });
  if (!response.ok) {
    const message = await response.text();
    throw new Error(message || `HTTP ${response.status}`);
  }
  return (await response.json()) as CatalogComponentSummary;
}

export async function activateCatalogComponentVersion(
  templateName: string,
  version: string
): Promise<CatalogComponentSummary> {
  const response = await fetch(
    `${API_BASE}/components/${encodeURIComponent(templateName)}/${encodeURIComponent(version)}/activate`,
    {
      method: "POST",
    }
  );
  if (!response.ok) {
    const message = await response.text();
    throw new Error(message || `HTTP ${response.status}`);
  }
  return (await response.json()) as CatalogComponentSummary;
}
