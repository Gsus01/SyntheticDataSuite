import type { Edge, Node } from "reactflow";
import type { FlowNodeData } from "@/types/flow";
import { API_BASE } from "@/lib/api";

type ExportRequest = {
  sessionId: string;
  nodes: Node<FlowNodeData>[];
  edges: Edge[];
};

function serializeNodes(nodes: Node<FlowNodeData>[]) {
  return nodes.map(({ id, type, data }) => ({ id, type, data }));
}

function serializeEdges(edges: Edge[]) {
  return edges.map(({ id, source, target }) => ({ id, source, target }));
}

export async function exportWorkflowYAML(request: ExportRequest): Promise<void> {
  const response = await fetch(`${API_BASE}/workflow/render`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      sessionId: request.sessionId,
      nodes: serializeNodes(request.nodes),
      edges: serializeEdges(request.edges),
    }),
  });

  if (!response.ok) {
    const message = await response.text();
    throw new Error(message || `HTTP ${response.status}`);
  }

  const { filename, yaml } = (await response.json()) as { filename: string; yaml: string };

  const blob = new Blob([yaml], { type: "application/x-yaml" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  document.body.removeChild(anchor);
  URL.revokeObjectURL(url);
}


