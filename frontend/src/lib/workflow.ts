import type { Edge, Node } from "reactflow";
import type { FlowNodeData } from "@/types/flow";
import { API_BASE } from "@/lib/api";

export type SubmitWorkflowRequest = {
  sessionId: string;
  nodes: Node<FlowNodeData>[];
  edges: Edge[];
};

export type SubmitWorkflowResult = {
  workflowName: string;
  namespace: string;
  bucket: string;
  key: string;
  manifestFilename: string;
  cliOutput?: string | null;
};

export type OutputArtifactInfo = {
  inputName: string;
  sourceNodeId?: string | null;
  sourceArtifactName: string;
  bucket: string;
  key: string;
  workflowInputName?: string | null;
  size?: number | null;
  contentType?: string | null;
  exists: boolean;
};

export type ArtifactPreviewResult = {
  content: string;
  truncated: boolean;
  contentType?: string | null;
  encoding: string;
  size?: number | null;
};

function serializeNodes(nodes: Node<FlowNodeData>[]) {
  return nodes.map(({ id, type, data }) => ({ id, type, data }));
}

function serializeEdges(edges: Edge[]) {
  return edges.map(({ id, source, target }) => ({ id, source, target }));
}

export async function submitWorkflow(
  request: SubmitWorkflowRequest
): Promise<SubmitWorkflowResult> {
  const response = await fetch(`${API_BASE}/workflow/submit`, {
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

  const payload = (await response.json()) as SubmitWorkflowResult;
  return payload;
}


export async function getOutputArtifacts(
  request: SubmitWorkflowRequest,
  nodeId: string
): Promise<OutputArtifactInfo[]> {
  const url = new URL(`${API_BASE}/workflow/output-artifacts`);
  url.searchParams.set("nodeId", nodeId);

  const response = await fetch(url.toString(), {
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

  const payload = (await response.json()) as OutputArtifactInfo[];
  return payload;
}


export async function previewArtifact(
  bucket: string,
  key: string,
  maxBytes = 65536
): Promise<ArtifactPreviewResult> {
  const url = new URL(`${API_BASE}/artifacts/preview`);
  url.searchParams.set("bucket", bucket);
  url.searchParams.set("key", key);
  url.searchParams.set("maxBytes", String(maxBytes));

  const response = await fetch(url.toString());
  if (!response.ok) {
    const message = await response.text();
    throw new Error(message || `HTTP ${response.status}`);
  }

  const payload = (await response.json()) as ArtifactPreviewResult;
  return payload;
}


