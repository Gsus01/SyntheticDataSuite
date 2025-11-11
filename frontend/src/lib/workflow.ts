import type { Edge, Node } from "reactflow";
import type { FlowNodeData, WorkflowNodeRuntimeStatus } from "@/types/flow";
import { API_BASE } from "@/lib/api";

export type SubmitWorkflowRequest = {
  sessionId: string;
  nodes: Node<FlowNodeData>[];
  edges: Edge[];
};

export type SubmitWorkflowResult = {
  workflowName: string;
  namespace: string;
  nodeSlugMap: Record<string, string>;
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

export type WorkflowLogsResult = {
  workflowName: string;
  namespace: string;
  logs: string;
};

export type WorkflowStatusNodeMap = Record<string, WorkflowNodeRuntimeStatus>;

export type WorkflowStatusResponse = {
  workflowName: string;
  namespace: string;
  phase?: string | null;
  finished: boolean;
  updatedAt?: string | null;
  nodes: WorkflowStatusNodeMap;
};

export type WorkflowStatusRequest = {
  workflowName: string;
  namespace?: string;
};

export type WorkflowLogFetchRequest = {
  workflowName: string;
  namespace?: string;
  podName?: string;
  follow?: boolean;
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


export async function fetchWorkflowLogs(
  request: WorkflowLogFetchRequest
): Promise<WorkflowLogsResult> {
  console.debug("[logs] fetch start", {
    workflowName: request.workflowName,
    namespace: request.namespace,
    podName: request.podName,
    follow: request.follow,
  });

  const url = new URL(`${API_BASE}/workflow/logs/stream`);
  url.searchParams.set("workflowName", request.workflowName);
  if (request.namespace) {
    url.searchParams.set("namespace", request.namespace);
  }
  if (request.podName) {
    url.searchParams.set("podName", request.podName);
  }
  if (request.follow) {
    url.searchParams.set("follow", "true");
  }

  try {
    const response = await fetch(url.toString());
    if (!response.ok) {
      const message = await response.text();
      console.debug("[logs] fetch error", { status: response.status, message });
      throw new Error(message || `HTTP ${response.status}`);
    }
    const payload = (await response.json()) as WorkflowLogsResult;
    console.debug("[logs] fetch success", {
      size: payload.logs?.length ?? 0,
    });
    return payload;
  } catch (err) {
    console.debug("[logs] fetch exception", err);
    throw err;
  }
}

export async function fetchWorkflowStatus(
  request: WorkflowStatusRequest
): Promise<WorkflowStatusResponse | null> {
  const url = new URL(`${API_BASE}/workflow/status`);
  url.searchParams.set("workflowName", request.workflowName);
  if (request.namespace) {
    url.searchParams.set("namespace", request.namespace);
  }

  const response = await fetch(url.toString());
  if (response.status === 404) {
    return null;
  }

  if (!response.ok) {
    const message = await response.text();
    throw new Error(message || `HTTP ${response.status}`);
  }

  const raw = (await response.json()) as Partial<WorkflowStatusResponse>;
  return {
    workflowName: raw.workflowName ?? request.workflowName,
    namespace: raw.namespace ?? request.namespace ?? "",
    phase: raw.phase ?? null,
    finished: Boolean(raw.finished),
    updatedAt: raw.updatedAt ?? null,
    nodes: raw.nodes ?? {},
  };
}
