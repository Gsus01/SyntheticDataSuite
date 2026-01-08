import type { Edge, Node } from "reactflow";
import type { FlowNodeData, WorkflowNodeRuntimeStatus } from "@/types/flow";
import { API_BASE, buildApiUrl } from "@/lib/api";

export type SubmitWorkflowRequest = {
  sessionId: string;
  nodes: Node<FlowNodeData>[];
  edges: Edge[];
  workflowId?: string;
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

export type CompileWorkflowResult = {
  manifest: string;
  manifestFilename: string;
  bucket: string;
  nodeSlugMap: Record<string, string>;
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

export type WorkflowSummary = {
  workflowId: string;
  name: string;
  description?: string | null;
  createdAt: string;
  updatedAt: string;
  lastSubmittedAt?: string | null;
  lastWorkflowName?: string | null;
  lastNamespace?: string | null;
};

export type WorkflowRecord = WorkflowSummary & {
  sessionId: string;
  nodes: Node<FlowNodeData>[];
  edges: Edge[];
  compiledManifest?: string | null;
  compiledAt?: string | null;
  manifestFilename?: string | null;
  nodeSlugMap?: Record<string, string>;
  lastBucket?: string | null;
  lastKey?: string | null;
  lastManifestFilename?: string | null;
  lastCliOutput?: string | null;
};

export type WorkflowSaveRequest = {
  workflowId?: string;
  name: string;
  description?: string | null;
  sessionId: string;
  nodes: unknown[];
  edges: unknown[];
  compiledManifest?: string | null;
  manifestFilename?: string | null;
  nodeSlugMap?: Record<string, string>;
};

function serializeNodes(nodes: Node<FlowNodeData>[]) {
  return nodes.map(({ id, type, data }) => {
    const { runtimeStatus, artifactPorts, ...rest } = data;
    void runtimeStatus;
    void artifactPorts;
    return { id, type, data: rest };
  });
}

function serializeEdges(edges: Edge[]) {
  return edges.map(({ id, source, target, sourceHandle, targetHandle }) => ({
    id,
    source,
    target,
    ...(sourceHandle ? { sourceHandle } : {}),
    ...(targetHandle ? { targetHandle } : {}),
  }));
}

export async function compileWorkflow(
  request: SubmitWorkflowRequest
): Promise<CompileWorkflowResult> {
  const response = await fetch(`${API_BASE}/workflow/compile`, {
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

  const payload = (await response.json()) as CompileWorkflowResult;
  return payload;
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
      workflowId: request.workflowId,
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
  const url = buildApiUrl("/workflow/output-artifacts");
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
  const url = buildApiUrl("/artifacts/preview");
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

  const url = buildApiUrl("/workflow/logs/stream");
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
  const url = buildApiUrl("/workflow/status");
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

export async function saveWorkflow(request: WorkflowSaveRequest): Promise<WorkflowRecord> {
  const response = await fetch(`${API_BASE}/workflows`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      workflowId: request.workflowId,
      name: request.name,
      description: request.description,
      sessionId: request.sessionId,
      nodes: request.nodes,
      edges: request.edges,
      compiledManifest: request.compiledManifest,
      manifestFilename: request.manifestFilename,
      nodeSlugMap: request.nodeSlugMap,
    }),
  });

  if (!response.ok) {
    const message = await response.text();
    throw new Error(message || `HTTP ${response.status}`);
  }

  const payload = (await response.json()) as WorkflowRecord;
  return payload;
}

export async function listWorkflows(): Promise<WorkflowSummary[]> {
  const response = await fetch(`${API_BASE}/workflows`);
  if (!response.ok) {
    const message = await response.text();
    throw new Error(message || `HTTP ${response.status}`);
  }
  const payload = (await response.json()) as WorkflowSummary[];
  return payload;
}

export async function getWorkflow(workflowId: string): Promise<WorkflowRecord> {
  const response = await fetch(`${API_BASE}/workflows/${workflowId}`);
  if (!response.ok) {
    const message = await response.text();
    throw new Error(message || `HTTP ${response.status}`);
  }
  const payload = (await response.json()) as WorkflowRecord;
  return payload;
}
