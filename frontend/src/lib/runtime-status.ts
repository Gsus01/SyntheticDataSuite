import type { WorkflowNodeRuntimeStatus } from "@/types/flow";

export const FINAL_PHASES = new Set([
  "succeeded",
  "failed",
  "error",
  "terminated",
  "skipped",
  "omitted",
  "cancelled",
]);

const RUNNING_HINT_PHASES = new Set(["pending", "queued", "waiting", "idle", "unknown"]);

export function isFinalPhase(phase?: string | null): boolean {
  if (!phase) {
    return false;
  }
  return FINAL_PHASES.has(phase.trim().toLowerCase());
}

export function derivePhase(status?: WorkflowNodeRuntimeStatus | null): string | null {
  if (!status) {
    return null;
  }
  const rawPhase = status.phase?.trim();
  const lowered = rawPhase?.toLowerCase();

  if (status.startedAt && !status.finishedAt) {
    if (!lowered || RUNNING_HINT_PHASES.has(lowered)) {
      return "Running";
    }
  }

  return rawPhase ?? null;
}

export function normalizeStatusForDisplay(
  status?: WorkflowNodeRuntimeStatus | null
): WorkflowNodeRuntimeStatus | undefined {
  if (!status) {
    return undefined;
  }
  const derived = derivePhase(status);
  if (!derived || derived === status.phase) {
    return status;
  }
  return {
    ...status,
    phase: derived,
  };
}

export function resolveStatusKey(status?: WorkflowNodeRuntimeStatus | null): string | null {
  const derived = derivePhase(status);
  return derived ? derived.toLowerCase() : null;
}
