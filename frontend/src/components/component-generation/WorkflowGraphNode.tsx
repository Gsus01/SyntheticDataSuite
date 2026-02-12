import { Handle, Position, type NodeProps, type NodeTypes } from "@xyflow/react";

import { cn } from "./ui/primitives";
import { nodeTheme, stateLabel } from "./utils";
import type { WorkflowGraphNodeType } from "./types";

const HANDLE_CLASS =
  "!size-2 !rounded-full !border !border-current !bg-white dark:!bg-slate-100";

function WorkflowGraphNode({ data }: NodeProps<WorkflowGraphNodeType>) {
  const theme = nodeTheme(data.state);
  return (
    <div
      className={cn(
        "min-w-44 rounded-lg border-2 p-2.5 text-xs shadow-sm backdrop-blur",
        theme.toneClass,
        theme.pulseClass
      )}
    >
      <Handle
        type="target"
        position={Position.Left}
        isConnectable={false}
        className={HANDLE_CLASS}
      />
      <Handle
        type="source"
        position={Position.Right}
        isConnectable={false}
        className={HANDLE_CLASS}
      />
      <div className="flex items-center justify-between gap-2">
        <span className="text-[10px] font-semibold uppercase tracking-wide">{data.label}</span>
        <span className="rounded-full border border-current px-1.5 py-0.5 text-[9px] uppercase">
          {stateLabel(data.state)}
        </span>
      </div>
      <div className="mt-1 truncate text-[10px] opacity-80">
        {data.message && data.message.trim() ? data.message : "LangGraph step"}
      </div>
    </div>
  );
}

export const graphNodeTypes: NodeTypes = {
  workflowNode: WorkflowGraphNode,
};
