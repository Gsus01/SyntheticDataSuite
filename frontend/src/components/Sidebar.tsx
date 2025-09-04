"use client";

import React from "react";
import NodeCard from "@/components/NodeCard";
import { DND_MIME, NODE_TYPES, type NodeTypeId } from "@/lib/flow-const";

type Item = {
  label: string;
  type: NodeTypeId;
};

const items: Item[] = [
  { label: "Input Node", type: NODE_TYPES.nodeInput },
  { label: "Default Node", type: NODE_TYPES.nodeDefault },
  { label: "Output Node", type: NODE_TYPES.nodeOutput },
];

function onDragStart(event: React.DragEvent<HTMLDivElement>, nodeType: Item["type"]) {
  event.dataTransfer.setData(DND_MIME, nodeType);
  event.dataTransfer.effectAllowed = "move";
}

export default function Sidebar() {
  return (
    <aside className="w-64 shrink-0 border-r border-gray-200 bg-gray-50 p-3 text-sm text-gray-800">
      <div className="mb-3 font-medium text-gray-600 uppercase tracking-wide">Nodos</div>
      <div className="space-y-2">
        {items.map((item) => (
          <div
            key={item.type}
            role="button"
            tabIndex={0}
            draggable
            onDragStart={(e) => onDragStart(e, item.type)}
            className="cursor-grab active:cursor-grabbing"
          >
            <NodeCard
              label={item.label}
              variant={
                item.type === "nodeInput"
                  ? "input"
                  : item.type === "nodeDefault"
                  ? "default"
                  : "output"
              }
              compact
            />
          </div>
        ))}
      </div>
    </aside>
  );
}
