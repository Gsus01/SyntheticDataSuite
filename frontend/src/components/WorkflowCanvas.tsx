import ReactFlow, { Background, Controls, MiniMap } from 'reactflow'

export default function WorkflowCanvas() {
  return (
    <div className="flex-1 h-full">
      <ReactFlow nodes={[]} edges={[]}
        fitView
      >
        <Background />
        <Controls />
        <MiniMap />
      </ReactFlow>
    </div>
  )
}
