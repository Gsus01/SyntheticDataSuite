import { create } from 'zustand'

interface WorkflowState {
  nodes: any[]
  edges: any[]
}

const useWorkflowStore = create<WorkflowState>(() => ({
  nodes: [],
  edges: [],
}))

export default useWorkflowStore
