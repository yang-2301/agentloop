export interface ToolCall {
  id: string
  type: string
  function: { name: string; arguments: string }
}

export interface Observation {
  call_id: string
  tool: string
  output: string
}

export interface Step {
  index: number
  thought: string | null
  tool_calls: ToolCall[]
  observations: Observation[]
  duration_ms: number
}

export interface RunResult {
  run_id: string
  task: string
  status: 'completed' | 'max_steps' | 'error'
  answer: string | null
  steps: Step[]
  tool_calls_total: number
  duration_ms: number
}

export interface ToolInfo {
  name: string
  description: string
}

export async function runTask(task: string, maxSteps = 8): Promise<RunResult> {
  const res = await fetch('/api/run', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ task, max_steps: maxSteps }),
  })
  if (!res.ok) throw new Error(`run failed: ${res.status}`)
  return res.json()
}

export async function fetchTools(): Promise<ToolInfo[]> {
  const res = await fetch('/api/tools')
  if (!res.ok) throw new Error(`tools failed: ${res.status}`)
  const body = await res.json()
  return body.tools
}

export async function fetchHealth(): Promise<{ status: string; llm_mode: string }> {
  const res = await fetch('/api/health')
  if (!res.ok) throw new Error(`health failed: ${res.status}`)
  return res.json()
}
