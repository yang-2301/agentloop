<script setup lang="ts">
import { onMounted, ref } from 'vue'
import {
  fetchHealth, fetchRuns, fetchTools, runTask,
  type RunResult, type RunSummary, type ToolInfo,
} from './api'

const task = ref('')
const running = ref(false)
const error = ref('')
const result = ref<RunResult | null>(null)
const tools = ref<ToolInfo[]>([])
const runs = ref<RunSummary[]>([])
const llmMode = ref('')
const maxSteps = ref(8)
const expandedTool = ref('')
const copied = ref(false)

const examples = [
  'Calculate 15% of 2400',
  'Search our docs: how much does the team plan cost per year for 10 users?',
  'Search docs for the refund policy, then summarize it',
  'Search docs for the refund policy, and demo error recovery',
]

async function run() {
  if (!task.value.trim() || running.value) return
  running.value = true
  error.value = ''
  result.value = null
  copied.value = false
  try {
    result.value = await runTask(task.value.trim(), maxSteps.value)
    await refreshRuns()
  } catch (e: any) {
    error.value = e.message ?? String(e)
  } finally {
    running.value = false
  }
}

async function refreshRuns() {
  try {
    runs.value = await fetchRuns()
  } catch {
    /* backend not up */
  }
}

function toggleSchema(name: string) {
  expandedTool.value = expandedTool.value === name ? '' : name
}

function schemaJson(t: ToolInfo): string {
  return JSON.stringify(t.parameters, null, 2)
}

async function copyAnswer() {
  if (!result.value?.answer) return
  try {
    await navigator.clipboard.writeText(result.value.answer)
    copied.value = true
    setTimeout(() => (copied.value = false), 1500)
  } catch { /* clipboard unavailable */ }
}

function prettyArgs(raw: string): string {
  try {
    return JSON.stringify(JSON.parse(raw))
  } catch {
    return raw
  }
}

function timeAgo(ts: number): string {
  const s = Math.max(0, Math.floor(Date.now() / 1000 - ts))
  if (s < 60) return `${s}s ago`
  if (s < 3600) return `${Math.floor(s / 60)}m ago`
  return `${Math.floor(s / 3600)}h ago`
}

onMounted(async () => {
  // Demo hooks: /?task=<text>&max_steps=<n>&tool=<name> auto-runs the agent
  // and pre-expands a tool schema (used for screenshots & docs —
  // e.g. max_steps=1 shows the guard).
  const params = new URLSearchParams(window.location.search)
  const preset = params.get('task')
  const steps = parseInt(params.get('max_steps') ?? '', 10)
  if (Number.isFinite(steps) && steps >= 1 && steps <= 20) maxSteps.value = steps
  const presetTool = params.get('tool')
  if (presetTool) expandedTool.value = presetTool
  await Promise.all([refreshRuns(), loadMeta()])
  if (preset) {
    task.value = preset
    run()
  }
})

async function loadMeta() {
  try {
    const [t, h] = await Promise.all([fetchTools(), fetchHealth()])
    tools.value = t
    llmMode.value = h.llm_mode
  } catch {
    /* backend not up yet */
  }
}
</script>

<template>
  <div class="shell">
    <header>
      <div>
        <h1>agentloop</h1>
        <p class="sub">A minimal ReAct agent runtime — watch it reason, call tools and observe.</p>
      </div>
      <span class="badge" :class="llmMode">{{ llmMode === 'openai' ? 'live LLM' : 'mock LLM' }}</span>
    </header>

    <section class="panel">
      <textarea
        v-model="task"
        rows="2"
        placeholder="Give the agent a task, e.g. 'Search our docs for the refund policy and summarize it'"
        @keydown.enter.exact.prevent="run"
      ></textarea>
      <div class="row">
        <button class="btn primary" :disabled="running || !task.trim()" @click="run">
          {{ running ? 'Agent is working…' : 'Run agent' }}
        </button>
        <label class="steps-label">max steps
          <select v-model.number="maxSteps">
            <option v-for="n in [1, 2, 4, 8, 12]" :key="n" :value="n">{{ n }}</option>
          </select>
        </label>
        <button v-for="ex in examples" :key="ex" class="btn ghost" :disabled="running" @click="task = ex">
          {{ ex.length > 38 ? ex.slice(0, 38) + '…' : ex }}
        </button>
      </div>
    </section>

    <p v-if="error" class="error">{{ error }}</p>

    <section v-if="result" class="panel result">
      <div class="meta">
        <span class="pill" :class="result.status">{{ result.status }}</span>
        <span>{{ result.steps.length }} steps</span>
        <span>{{ result.tool_calls_total }} tool calls</span>
        <span>{{ result.duration_ms }} ms</span>
        <span class="runid">run {{ result.run_id }}</span>
      </div>

      <ol class="timeline">
        <li v-for="step in result.steps" :key="step.index" class="step">
          <div class="step-head">
            <span class="step-n">{{ step.index + 1 }}</span>
            <span class="step-ms">{{ step.duration_ms }} ms</span>
          </div>
          <p v-if="step.thought" class="thought">
            <strong>Thought</strong>{{ step.thought }}
          </p>
          <div v-for="(call, i) in step.tool_calls" :key="i" class="action">
            <strong>Action</strong>
            <code class="tool-name">{{ call.function.name }}</code>
            <code class="args">{{ prettyArgs(call.function.arguments) }}</code>
          </div>
          <div v-for="(obs, i) in step.observations" :key="'o' + i" class="observation" :class="{ errored: obs.output.startsWith('ERROR') }">
            <strong>Observation</strong>
            <pre>{{ obs.output }}</pre>
          </div>
        </li>
      </ol>

      <div v-if="result.answer" class="answer">
        <div class="answer-head">
          <strong>Final answer</strong>
          <button class="btn tiny" @click="copyAnswer">{{ copied ? '✓ copied' : 'copy' }}</button>
        </div>
        <p>{{ result.answer }}</p>
      </div>
      <div v-else class="answer warn">
        No final answer — the run ended with status <code>{{ result.status }}</code>.
      </div>
    </section>

    <section v-if="runs.length" class="panel history">
      <h2>Recent runs</h2>
      <table>
        <thead>
          <tr><th>task</th><th>status</th><th>calls</th><th>duration</th><th>when</th></tr>
        </thead>
        <tbody>
          <tr v-for="r in runs.slice(0, 8)" :key="r.run_id">
            <td class="task-cell" :title="r.task">{{ r.task.length > 46 ? r.task.slice(0, 46) + '…' : r.task }}</td>
            <td><span class="pill" :class="r.status">{{ r.status }}</span></td>
            <td>{{ r.tool_calls_total }}</td>
            <td>{{ r.duration_ms }} ms</td>
            <td class="muted">{{ timeAgo(r.created_at) }}</td>
          </tr>
        </tbody>
      </table>
    </section>

    <section class="panel tools">
      <h2>Available tools</h2>
      <ul>
        <li v-for="t in tools" :key="t.name" :class="{ open: expandedTool === t.name }">
          <div class="tool-row" @click="toggleSchema(t.name)">
            <code>{{ t.name }}</code>
            <span>{{ t.description }}</span>
            <span class="chevron">{{ expandedTool === t.name ? '▾' : '▸' }}</span>
          </div>
          <pre v-if="expandedTool === t.name" class="schema">{{ schemaJson(t) }}</pre>
        </li>
      </ul>
      <p class="hint">click a tool to inspect the JSON Schema the model sees</p>
    </section>

    <footer>agentloop · ReAct loop, tool registry, bounded memory, full trace — no framework, no magic.</footer>
  </div>
</template>
