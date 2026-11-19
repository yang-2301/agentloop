<script setup lang="ts">
import { onMounted, ref } from 'vue'
import { fetchHealth, fetchTools, runTask, type RunResult, type ToolInfo } from './api'

const task = ref('')
const running = ref(false)
const error = ref('')
const result = ref<RunResult | null>(null)
const tools = ref<ToolInfo[]>([])
const llmMode = ref('')

const examples = [
  'Calculate 15% of 2400',
  'Search our docs: how much does the team plan cost per year for 10 users?',
  'Search docs for the refund policy, then summarize it',
]

async function run() {
  if (!task.value.trim() || running.value) return
  running.value = true
  error.value = ''
  result.value = null
  try {
    result.value = await runTask(task.value.trim())
  } catch (e: any) {
    error.value = e.message ?? String(e)
  } finally {
    running.value = false
  }
}

function prettyArgs(raw: string): string {
  try {
    return JSON.stringify(JSON.parse(raw))
  } catch {
    return raw
  }
}

onMounted(async () => {
  try {
    const [t, h] = await Promise.all([fetchTools(), fetchHealth()])
    tools.value = t
    llmMode.value = h.llm_mode
  } catch {
    /* backend not up yet */
  }
})
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
        <button v-for="ex in examples" :key="ex" class="btn ghost" :disabled="running" @click="task = ex">
          {{ ex.length > 42 ? ex.slice(0, 42) + '…' : ex }}
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
          <div v-for="(obs, i) in step.observations" :key="'o' + i" class="observation">
            <strong>Observation</strong>
            <pre>{{ obs.output }}</pre>
          </div>
        </li>
      </ol>

      <div v-if="result.answer" class="answer">
        <strong>Final answer</strong>
        <p>{{ result.answer }}</p>
      </div>
      <div v-else class="answer warn">
        No final answer — the run ended with status <code>{{ result.status }}</code>.
      </div>
    </section>

    <section class="panel tools">
      <h2>Available tools</h2>
      <ul>
        <li v-for="t in tools" :key="t.name">
          <code>{{ t.name }}</code>
          <span>{{ t.description }}</span>
        </li>
      </ul>
    </section>

    <footer>agentloop · ReAct loop, tool registry, bounded memory, full trace — no framework, no magic.</footer>
  </div>
</template>
