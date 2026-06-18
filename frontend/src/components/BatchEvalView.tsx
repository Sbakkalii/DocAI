import { Button } from "@/components/ui/button"
import { useState, useEffect } from 'react'
import { HugeiconsIcon } from "@hugeicons/react"
import { Loading01Icon } from "@hugeicons/core-free-icons"

export function BatchEvalView() {
  const [mode, setMode] = useState('hybrid')
  const [model, setModel] = useState('qwen2.5:7b-instruct-q4_K_M')
  const [numDocs, setNumDocs] = useState(10)
  const [running, setRunning] = useState(false)
  const [result, setResult] = useState<Record<string, unknown> | null>(null)
  const [error, setError] = useState('')
  const [ollamaModels, setOllamaModels] = useState<string[]>([])

  useEffect(() => {
    fetch('/api/ollama/models').then(r => r.json()).then(d => { if (d.models?.length) setOllamaModels(d.models) }).catch(() => {})
  }, [])

  async function runBatch() {
    setRunning(true); setResult(null); setError('')
    try {
      const res = await fetch('/api/eval/batch', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mode, model, num_docs: numDocs }),
      })
      if (!res.ok) throw new Error((await res.json()).detail || 'Batch eval failed')
      const data = await res.json()
      setResult(data)
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Batch eval failed')
    }
    setRunning(false)
  }

  const agg = result?.aggregate as Record<string, unknown> | undefined
  const perDoc = result?.per_doc as Array<Record<string, unknown>> | undefined
  const lat = agg?.latency_seconds as Record<string, unknown> | undefined

  return (
    <div className="h-full overflow-y-auto p-6">
      <div className="max-w-4xl mx-auto space-y-6">
        <h2 className="text-base font-semibold text-text-primary">Batch Evaluation</h2>

        <div className="bg-bg-surface border border-border rounded-xl p-5 space-y-4">
          <div className="grid grid-cols-4 gap-4">
            <div>
              <label className="text-xs font-medium text-text-muted block mb-1">Mode</label>
              <div className="flex gap-1.5">
                {['hybrid', 'graph', 'end_to_end'].map(m => (
                  <Button key={m} variant={mode === m ? 'outline' : 'ghost'} size="sm" onClick={() => setMode(m)}
                    className={`text-xs ${mode === m ? 'bg-accent-violet/20 text-accent-violet ring-1 ring-accent-violet/40' : 'text-text-muted hover:text-text-primary'}`}>
                    {m.replace('_', ' ')}
                  </Button>
                ))}
              </div>
            </div>
            <div>
              <label className="text-xs font-medium text-text-muted block mb-1">Model</label>
              <select value={model} onChange={e => setModel(e.target.value)}
                className="w-full bg-bg-elevated text-xs text-text-primary px-2 py-1.5 rounded-lg border border-border">
                {(() => {
                  const batchModels = ollamaModels.length > 0 ? ollamaModels : ['qwen2.5:7b-instruct-q4_K_M', 'llama3.2:3b-instruct-q4_K_M']
                  const batchInstruct = batchModels.filter(m => m.includes('instruct'))
                  const batchBase = batchModels.filter(m => !m.includes('instruct'))
                  return (
                    <>
                      {batchInstruct.length > 0 && <optgroup label="Instruct (tuned)">
                        {batchInstruct.map(m => <option key={m} value={m}>{m.replace(':latest', '')}</option>)}
                      </optgroup>}
                      {batchBase.length > 0 && <optgroup label="Base (untuned)">
                        {batchBase.map(m => <option key={m} value={m}>{m.replace(':latest', '')}</option>)}
                      </optgroup>}
                    </>
                  )
                })()}
              </select>
            </div>
            <div>
              <label className="text-xs font-medium text-text-muted block mb-1">Documents</label>
              <input type="number" value={numDocs} onChange={e => setNumDocs(Math.max(1, Math.min(200, parseInt(e.target.value) || 1)))}
                className="w-full bg-bg-elevated text-xs text-text-primary px-2 py-1.5 rounded-lg border border-border" min={1} max={200} />
            </div>
            <div className="flex items-end">
              <Button variant="default" onClick={runBatch} disabled={running}
                className="w-full">
                {running ? 'Running...' : 'Run Batch'}
              </Button>
            </div>
          </div>
          {error && <div className="text-xs text-accent-coral bg-accent-coral/20 px-3 py-2 rounded-lg">{error}</div>}
          {running && (
            <div className="flex items-center gap-2 text-xs text-text-muted">
              <HugeiconsIcon icon={Loading01Icon} className="size-3 animate-spin" /> Processing {numDocs} documents...
            </div>
          )}
        </div>

        {agg && (
          <>
            <div className="grid grid-cols-5 gap-3">
              <div className="bg-bg-surface border border-border rounded-xl p-3">
                <div className="text-xs font-semibold text-text-muted uppercase">Documents</div>
                <div className="text-lg font-bold text-text-primary tabular-nums">{agg.total_docs as number}</div>
                {(agg.failed as number) > 0 && <div className="text-xs text-accent-coral">{agg.failed as number} failed</div>}
              </div>
              <div className="bg-bg-surface border border-border rounded-xl p-3">
                <div className="text-xs font-semibold text-text-muted uppercase">Throughput</div>
                <div className="text-lg font-bold text-accent-green tabular-nums">{(agg.throughput_docs_per_sec as number)?.toFixed(2)}</div>
                <div className="text-xs text-text-muted">docs/sec</div>
              </div>
              <div className="bg-bg-surface border border-border rounded-xl p-3">
                <div className="text-xs font-semibold text-text-muted uppercase">P50</div>
                <div className="text-lg font-bold text-accent-violet tabular-nums">{(lat?.p50 as number)?.toFixed(2)}s</div>
              </div>
              <div className="bg-bg-surface border border-border rounded-xl p-3">
                <div className="text-xs font-semibold text-text-muted uppercase">P95</div>
                <div className="text-lg font-bold text-accent-yellow tabular-nums">{(lat?.p95 as number)?.toFixed(2)}s</div>
              </div>
              <div className="bg-bg-surface border border-border rounded-xl p-3">
                <div className="text-xs font-semibold text-text-muted uppercase">P99</div>
                <div className="text-lg font-bold text-accent-coral tabular-nums">{(lat?.p99 as number)?.toFixed(2)}s</div>
              </div>
            </div>

            {agg.accuracy && (
              <div className="bg-bg-surface border border-border rounded-xl p-4">
                <h3 className="text-sm font-semibold text-text-primary mb-3">Accuracy</h3>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <div className="text-xs text-text-muted">Mean Exact Match</div>
                    <div className="text-lg font-bold text-accent-green tabular-nums">{((agg.accuracy as Record<string, unknown>).mean_exact_match as number * 100).toFixed(1)}%</div>
                  </div>
                  <div>
                    <div className="text-xs text-text-muted">Mean Token F1</div>
                    <div className="text-lg font-bold text-accent-green tabular-nums">{((agg.accuracy as Record<string, unknown>).mean_token_f1 as number * 100).toFixed(1)}%</div>
                  </div>
                </div>
              </div>
            )}

            {agg.step_timings && Object.keys(agg.step_timings as Record<string, unknown>).length > 0 && (
              <div className="bg-bg-surface border border-border rounded-xl p-4">
                <h3 className="text-sm font-semibold text-text-primary mb-3">Step Timing Breakdown</h3>
                <div className="space-y-2">
                  {Object.entries(agg.step_timings as Record<string, Record<string, unknown>>).map(([step, times]) => {
                    const p50 = times.p50 as number
                    const p95 = times.p95 as number
                    const maxLat = Math.max(...Object.values(agg.step_timings as Record<string, Record<string, unknown>>).map(t => t.p95 as number), 1)
                    return (
                      <div key={step} className="flex items-center gap-3">
                        <span className="text-xs text-text-muted min-w-[7rem] truncate">{step}</span>
                        <div className="flex-1 h-3 bg-bg-elevated rounded-full overflow-hidden">
                          <div className="h-full rounded-full bg-accent-violet/60" style={{ width: `${(p95 / maxLat) * 100}%` }} />
                        </div>
                        <span className="text-xs text-text-muted font-mono tabular-nums min-w-[8rem] text-right">
                          P50: {p50.toFixed(2)}s · P95: {p95.toFixed(2)}s
                        </span>
                      </div>
                    )
                  })}
                </div>
              </div>
            )}

            {perDoc && perDoc.length > 0 && (
              <div className="bg-bg-surface border border-border rounded-xl p-4">
                <h3 className="text-sm font-semibold text-text-primary mb-3">Per-Document Results</h3>
                <div className="overflow-x-auto">
                  <table className="w-full text-xs">
                    <thead>
                      <tr className="text-text-muted border-b border-border">
                        <th className="text-left py-2 pr-4">Document</th>
                        <th className="text-right py-2 px-2">Time</th>
                        <th className="text-right py-2 px-2">Exact Match</th>
                        <th className="text-right py-2 px-2">Token F1</th>
                        <th className="text-right py-2 pl-2">Status</th>
                      </tr>
                    </thead>
                    <tbody>
                      {perDoc.map((d, i) => {
                        const acc = d.accuracy as Record<string, unknown> | null
                        return (
                          <tr key={i} className="border-b border-border/50 hover:bg-bg-elevated/30">
                            <td className="py-2 pr-4 text-text-primary truncate max-w-[200px]">{d.doc as string}</td>
                            <td className="py-2 px-2 text-right text-text-muted font-mono">{(d.total_time as number).toFixed(2)}s</td>
                            <td className="py-2 px-2 text-right font-mono">
                              {acc?.exact_match != null ? <span className={acc.exact_match as number > 0.5 ? 'text-accent-violet' : 'text-accent-coral'}>{(acc.exact_match as number * 100).toFixed(0)}%</span> : <span className="text-text-muted">-</span>}
                            </td>
                            <td className="py-2 px-2 text-right font-mono">
                              {acc?.token_f1 != null ? <span className={acc.token_f1 as number > 0.5 ? 'text-accent-violet' : 'text-accent-coral'}>{(acc.token_f1 as number * 100).toFixed(0)}%</span> : <span className="text-text-muted">-</span>}
                            </td>
                            <td className="py-2 pl-2 text-right">{d.error ? <span className="text-accent-coral">Failed</span> : <span className="text-accent-violet">OK</span>}</td>
                          </tr>
                        )
                      })}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  )
}
