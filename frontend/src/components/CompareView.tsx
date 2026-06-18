import { Button } from "@/components/ui/button"
import { useState, useEffect, useCallback } from 'react'
import { HugeiconsIcon } from "@hugeicons/react"
import { Loading01Icon, Cancel01Icon, CheckmarkCircleIcon, AlertCircleIcon, BarChartIcon } from "@hugeicons/core-free-icons"
import type { PipelineResult } from '../types'

interface CompareSession {
  mode: string
  session_id: string
  status: 'pending' | 'running' | 'completed' | 'failed'
  result?: PipelineResult
  error?: string
}

const MODE_LABELS: Record<string, string> = {
  hybrid: 'Hybrid (OCR + VLM)',
  graph: 'Graph-Based',
  end_to_end: 'End-to-End VLM',
}

const MODE_COLORS: Record<string, string> = {
  hybrid: 'text-emerald-400 border-emerald-500/30 bg-emerald-500/8',
  graph: 'text-blue-400 border-blue-500/30 bg-blue-500/8',
  end_to_end: 'text-purple-400 border-purple-500/30 bg-purple-500/8',
}

export function CompareView({ sessionId, onClose }: { sessionId: string; onClose: () => void }) {
  const [sessions, setSessions] = useState<CompareSession[]>([])
  const [launching, setLaunching] = useState(false)
  const [error, setError] = useState('')

  const launchCompare = useCallback(async () => {
    setLaunching(true)
    setError('')
    try {
      const res = await fetch(`/api/compare/${sessionId}`, { method: 'POST' })
      if (!res.ok) {
        const errData = await res.json().catch(() => ({}))
        throw new Error(errData.detail || `HTTP ${res.status}`)
      }
      const data = await res.json()
      const initial: CompareSession[] = data.sessions.map((s: { mode: string; session_id: string }) => ({
        mode: s.mode,
        session_id: s.session_id,
        status: 'pending' as const,
      }))
      setSessions(initial)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to launch comparison')
    }
    setLaunching(false)
  }, [sessionId])

  // Poll each session's status and fetch result when complete
  useEffect(() => {
    if (sessions.length === 0) return
    if (sessions.every(s => s.status === 'completed' || s.status === 'failed')) return

    const interval = setInterval(async () => {
      const updated = await Promise.all(
        sessions.map(async (s) => {
          if (s.status === 'completed' || s.status === 'failed') return s
          try {
            const statusRes = await fetch(`/api/status/${s.session_id}`)
            const status = await statusRes.json()
            if (status.status === 'completed') {
              const resultRes = await fetch(`/api/result/${s.session_id}`)
              const result = await resultRes.json() as PipelineResult
              return { ...s, status: 'completed' as const, result }
            } else if (status.status === 'failed') {
              return { ...s, status: 'failed' as const, error: status.error || 'Unknown error' }
            }
            return { ...s, status: 'running' as const }
          } catch {
            return s
          }
        })
      )
      setSessions(updated)
    }, 2000)

    return () => clearInterval(interval)
  }, [sessions])

  // Collect all field names across all completed sessions
  const allFields = new Set<string>()
  sessions.filter(s => s.result).forEach(s => {
    s.result!.pages.forEach(p => {
      Object.keys(p.extracted_fields || {}).forEach(f => allFields.add(f))
    })
  })
  const fieldList = Array.from(allFields).sort()

  const allDone = sessions.length > 0 && sessions.every(s => s.status === 'completed' || s.status === 'failed')
  const anyRunning = sessions.some(s => s.status === 'running' || s.status === 'pending')

  return (
    <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-50 p-4" onClick={onClose}>
      <div className="bg-bg-surface border border-border rounded-2xl shadow-2xl w-full max-w-5xl max-h-[90vh] flex flex-col"
        onClick={e => e.stopPropagation()}>
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-3 border-b border-border shrink-0">
          <h3 className="text-sm font-semibold text-text-primary">Mode Comparison</h3>
          <Button variant="ghost" size="icon-sm" onClick={onClose}
            className="text-text-muted hover:text-text-primary">
            <HugeiconsIcon icon={Cancel01Icon} className="size-4" />
          </Button>
        </div>

        {/* Body */}
        <div className="flex-1 overflow-y-auto p-5">
          {error && (
            <div className="mb-4 px-4 py-2.5 bg-accent-coral/20 border border-accent-coral/30 rounded-xl text-sm text-accent-coral">
              {error}
            </div>
          )}

          {sessions.length === 0 && !launching && (
            <div className="text-center py-12">
              <HugeiconsIcon icon={BarChartIcon} className="size-12 text-border mx-auto mb-4" />
              <p className="text-sm text-text-muted mb-2">Run all 3 pipeline modes on this document</p>
              <p className="text-xs text-text-muted mb-6">Results will be compared side-by-side</p>
              <Button variant="default" onClick={launchCompare}
                className="text-sm">
                Start Comparison
              </Button>
            </div>
          )}

          {launching && (
            <div className="flex items-center justify-center gap-3 py-12">
              <HugeiconsIcon icon={Loading01Icon} className="size-5 text-accent-violet animate-spin" />
              <span className="text-sm text-text-muted">Launching pipeline sessions...</span>
            </div>
          )}

          {sessions.length > 0 && (
            <>
              {/* Session status cards */}
              <div className="grid grid-cols-3 gap-3 mb-6">
                {sessions.map(s => (
                  <div key={s.mode}
                    className={`rounded-xl border p-4 ${
                      s.status === 'completed' ? 'border-accent-green/30 bg-accent-green/5'
                        : s.status === 'failed' ? 'border-accent-coral/30 bg-accent-coral/5'
                        : 'border-border bg-bg-elevated/30'
                    }`}>
                    <div className="flex items-center justify-between mb-2">
                      <span className={`text-xs font-semibold ${MODE_COLORS[s.mode]?.split(' ')[0] || 'text-text-primary'}`}>
                        {MODE_LABELS[s.mode] || s.mode}
                      </span>
                      {s.status === 'completed' && <HugeiconsIcon icon={CheckmarkCircleIcon} className="size-4 text-accent-green" />}
                      {s.status === 'failed' && <HugeiconsIcon icon={AlertCircleIcon} className="size-4 text-accent-coral" />}
                      {(s.status === 'running' || s.status === 'pending') && (
                        <HugeiconsIcon icon={Loading01Icon} className="size-4 text-accent-violet animate-spin" />
                      )}
                    </div>
                    <div className="text-[13px] text-text-muted">
                      {s.status === 'completed' && `Session: ${s.session_id}`}
                      {s.status === 'running' && 'Running...'}
                      {s.status === 'pending' && 'Waiting...'}
                      {s.status === 'failed' && (s.error || 'Failed')}
                    </div>
                  </div>
                ))}
              </div>

              {/* Comparison table */}
              {allDone && fieldList.length > 0 && (
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b border-border">
                        <th className="text-left py-2 pr-4 text-xs font-semibold text-text-muted uppercase tracking-wider">Field</th>
                        {sessions.filter(s => s.status === 'completed').map(s => (
                          <th key={s.mode} className={`text-left py-2 px-3 text-xs font-semibold uppercase tracking-wider ${MODE_COLORS[s.mode]?.split(' ')[0] || 'text-text-muted'}`}>
                            {MODE_LABELS[s.mode] || s.mode}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {fieldList.map(field => {
                        const values = sessions
                          .filter(s => s.status === 'completed')
                          .map(s => {
                            for (const p of s.result?.pages || []) {
                              if (field in (p.extracted_fields || {})) return typeof p.extracted_fields[field] === 'object' && p.extracted_fields[field] !== null ? JSON.stringify(p.extracted_fields[field]) : String(p.extracted_fields[field])
                            }
                            return null
                          })
                        const uniqueValues = new Set(values.filter(Boolean))
                        const hasDiff = uniqueValues.size > 1

                        return (
                          <tr key={field} className={`border-b border-border/50 hover:bg-bg-elevated/20 ${hasDiff ? 'bg-amber-500/5' : ''}`}>
                            <td className="py-2.5 pr-4 text-xs font-medium text-text-primary whitespace-nowrap">
                              {field.replace(/_/g, ' ')}
                              {hasDiff && <span className="ml-2 text-[13px] text-amber-500 font-semibold">DIFF</span>}
                            </td>
                            {values.map((v, i) => {
                              const firstVal = values.find(Boolean)
                              const isDiff = v !== firstVal && v !== null
                              return (
                                <td key={i} className={`py-2.5 px-3 text-xs font-mono max-w-[200px] truncate ${
                                  v ? (isDiff ? 'text-amber-300' : 'text-text-primary') : 'text-text-muted italic'
                                }`}>
                                  {v || '—'}
                                </td>
                              )
                            })}
                          </tr>
                        )
                      })}
                    </tbody>
                  </table>
                </div>
              )}

              {allDone && fieldList.length === 0 && (
                <div className="text-center py-8 text-sm text-text-muted">
                  No extracted fields found across modes
                </div>
              )}

              {anyRunning && (
                <div className="flex items-center justify-center gap-2 py-4 text-xs text-text-muted">
                  <HugeiconsIcon icon={Loading01Icon} className="size-3 animate-spin" />
                  Waiting for all modes to complete...
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  )
}
