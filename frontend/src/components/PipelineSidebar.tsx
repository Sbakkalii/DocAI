import { Button } from "@/components/ui/button"
import { useState, useEffect } from 'react'
import { HugeiconsIcon } from "@hugeicons/react"
import { ChevronLeftIcon, CheckmarkCircleIcon, AlertCircleIcon, PlayIcon, InformationCircleIcon, Cancel01Icon } from "@hugeicons/core-free-icons"
import { MODE_OPTIONS, MODE_INFO, FIELD_CATEGORIES } from './constants'

export function PipelineSidebar({ collapsed, onToggle,
  currentMode, onModeChange: _onModeChange, onRunMode: _onRunMode,
  selectedFields, onFieldToggle, sendFields,
  embeddingModels, currentEmbeddingModel, setCurrentEmbeddingModel,
  currentVlmModel, setCurrentVlmModel, vlmModels,
  progressPct, completedCount, totalSteps,
}: {
  collapsed: boolean
  onToggle: () => void
  currentMode: string; onModeChange: (m: string) => void; onRunMode: (m: string) => void
  selectedFields: string[]; onFieldToggle: (f: string) => void; sendFields: (f: string[]) => void
  embeddingModels: Array<{id: string; name: string; provider: string}>
  currentEmbeddingModel: string; setCurrentEmbeddingModel: (m: string) => void
  currentVlmModel: string; setCurrentVlmModel: (m: string) => void
  vlmModels: string[]
  progressPct: number; completedCount: number; totalSteps: number
}) {
  const allFieldKeys = FIELD_CATEGORIES.flatMap(c => c.fields)
  const [reviewCount, setReviewCount] = useState(0)
  const [modeInfo, setModeInfo] = useState<string | null>(null)

  useEffect(() => {
    fetch('/api/review/queue').then(r => r.json()).then(d => setReviewCount(d.count || 0)).catch(() => {})
    const interval = setInterval(() => {
      fetch('/api/review/queue').then(r => r.json()).then(d => setReviewCount(d.count || 0)).catch(() => {})
    }, 30000)
    return () => clearInterval(interval)
  }, [])

  return (
    <div className={`h-screen max-h-screen bg-bg-surface border-r border-border flex flex-col overflow-y-auto transition-all duration-200 ${collapsed ? 'w-0 overflow-hidden' : 'w-80'}`}>
      {/* ── Header ── */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-border shrink-0">
        <div className="flex items-center gap-2">
          <div className="text-xs font-semibold text-text-muted uppercase tracking-wider">Pipeline</div>
          <span className="text-xs tabular-nums text-text-muted">{completedCount}/{totalSteps}</span>
        </div>
        <Button variant="ghost" size="icon-sm" onClick={onToggle}
          className="text-text-muted hover:text-text-primary">
          <HugeiconsIcon icon={ChevronLeftIcon} className="size-4" />
        </Button>
      </div>

      {/* ── Progress ── */}
      <div className="px-4 py-2 border-b border-border/50 shrink-0">
        <div className="h-1 bg-border rounded-full overflow-hidden">
          <div className="h-full rounded-full bg-accent-violet transition-all duration-500" style={{ width: `${progressPct}%` }} />
        </div>
      </div>

      {/* ── Review Queue banner ── */}
      {reviewCount > 0 && (
        <a href="/#/batch" className="block px-4 py-2 border-b border-border/50 shrink-0 bg-accent-coral/5 hover:bg-accent-coral/10 transition-colors">
          <div className="flex items-center gap-2 text-xs">
            <HugeiconsIcon icon={AlertCircleIcon} className="size-3.5 text-accent-coral" />
            <span className="text-text-primary font-medium">Review Queue</span>
            <span className="ml-auto bg-accent-coral/20 text-accent-coral px-1.5 py-0.5 rounded-full text-[11px] font-bold tabular-nums">{reviewCount}</span>
          </div>
        </a>
      )}

      {/* ── Extraction Mode ── */}
      <div className="px-4 py-3 border-b border-border/50 space-y-2 shrink-0 relative">
        <div className="text-xs font-semibold text-text-muted">Extraction Mode</div>
        {MODE_OPTIONS.map(m => (
          <div key={m.value} className="flex items-start gap-1">
            <button onClick={() => !m.disabled && _onModeChange(m.value)}
              className={`flex-1 flex flex-col items-start px-3 py-2 rounded-lg border text-xs transition-colors ${
                m.disabled
                  ? 'bg-bg-elevated/20 text-text-muted/40 border-border/20 cursor-default'
                  : m.value === currentMode
                    ? 'bg-accent-violet/20 text-accent-violet border-accent-violet/40 font-medium'
                    : 'bg-bg-elevated/40 text-text-muted border-border/40 hover:bg-accent-violet/10 hover:border-accent-violet/30'
              }`}>
              <span className="text-left">{m.label}</span>
              <span className="text-[11px] text-text-muted/70 text-left mt-0.5 font-normal">{m.desc}</span>
            </button>
            <Button variant="ghost" size="icon-sm" onClick={() => setModeInfo(modeInfo === m.value ? null : m.value)}
              title={`About ${m.label}`}
              className="shrink-0 mt-0.5 text-text-muted hover:text-accent-violet">
              <HugeiconsIcon icon={InformationCircleIcon} className="size-4" />
            </Button>
            {!m.disabled && (
              <Button variant="default" size="icon-sm" onClick={() => _onRunMode(m.value)}
                title={`Run ${m.label}`}
                className="shrink-0 mt-0.5">
                <HugeiconsIcon icon={PlayIcon} className="size-3.5" />
              </Button>
            )}
            {modeInfo === m.value && MODE_INFO[m.value] && (
              <div className="absolute left-2 right-2 top-full z-50 mt-1 p-3 bg-bg-elevated border border-accent-violet/30 rounded-xl shadow-xl animate-fade-in"
                onClick={e => e.stopPropagation()}>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-xs font-semibold text-accent-violet">{m.label}</span>
                  <Button variant="ghost" size="icon-sm" onClick={() => setModeInfo(null)} className="text-text-muted hover:text-text-primary">
                    <HugeiconsIcon icon={Cancel01Icon} className="size-3" />
                  </Button>
                </div>
                <div className="space-y-1.5 text-[13px]">
                  <div><span className="text-text-muted font-medium">What it does:</span> <span className="text-text-primary">{MODE_INFO[m.value].what}</span></div>
                  <div><span className="text-text-muted font-medium">How:</span> <span className="text-text-primary">{MODE_INFO[m.value].how}</span></div>
                  <div><span className="text-text-muted font-medium">Expected output:</span> <span className="text-text-primary">{MODE_INFO[m.value].expected}</span></div>
                </div>
              </div>
            )}
          </div>
        ))}
        <div className="text-[11px] text-text-muted/70">
          {currentMode === 'end_to_end' ? `VLM: ${currentVlmModel}` :
           currentMode === 'hybrid' ? `PaddleOCR + ${currentVlmModel} VLM` :
           currentMode === 'graph' ? `PaddleOCR (spatial graph) + ${currentVlmModel}` : ''}
        </div>
      </div>

      {/* ── Embedding Model ── */}
      {currentMode !== 'end_to_end' && (
      <div className="px-4 py-3 border-b border-border/50 space-y-2 shrink-0">
        <div className="text-xs font-semibold text-text-muted">Embedding Model</div>
        <div className="space-y-0.5">
          {embeddingModels.map(m => (
            <button key={m.id} onClick={() => setCurrentEmbeddingModel(m.id)}
              className={`w-full flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs transition-colors
                ${currentEmbeddingModel === m.id ? 'bg-accent-violet/15 text-accent-violet border border-accent-violet/30' : 'text-text-muted hover:text-text-primary hover:bg-bg-elevated/40 border border-transparent'}`}>
              <span className="flex-1 text-left truncate">{m.name}</span>
              {currentEmbeddingModel === m.id && <span className="text-accent-violet text-lg leading-none">·</span>}
            </button>
          ))}
        </div>
      </div>
      )}

      {/* ── VLM Model ── */}
      <div className="px-4 py-3 border-b border-border/50 space-y-2 shrink-0">
        <div className="text-xs font-semibold text-text-muted">VLM Model (OCR)</div>
        <div className="space-y-0.5">
          {vlmModels.map(m => (
            <button key={m} onClick={() => setCurrentVlmModel(m)}
              className={`w-full flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs transition-colors
                ${currentVlmModel === m ? 'bg-accent-violet/15 text-accent-violet border border-accent-violet/30' : 'text-text-muted hover:text-text-primary hover:bg-bg-elevated/40 border border-transparent'}`}>
              <span className="flex-1 text-left truncate">{m}</span>
              {currentVlmModel === m && <span className="text-accent-violet text-lg leading-none">·</span>}
            </button>
          ))}
        </div>
      </div>

      {/* ── Target Fields ── */}
      <div className="px-4 py-3">
        <div className="flex items-center justify-between mb-2">
          <div className="text-xs font-semibold text-text-muted">Target Fields</div>
          <div className="flex gap-2">
            <Button variant="link" size="sm" onClick={() => sendFields(allFieldKeys)}
              className="text-xs text-text-muted hover:text-accent-violet">All</Button>
            <Button variant="link" size="sm" onClick={() => sendFields([])}
              className="text-xs text-text-muted hover:text-accent-coral">None</Button>
          </div>
        </div>
        {FIELD_CATEGORIES.map(cat => (
          <div key={cat.label} className="mb-2">
            <div className="text-[13px] text-text-muted uppercase tracking-wider mb-1 font-semibold">{cat.label}</div>
            <div className="space-y-0.5">
              {cat.fields.map(f => {
                const checked = selectedFields.includes(f)
                return (
                  <button key={f} onClick={() => onFieldToggle(f)}
                    className={`w-full flex items-center gap-2 px-2 py-1 rounded text-xs text-left transition-colors ${checked ? 'bg-accent-violet/15 text-accent-violet' : 'text-text-muted hover:text-text-primary hover:bg-bg-elevated/50'}`}>
                    <HugeiconsIcon icon={CheckmarkCircleIcon} className={`size-3.5 shrink-0 ${checked ? 'opacity-100' : 'opacity-30'}`} />
                    <span>{f.replace(/_/g, ' ')}</span>
                  </button>
                )
              })}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
