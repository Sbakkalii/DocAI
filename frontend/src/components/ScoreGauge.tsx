import { Button } from "@/components/ui/button"
import { useState } from 'react'
import { HugeiconsIcon } from "@hugeicons/react"
import { ChevronRightIcon, AlertCircleIcon, InformationCircleIcon } from "@hugeicons/core-free-icons"
import { Tooltip, TooltipTrigger, TooltipContent } from '@/components/ui/tooltip'
import type { ValidationIssue } from '../types'

export function ScoreGauge({ label, score, subtitle, info }: { label: string; score: number | null; subtitle?: string; info?: string }) {
  const pct = score != null ? Math.round(score * 100) : 0
  const color = score == null ? 'bg-bg-elevated' : score >= 0.8 ? 'bg-accent-green' : score >= 0.5 ? 'bg-accent-yellow' : 'bg-accent-coral'
  return (
    <div className="bg-bg-surface border border-border rounded-xl p-4">
      <div className="flex items-center gap-1 text-xs font-semibold text-text-muted uppercase tracking-wider mb-1">
        {label}
        {info && (
          <Tooltip>
            <TooltipTrigger asChild>
              <span className="inline-flex items-center cursor-help text-text-muted/40 hover:text-text-muted/80 transition-colors">
                <HugeiconsIcon icon={InformationCircleIcon} className="size-3" />
              </span>
            </TooltipTrigger>
            <TooltipContent side="top" className="max-w-[280px] text-xs leading-relaxed">
              {info}
            </TooltipContent>
          </Tooltip>
        )}
      </div>
      <div className="flex items-baseline gap-2">
        <span className={`text-2xl font-bold font-mono tabular-nums ${score == null ? 'text-text-muted' : score >= 0.8 ? 'text-accent-violet' : score >= 0.5 ? 'text-accent-yellow' : 'text-accent-coral'}`}>
          {score != null ? `${pct}%` : 'N/A'}
        </span>
        {subtitle && <span className="text-xs text-text-muted">{subtitle}</span>}
      </div>
      <div className="mt-2 h-1.5 bg-border rounded-full overflow-hidden">
        <div className={`h-full rounded-full transition-all duration-500 ${color}`} style={{ width: `${pct}%` }} />
      </div>
    </div>
  )
}

export function FieldMetricRow({ name, metric }: { name: string; metric: { count: number; exact_match: number; avg_token_f1: number; entries?: Array<Record<string, unknown>> } }) {
  const [open, setOpen] = useState(false)
  const f1 = metric.avg_token_f1
  const f1Pct = Math.round(f1 * 100)
  const exactPct = Math.round(metric.exact_match * 100)
  const barColor = f1 >= 0.8 ? 'bg-accent-green' : f1 >= 0.5 ? 'bg-accent-yellow' : 'bg-accent-coral'
  const f1Color = f1 >= 0.8 ? 'text-accent-violet' : f1 >= 0.5 ? 'text-accent-yellow' : 'text-accent-coral'
  return (
    <div className="border border-border rounded-lg overflow-hidden">
      <Button variant="ghost" onClick={() => setOpen(!open)} className="w-full flex items-center gap-3 px-4 py-2.5 text-left">
        <span className="text-xs font-semibold text-text-primary min-w-[9rem]">{name}</span>
        <div className="flex-1 h-2 bg-border rounded-full overflow-hidden">
          <div className={`h-full rounded-full transition-all ${barColor}`} style={{ width: `${f1Pct}%` }} />
        </div>
        <span className={`text-xs font-mono font-bold tabular-nums ${f1Color}`}>{f1Pct}%</span>
        <span className="text-xs text-text-muted font-mono tabular-nums">exact {exactPct}%</span>
        <span className="text-xs text-text-muted font-mono">({metric.count})</span>
        <HugeiconsIcon icon={ChevronRightIcon} className={`size-3.5 text-text-muted transition-transform duration-200 ${open ? 'rotate-90' : ''}`} />
      </Button>
      {open && metric.entries && (
        <div className="border-t border-border divide-y divide-border/50 animate-fade-in">
          {metric.entries.map((e, i) => (
            <div key={i} className="px-4 py-2 text-xs grid grid-cols-[1fr_1fr_2rem] gap-2">
              <div className="text-text-muted truncate">GT: <span className="text-text-primary font-mono">{String(e.gt ?? '(missing)')}</span></div>
              <div className="text-text-muted truncate">Pred: <span className={e.exact ? 'text-accent-violet font-mono' : 'text-accent-coral font-mono'}>{String(e.pred ?? '(none)')}</span></div>
              <div className="text-right font-mono text-xs">{e.exact ? <span className="text-accent-violet font-bold">✓</span> : `F1 ${(e.token_f1 as number).toFixed(2)}`}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

export function IssueRow({ issue, type }: { issue: ValidationIssue; type: 'error' | 'warning' }) {
  const [expanded, setExpanded] = useState(false)
  return (
    <div className="hover:bg-bg-elevated/30 transition-colors">
      <Button variant="ghost" onClick={() => setExpanded(!expanded)} className="w-full flex items-start gap-3 px-5 py-3 text-left">
        <HugeiconsIcon icon={AlertCircleIcon} className={`size-4 mt-0.5 shrink-0 ${type === 'error' ? 'text-accent-coral' : 'text-accent-yellow'}`} />
        <div className="flex-1 min-w-0">
          <div className="text-sm text-text-primary">{issue.message}</div>
          <div className="text-xs text-text-muted mt-0.5 font-mono">{issue.rule}</div>
          {expanded && issue.fields.length > 0 && (
            <div className="mt-2 flex flex-wrap gap-1.5 animate-fade-in">
              {issue.fields.map(f => <span key={f} className="inline-flex items-center gap-1 px-2 py-0.5 text-xs font-medium rounded-full bg-bg-elevated text-text-muted font-mono">{f}</span>)}
            </div>
          )}
        </div>
        <HugeiconsIcon icon={ChevronRightIcon} className={`size-3.5 text-text-muted mt-1 transition-transform duration-200 shrink-0 ${expanded ? 'rotate-90' : ''}`} />
      </Button>
    </div>
  )
}
