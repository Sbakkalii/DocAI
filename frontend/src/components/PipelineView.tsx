import { useState, useEffect, useRef, useCallback, useMemo } from 'react'
import { createPortal } from 'react-dom'
import { HugeiconsIcon } from '@hugeicons/react'
import {
  Loading01Icon, CheckmarkCircleIcon, AlertCircleIcon, EyeIcon, EyeOffIcon,
  CodeIcon, AiNetworkIcon, Image01Icon,
  ChevronRightIcon, RotateLeft01Icon, PlayIcon, FastForwardIcon, Clock01Icon,
  InformationCircleIcon, Cancel01Icon, BubbleChatIcon, MailSend01Icon,
  PencilIcon,
} from '@hugeicons/core-free-icons'
import KnowledgeGraphView from './KnowledgeGraph'
import MarkdownPreview from './MarkdownPreview'
import { SectionCard } from './SectionCard'
import { ScoreGauge, FieldMetricRow, IssueRow } from './ScoreGauge'
import { ErrorBoundary } from './ErrorBoundary'
import { QAMessage } from './QAMessage'
import { PipelineSidebar } from './PipelineSidebar'
import { StepInfoTooltip } from './StepInfoTooltip'
import { DEFAULT_FIELDS, STEP_LABELS, STEP_ORDER, STEP_GROUPS, getPreprocSteps, getEnabledSteps, fmtTime, DEFAULT_QA_PROMPT, METRIC_INFO, CATEGORY_METRICS, getDownstream, MULTI_TASK_TASK_OPTIONS, AVAILABLE_VLM_MODELS } from './constants'
import { Button } from '@/components/ui/button'
import { Popover, PopoverTrigger, PopoverContent } from '@/components/ui/popover'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { PDFViewer, type PDFViewerHandle } from '@/components/ui/pdf-viewer'
import { HumanReviewPanel, HumanReviewHighlight, type ReviewField, type ReviewFieldSchema, type ReviewLocation } from '@/components/ui/bounding-box-citations'
import { getOcrBlocks, OcrBlocksPanel, OcrBlockOverlay, blockToArea, type ParsedOcrOutput, type ParsedOcrBlock, type OcrBlock } from '@/components/ui/layout-blocks'



import type { StepState, PipelineResult, PageResult, ValidationIssue } from '../types'

/* ── StepCard / StepHeader helpers ── */

export function StepHeader({ label }: { label: string }) {
  return (
    <div className="flex items-center gap-1.5 mb-1.5 px-1">
      <div className="w-1.5 h-1.5 rounded-full bg-accent-violet/60" />
      <span className="text-xs font-semibold text-accent-violet uppercase tracking-wider">{label}</span>
    </div>
  )
}

export function StepCard({
  name, step, runningStep, selectedStep, onSelectStep,
  infoTarget, setInfoTarget,
  ocrPostCorrect, visionOcrPostCorrect, embeddingTextSource, retrievalStrategy,
  waiting,
  runStep, stopPipeline,
}: {
  name: string
  step?: StepState
  runningStep: string | null
  selectedStep: string | null
  onSelectStep: (s: string | null) => void
  infoTarget: { step: string; top: number; left: number; width: number } | null
  setInfoTarget: (target: { step: string; top: number; left: number; width: number } | null) => void
  ocrPostCorrect: boolean
  visionOcrPostCorrect: boolean
  embeddingTextSource: string
  retrievalStrategy: string
  waiting: boolean
  runStep: (step: string, config?: Record<string, unknown>) => void
  stopPipeline: () => void
}) {
  const isDone = step?.status === 'completed'
  const isRunning = step?.status === 'running' || runningStep === name
  const isFailed = step?.status === 'failed'
  const isSelected = selectedStep === name
  const canRun = !isDone && !isRunning && waiting

  const [configOpen, setConfigOpen] = useState(false)
  const [selectedTasks, setSelectedTasks] = useState<string[]>(MULTI_TASK_TASK_OPTIONS.map(t => t.id))

  const borderL = isSelected ? 'border-l-accent-violet' : isDone ? 'border-l-[#16A34A]/60' : isRunning ? 'border-l-accent-violet animate-border-pulse' : isFailed ? 'border-l-accent-coral' : 'border-l-bg-elevated'
  const bg = isSelected ? 'bg-accent-violet/12 ring-1 ring-accent-violet/30' : isDone ? 'bg-bg-surface/60 hover:bg-bg-elevated/60' : isRunning ? 'bg-accent-violet/8 hover:bg-accent-violet/12' : isFailed ? 'bg-accent-coral/15 hover:bg-accent-coral/25' : 'bg-bg-surface/30 hover:bg-bg-elevated/40'
  const border = isSelected ? 'border-accent-violet/40' : isDone ? 'border-border/40' : isRunning ? 'border-accent-violet/25' : isFailed ? 'border-accent-coral/30' : 'border-border/20'

  return (
    <div
      onClick={() => onSelectStep(isSelected ? null : name)}
      className={`flex items-center gap-2 px-3 py-2 rounded-lg border ${border} ${bg} ${borderL} border-l-2 transition-all cursor-pointer shrink-0 group`}>
      <div className="w-5 h-5 flex items-center justify-center shrink-0">
        {isRunning ? (
          <HugeiconsIcon icon={Loading01Icon} className="size-3.5 animate-spin text-accent-violet" />
        ) : isDone ? (
          <HugeiconsIcon icon={CheckmarkCircleIcon} className="size-3.5 text-accent-green" />
        ) : isFailed ? (
          <HugeiconsIcon icon={AlertCircleIcon} className="size-3.5 text-accent-coral" />
        ) : (
          <div className="w-2.5 h-2.5 rounded-full bg-border" />
        )}
      </div>

      <span className={`text-xs font-medium truncate flex-1 ${
        isDone ? 'text-text-primary' : isRunning ? 'text-accent-violet' : isFailed ? 'text-accent-coral' : 'text-text-muted'
      }`}>{STEP_LABELS[name]}</span>

      {name === 'multi_task' && selectedTasks.length > 0 && (
        <div className="flex items-center gap-1 shrink-0">
          {selectedTasks.map(t => (
            <span key={t} className="text-[10px] px-1.5 py-0.5 rounded-full bg-accent-violet/10 text-accent-violet border border-accent-violet/20">{t.replace(/_/g, ' ')}</span>
          ))}
        </div>
      )}

      {isDone && step?.elapsed != null && (
        <span className="text-xs text-text-muted tabular-nums shrink-0">{fmtTime(step.elapsed)}</span>
      )}

      <Button variant="ghost" size="icon-sm" onClick={e => {
          e.stopPropagation()
          const rect = (e.currentTarget as HTMLElement).getBoundingClientRect()
          setInfoTarget(infoTarget?.step === name ? null : {
            step: name,
            top: rect.bottom,
            left: rect.left,
            width: Math.max(rect.width, 280),
          })
        }}
        title={`About ${STEP_LABELS[name]}`}>
        <HugeiconsIcon icon={InformationCircleIcon} className="size-3.5" />
      </Button>

      {name === 'multi_task' && (
        <Popover open={configOpen} onOpenChange={setConfigOpen}>
          <PopoverTrigger asChild>
            <Button variant="ghost" size="icon-sm" onClick={e => { e.stopPropagation() }}
              title={`Configure ${STEP_LABELS[name]}`}>
              <HugeiconsIcon icon={CodeIcon} className="size-3.5" />
            </Button>
          </PopoverTrigger>
          <PopoverContent side="bottom" align="end" className="w-72 p-3 space-y-3" onClick={e => e.stopPropagation()}>
            <div className="text-xs font-semibold text-text-muted uppercase tracking-wider mb-1">Tasks</div>
            <div className="space-y-1">
              {MULTI_TASK_TASK_OPTIONS.map(opt => {
                const checked = selectedTasks.includes(opt.id)
                return (
                  <button key={opt.id} onClick={() => {
                    setSelectedTasks(prev => checked ? prev.filter(t => t !== opt.id) : [...prev, opt.id])
                  }}
                    className="w-full flex items-start gap-2 px-2 py-1.5 rounded text-xs text-left transition-colors hover:bg-bg-elevated/40">
                    <div className={`w-3.5 h-3.5 shrink-0 mt-0.5 rounded border flex items-center justify-center ${checked ? 'bg-accent-violet border-accent-violet' : 'border-border'}`}>
                      {checked && <span className="text-white text-[9px] font-bold">✓</span>}
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="text-text-primary font-medium truncate">{opt.label}</div>
                      <div className="text-text-muted/70 text-[11px] leading-tight">{opt.description}</div>
                    </div>
                  </button>
                )
              })}
            </div>

            <Button variant="default" size="sm" className="w-full" onClick={e => {
              e.stopPropagation()
              setConfigOpen(false)
              const cfg: Record<string, unknown> = {
                multi_task_tasks: selectedTasks,
              }
              runStep(name, cfg)
            }}>
              <HugeiconsIcon icon={PlayIcon} className="size-3 mr-1" />
              Run Multi-Task NLP
            </Button>
          </PopoverContent>
        </Popover>
      )}

      {isRunning && (
        <Button variant="default" size="icon-sm" onClick={e => {
          e.stopPropagation(); stopPipeline()
        }}
          className="bg-accent-coral/20 text-accent-coral border border-accent-coral/40 hover:bg-accent-coral/30"
          title={`Stop ${STEP_LABELS[name]}`}>
          <HugeiconsIcon icon={Cancel01Icon} className="size-3.5" />
        </Button>
      )}
      {canRun && (
        <Button variant="default" size="icon-sm" onClick={e => {
          e.stopPropagation()
          const cfg: Record<string, unknown> = {}
          if (name === 'ocr') cfg.ocr_post_correct = ocrPostCorrect
          else if (name === 'vision_ocr') cfg.vision_ocr_post_correct = visionOcrPostCorrect
          else if (name === 'embedding') cfg.embedding_text_source = embeddingTextSource
          else if (name === 'retrieval') cfg.retrieval_strategy = retrievalStrategy
          else if (name === 'multi_task') { cfg.multi_task_tasks = selectedTasks }
          if (Object.keys(cfg).length > 0) runStep(name, cfg)
          else runStep(name)
        }}
          title={`Run ${STEP_LABELS[name]}`}>
          <HugeiconsIcon icon={PlayIcon} className="size-3" />
        </Button>
      )}
      {isDone && (
        <Button variant="ghost" size="icon-sm" onClick={e => {
          e.stopPropagation()
          const cfg: Record<string, unknown> = {}
          if (name === 'ocr') cfg.ocr_post_correct = ocrPostCorrect
          else if (name === 'vision_ocr') cfg.vision_ocr_post_correct = visionOcrPostCorrect
          else if (name === 'embedding') cfg.embedding_text_source = embeddingTextSource
          else if (name === 'retrieval') cfg.retrieval_strategy = retrievalStrategy
          else if (name === 'multi_task') { cfg.multi_task_tasks = selectedTasks }
          if (Object.keys(cfg).length > 0) runStep(name, cfg)
          else runStep(name)
        }}
          title={`Rerun ${STEP_LABELS[name]}`}>
          <HugeiconsIcon icon={RotateLeft01Icon} className="size-3" />
        </Button>
      )}
    </div>
  )
}

/* ── Ingestion Data ── */

export function IngestionDataContent({ data }: { data: Record<string, unknown> }) {
  return (
    <>
      <div className="p-5 grid grid-cols-2 gap-4">
        <div><div className="text-xs font-medium text-text-muted uppercase tracking-wider mb-0.5">Type</div><div className="text-sm font-medium text-text-primary">{String(data.document_type || 'Unknown')}</div></div>
        <div><div className="text-xs font-medium text-text-muted uppercase tracking-wider mb-0.5">Pages</div><div className="text-sm font-medium text-text-primary">{String(data.total_pages || 0)}</div></div>
      </div>
      <div className="px-5 py-3 border-t border-border bg-bg-elevated/50 rounded-b-xl flex items-center gap-2 text-xs text-text-muted">
        <HugeiconsIcon icon={CheckmarkCircleIcon} className="size-3.5 text-accent-green" />
        Document ingested — {String(data.total_pages)} page{Number(data.total_pages) !== 1 ? 's' : ''}
      </div>
    </>
  )
}

/* ── Data Card Components ── */

function DataCardOCR({ page }: { page: Record<string, unknown> | undefined }) {
  const [showFull, setShowFull] = useState(false)
  const [mode, setMode] = useState<'plain' | 'raw' | 'preview'>('preview')
  const text = (page?.text as string) || ''
  const markdown = (page?.markdown as string) || ''
  const boxes = (page?.boxes as Array<Record<string, unknown>>) || []
  const modes = [
    { key: 'preview', label: 'Preview' },
    { key: 'raw', label: 'Formatted' },
    { key: 'plain', label: 'Plain' },
  ] as const
  const content = mode === 'plain' ? text : markdown
  const displayContent = showFull ? content : content.slice(0, 3000)
  return (
    <SectionCard title="OCR Text Transcription">
      <div className="flex items-center justify-between px-5 py-2.5 border-b border-border bg-bg-elevated/50">
        <div className="flex items-center gap-3 text-xs text-text-muted">
          <HugeiconsIcon icon={Image01Icon} className="size-3.5 text-accent-violet/70" /><span>{boxes.length} text regions</span>
          <div className="flex gap-0.5 ml-1 rounded-md bg-bg-elevated border border-accent-violet/30 overflow-hidden">
            {modes.map(m => (
              <Button key={m.key} variant="ghost" size="sm" onClick={() => setMode(m.key)}
                className={mode === m.key ? '!bg-accent-violet/20 !text-accent-violet' : ''}>
                {m.label}
              </Button>
            ))}
          </div>
        </div>
        {content.length > 3000 && (
          <Button variant="link" size="sm" onClick={() => setShowFull(!showFull)}>
            {showFull ? <HugeiconsIcon icon={EyeOffIcon} className="size-3" /> : <HugeiconsIcon icon={EyeIcon} className="size-3" />}{showFull ? 'Collapse' : 'Show all'}
          </Button>
        )}
      </div>
      <div className="p-5 max-h-80 overflow-y-auto">
        {!content ? <div className="text-sm text-text-muted italic">No OCR text available</div> : mode === 'preview' ? (
          <MarkdownPreview content={displayContent} />
        ) : (
          <pre className="text-sm text-text-primary font-mono leading-relaxed whitespace-pre-wrap" style={{ fontFamily: "'JetBrains Mono', monospace" }}>
            {displayContent}{!showFull && content.length > 3000 && <span className="text-text-muted"> ... ({content.length - 3000} more chars)</span>}
          </pre>
        )}
      </div>
    </SectionCard>
  )
}

function DataCardVisionOCR({ page }: { page: Record<string, unknown> | undefined }) {
  const [showFull, setShowFull] = useState(false)
  const [mode, setMode] = useState<'raw' | 'plain'>('raw')
  const text = (page?.text as string) || ''
  const markdown = (page?.markdown as string) || ''
  const model = (page?.model as string) || ''
  const content = mode === 'plain' ? text : markdown
  const displayContent = showFull ? content : content.slice(0, 3000)
  return (
    <SectionCard title="Vision OCR (VLM)">
      <div className="flex items-center justify-between px-5 py-2.5 border-b border-border bg-bg-elevated/50">
        <div className="flex items-center gap-3 text-xs text-text-muted">
          <HugeiconsIcon icon={EyeIcon} className="size-3.5 text-accent-violet/70" /><span>{model || 'VLM'}</span>
          <div className="flex gap-0.5 ml-1 rounded-md bg-bg-elevated border border-accent-violet/30 overflow-hidden">
            {([{ key: 'raw', label: 'Formatted' }, { key: 'plain', label: 'Plain' }] as const).map(m => (
              <Button key={m.key} variant="ghost" size="sm" onClick={() => setMode(m.key)}
                className={mode === m.key ? '!bg-accent-violet/20 !text-accent-violet' : ''}>
                {m.label}
              </Button>
            ))}
          </div>
        </div>
        {content.length > 3000 && (
          <Button variant="link" size="sm" onClick={() => setShowFull(!showFull)}>
            {showFull ? <HugeiconsIcon icon={EyeOffIcon} className="size-3" /> : <HugeiconsIcon icon={EyeIcon} className="size-3" />}{showFull ? 'Collapse' : 'Show all'}
          </Button>
        )}
      </div>
      <div className="p-5 max-h-80 overflow-y-auto">
        {!content ? <div className="text-sm text-text-muted italic">No VLM text available</div> : (
          <pre className="text-sm text-text-primary font-mono leading-relaxed whitespace-pre-wrap" style={{ fontFamily: "'JetBrains Mono', monospace" }}>
            {displayContent}{!showFull && content.length > 3000 && <span className="text-text-muted"> ... ({content.length - 3000} more chars)</span>}
          </pre>
        )}
      </div>
    </SectionCard>
  )
}

function DataCardHybridOCR({ page }: { page: Record<string, unknown> | undefined }) {
  const [showFull, setShowFull] = useState(false)
  const text = (page?.text as string) || ''
  const markdown = (page?.markdown as string) || ''
  const hybridUsed = page?.hybrid_used as boolean
  const displayContent = showFull ? markdown : markdown.slice(0, 3000)
  return (
    <SectionCard title="Hybrid OCR (PaddleOCR + VLM)">
      <div className="flex items-center justify-between px-5 py-2.5 border-b border-border bg-bg-elevated/50">
        <div className="flex items-center gap-2 text-xs text-text-muted">
          {hybridUsed && <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded bg-accent-green/10 text-accent-green text-xs font-medium">VLM overlay</span>}
          <span>{text.length} chars</span>
        </div>
        {markdown.length > 3000 && (
          <Button variant="link" size="sm" onClick={() => setShowFull(!showFull)}>
            {showFull ? <HugeiconsIcon icon={EyeOffIcon} className="size-3" /> : <HugeiconsIcon icon={EyeIcon} className="size-3" />}{showFull ? 'Collapse' : 'Show all'}
          </Button>
        )}
      </div>
      <div className="p-5 max-h-80 overflow-y-auto">
        <pre className="text-sm text-text-primary font-mono leading-relaxed whitespace-pre-wrap" style={{ fontFamily: "'JetBrains Mono', monospace" }}>
          {displayContent || <span className="text-text-muted italic">No hybrid result</span>}
          {!showFull && markdown.length > 3000 && <span className="text-text-muted"> ... ({markdown.length - 3000} more chars)</span>}
        </pre>
      </div>
    </SectionCard>
  )
}

function DataCardDocGraph({ page }: { page: Record<string, unknown> | undefined }) {
  const graph = (page?.graph as Record<string, unknown>) || {}
  const markdown = (page?.markdown as string) || ''
  const text = (page?.text as string) || ''
  const nodeCount = graph?.node_count as number || 0
  const edgeCount = graph?.edge_count as number || 0
  const tables = (graph?.tables as Array<Record<string, unknown>>) || []
  const kvPairs = (graph?.kv_pairs as Array<Record<string, unknown>>) || []
  return (
    <SectionCard title="Document Graph">
      <div className="flex items-center gap-4 px-5 py-3 border-b border-border bg-bg-elevated/50">
        <span className="text-xs text-text-muted">{nodeCount} nodes</span>
        <span className="text-xs text-text-muted">{edgeCount} edges</span>
        <span className="text-xs text-text-muted">{tables.length} tables</span>
        <span className="text-xs text-text-muted">{kvPairs.length} KV pairs</span>
      </div>
      {kvPairs.length > 0 && (
        <div className="px-5 py-3 border-b border-border">
          <div className="text-xs font-semibold text-text-muted uppercase tracking-wider mb-2">Key-Value Pairs</div>
          <div className="space-y-1">{kvPairs.map((kv, i) => (
            <div key={i} className="flex items-center gap-2 text-xs">
              <span className="font-medium text-text-primary">{String(kv.label)}</span>
              <span className="text-text-muted">→</span>
              <span className="text-text-muted">{String(kv.value)}</span>
            </div>
          ))}</div>
        </div>
      )}
      {tables.length > 0 && (
        <div className="px-5 py-3 border-b border-border">
          <div className="text-xs font-semibold text-text-muted uppercase tracking-wider mb-2">Detected Tables</div>
          {tables.map((t, i) => {
            const rows = t.rows as number || 0
            const cols = t.cols as number || 0
            return (
              <div key={i} className="text-xs text-text-muted mb-1">
                Table {i + 1}: {rows} rows × {cols} columns
              </div>
            )
          })}
        </div>
      )}
      <div className="p-5 max-h-80 overflow-y-auto">
        <pre className="text-xs text-text-primary font-mono leading-relaxed whitespace-pre-wrap" style={{ fontFamily: "'JetBrains Mono', monospace" }}>
          {markdown || text || <span className="text-text-muted italic">No graph reconstruction</span>}
        </pre>
      </div>
    </SectionCard>
  )
}

function DataCardE2EVLM({ page }: { page: Record<string, unknown> | undefined }) {
  const fields = (page?.fields as Record<string, unknown>) || {}
  return (
    <SectionCard title="End-to-End VLM Extraction">
      {Object.keys(fields).length === 0 ? (
        <div className="p-5 text-sm text-text-muted italic">No fields extracted</div>
      ) : (
        <div className="divide-y divide-border">
          {Object.entries(fields).filter(([k]) => k !== '_raw' && k !== '_evidence').map(([field, value]) => (
            <div key={field} className="flex items-start gap-3 px-5 py-3 hover:bg-bg-elevated/30 transition-colors">
              <div className="w-32 shrink-0"><span className="text-[13px] font-medium text-text-muted uppercase tracking-wider">{field.replace(/_/g, ' ')}</span></div>
              <div className="flex-1 min-w-0">
                {Array.isArray(value) ? (
                  <div className="space-y-1">{(value as Array<unknown>).map((item, i) => (
                      <div key={i} className="text-sm text-text-primary font-mono bg-bg-elevated/50 px-3 py-1.5 rounded-lg border border-border/50" style={{ fontFamily: "'JetBrains Mono', monospace" }}>{typeof item === 'object' && item !== null ? JSON.stringify(item) : String(item)}</div>
                    ))}</div>
                  ) : <span className="text-sm text-text-primary font-mono" style={{ fontFamily: "'JetBrains Mono', monospace" }}>{typeof value === 'object' && value !== null ? JSON.stringify(value) : String(value)}</span>}
              </div>
            </div>
          ))}
        </div>
      )}
    </SectionCard>
  )
}

function DataCardRetrieval({ page }: { page: Record<string, unknown> | undefined }) {
  const raw = (page?.examples as Array<Record<string, unknown>>) || []
  const [expanded, setExpanded] = useState<number | null>(null)
  return (
    <SectionCard title="Retrieved Examples (Few-Shot)">
      {raw.length === 0 ? <div className="p-5 text-sm text-text-muted italic">No examples retrieved</div> : (
        <div className="divide-y divide-border">
          {raw.map((ex, i) => {
            const imgPath = typeof ex.image_path === 'string' ? ex.image_path : ''
            const source = typeof ex.source === 'string' ? ex.source : ''
            const ocrText = typeof ex.ocr_text === 'string' ? ex.ocr_text : ''
            const fields = ex.fields && typeof ex.fields === 'object' ? ex.fields as Record<string, string> : undefined
            return (
              <div key={i} className="hover:bg-bg-elevated/30 transition-colors">
                <div className="px-5 py-3">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-[13px] font-semibold text-accent-violet bg-accent-violet/10 px-2 py-0.5 rounded-md">Example {i + 1}</span>
                    {source && <span className="text-xs text-text-muted font-mono">{source}</span>}
                  </div>
                  <pre className="text-xs text-text-muted font-mono leading-relaxed whitespace-pre-wrap line-clamp-3 mb-2" style={{ fontFamily: "'JetBrains Mono', monospace" }}>{ocrText}</pre>
                  {fields && Object.keys(fields).length > 0 && (
                    <div className="flex flex-wrap gap-1.5 mb-2">
                      {Object.entries(fields).map(([k, v]) => (
                        <span key={k} className="text-xs bg-bg-elevated text-text-muted px-1.5 py-0.5 rounded font-mono">{k}: {String(v).slice(0, 30)}</span>
                      ))}
                    </div>
                  )}
                  {imgPath && (
                    <Button variant="link" size="sm" onClick={() => setExpanded(expanded === i ? null : i)}>
                      {expanded === i ? 'Hide source' : 'Show source document'}
                    </Button>
                  )}
                  {imgPath && expanded === i && (
                    <div className="mt-2 border border-border rounded-lg overflow-hidden animate-fade-in">
                      <img src={`/api/image/${encodeURIComponent(imgPath)}`} alt="Source" className="w-full h-auto max-h-72 object-contain bg-bg-base" />
                    </div>
                  )}
                </div>
              </div>
            )
          })}
        </div>
      )}
    </SectionCard>
  )
}

function DataCardRAG({ page }: { page: Record<string, unknown> | undefined }) {
  const rawRules = (page?.rules as Array<Record<string, unknown>>) || []
  const rawTemplates = (page?.templates as Array<Record<string, unknown>>) || []
  return (
    <SectionCard title="RAG Context">
      <div className="p-5 space-y-4">
        <div><div className="text-[13px] font-semibold text-text-muted uppercase tracking-wider mb-2">Field Rules ({rawRules.length})</div>
          {rawRules.length === 0 ? <p className="text-xs text-text-muted italic">No rules</p> : (
            <div className="space-y-1.5">{rawRules.map((r, i) => (
              <div key={i} className="text-xs bg-bg-elevated/50 px-3 py-2 rounded-lg border border-border/50 space-y-0.5">
                <span className="font-semibold text-text-primary">{String(r.field_name ?? '')}</span>
                <p className="text-text-muted">{String(r.description ?? '')}</p>
                {Array.isArray(r.format_patterns) && r.format_patterns.length > 0 && (
                  <p className="text-text-muted text-xs font-mono">Formats: {(r.format_patterns as string[]).join(', ')}</p>
                )}
              </div>
            ))}</div>
          )}
        </div>
        <div><div className="text-[13px] font-semibold text-text-muted uppercase tracking-wider mb-2">Templates ({rawTemplates.length})</div>
          {rawTemplates.length === 0 ? <p className="text-xs text-text-muted italic">No templates</p> : (
            <div className="space-y-1.5">{rawTemplates.map((t, i) => (
              <div key={i} className="text-xs bg-bg-elevated/50 px-3 py-2 rounded-lg border border-border/50 flex items-center gap-2">
                <span className="font-semibold text-text-primary">{String(t.template_id ?? '')}</span>
                <span className="text-text-muted">— {String(t.description ?? '')}</span>
              </div>
            ))}</div>
          )}
        </div>
      </div>
    </SectionCard>
  )
}

function DataCardLLM({ page, onFieldSelect }: { page: Record<string, unknown> | undefined; onFieldSelect?: (field: string) => void }) {
  const [showPrompt, setShowPrompt] = useState(false)
  const fields = (page?.fields as Record<string, unknown>) || {}
  const prompt = (page?.prompt as string) || ''
  return (
    <div className="space-y-3">
      <SectionCard title="Extracted Fields">
        {Object.keys(fields).length === 0 ? <div className="p-5 text-sm text-text-muted italic">No fields extracted</div> : (
          <div className="divide-y divide-border">
            {Object.entries(fields).map(([field, value]) => (
              <div key={field} className="flex items-start gap-3 px-5 py-3 hover:bg-bg-elevated/30 transition-colors">
                <div className="w-32 shrink-0"><span className="text-[13px] font-medium text-text-muted uppercase tracking-wider">{field.replace(/_/g, ' ')}</span></div>
                <div className="flex-1 min-w-0">
                  {Array.isArray(value) ? (
                    <div className="space-y-1">{(value as Array<Record<string, unknown>>).map((item, i) => (
                      <div key={i} className="text-sm text-text-primary font-mono bg-bg-elevated/50 px-3 py-1.5 rounded-lg border border-border/50" style={{ fontFamily: "'JetBrains Mono', monospace" }}>{JSON.stringify(item)}</div>
                    ))}</div>
                  ) : (
                    <Button variant="secondary" size="sm" onClick={() => onFieldSelect?.(field)}
                      className="w-full !text-left font-mono"
                      style={{ fontFamily: "'JetBrains Mono', monospace" }}>
                      {typeof value === 'object' && value !== null ? JSON.stringify(value) : String(value)}
                    </Button>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
      </SectionCard>
      {prompt && (
        <SectionCard title="">
          <Button variant="ghost" size="sm" onClick={() => setShowPrompt(!showPrompt)} className="w-full !justify-between !px-5 !py-3">
            <div className="flex items-center gap-2"><HugeiconsIcon icon={CodeIcon} className="size-4 text-text-muted" /><h3 className="text-sm font-semibold text-text-primary">LLM Prompt</h3></div>
            <div className="flex items-center gap-1 text-xs text-text-muted">{showPrompt ? <HugeiconsIcon icon={EyeOffIcon} className="size-3" /> : <HugeiconsIcon icon={EyeIcon} className="size-3" />}{showPrompt ? 'Hide' : 'Show'}</div>
          </Button>
          {showPrompt && (
            <div className="px-5 pb-4 animate-fade-in">
              <pre className="text-xs text-text-muted font-mono leading-relaxed whitespace-pre-wrap bg-bg-base p-4 rounded-lg border border-border max-h-96 overflow-y-auto" style={{ fontFamily: "'JetBrains Mono', monospace" }}>{prompt}</pre>
            </div>
          )}
        </SectionCard>
      )}
    </div>
  )
}

function DataCardValidation({ page }: { page: Record<string, unknown> | undefined }) {
  const v = page?.validation as Record<string, unknown> | undefined
  if (!v) return <div className="text-sm text-text-muted italic p-4">No validation results</div>
  const allIssues = (v.issues as Array<Record<string, unknown>>) || []
  const errors = allIssues.filter((i: Record<string, unknown>) => i.severity === 'error')
  const warnings = allIssues.filter((i: Record<string, unknown>) => i.severity === 'warning')
  return (
    <SectionCard title="Field Validation">
      <div className="flex items-center gap-3 px-5 py-3 border-b border-border">
        <div className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium ${v.is_valid ? 'bg-accent-green/10 text-accent-green' : 'bg-accent-yellow/10 text-accent-yellow'}`}>
          {v.is_valid ? <HugeiconsIcon icon={CheckmarkCircleIcon} className="size-3" /> : <HugeiconsIcon icon={AlertCircleIcon} className="size-3" />}{v.is_valid ? 'Valid' : 'Issues'}
        </div>
      </div>
      {allIssues.length === 0 ? <div className="p-5 text-sm text-text-muted italic">All checks passed</div> : (
        <div className="divide-y divide-border">
          {errors.map((issue, i) => <IssueRow key={`e-${i}`} issue={issue as unknown as ValidationIssue} type="error" />)}
          {warnings.map((issue, i) => <IssueRow key={`w-${i}`} issue={issue as unknown as ValidationIssue} type="warning" />)}
        </div>
      )}
    </SectionCard>
  )
}

function DataCardConfidence({ page }: { page: Record<string, unknown> | undefined }) {
  const overall = page?.overall_confidence as number | null | undefined
  const needsReview = page?.needs_review as boolean | undefined
  const fieldConf = page?.field_confidence as Record<string, { confidence: number; level: string; needs_review: boolean; signals: { ocr_confidence: number; evidence_match: number; format_valid: number } }> | undefined

  if (!fieldConf || Object.keys(fieldConf).length === 0) return <div className="text-sm text-text-muted italic p-4">No confidence scores</div>

  const pct = overall != null ? Math.round(overall * 100) : 0
  const color = overall == null ? 'bg-border' : overall >= 0.85 ? 'bg-accent-green' : overall >= 0.7 ? 'bg-accent-yellow' : 'bg-accent-coral'
  const level = overall == null ? 'N/A' : overall >= 0.85 ? 'High' : overall >= 0.7 ? 'Medium' : 'Low'

  return (
    <SectionCard title={`Calibrated Confidence — ${level}`}>
      <div className="p-5 space-y-4">
        <div className="flex items-center gap-4">
          <div className="flex-1">
            <div className="flex items-center justify-between mb-1">
              <span className="text-xs font-semibold text-text-muted uppercase">Overall</span>
              <span className={`text-xs font-mono font-bold ${overall == null ? 'text-text-muted' : overall >= 0.85 ? 'text-accent-violet' : overall >= 0.7 ? 'text-accent-yellow' : 'text-accent-coral'}`}>{overall != null ? `${pct}%` : 'N/A'}</span>
            </div>
            <div className="h-2 bg-border rounded-full overflow-hidden">
              <div className={`h-full rounded-full transition-all ${color}`} style={{ width: `${pct}%` }} />
            </div>
          </div>
          {needsReview && (
            <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-xs font-medium bg-accent-coral/15 text-accent-coral border border-accent-coral/30 shrink-0">
              Needs Review
            </span>
          )}
        </div>
        <div className="text-xs text-text-muted">3-signal weighted score: OCR confidence (40%) + evidence fuzzy match (40%) + format validity (20%)</div>
        <div className="border-t border-border pt-3">
          <div className="text-xs font-semibold text-text-muted uppercase mb-2">Per-Field Scores</div>
          <div className="space-y-1.5">
            {Object.entries(fieldConf).filter(([k]) => !k.startsWith('LINE/')).sort(([, a], [, b]) => a.confidence - b.confidence).map(([name, c]) => (
              <div key={name} className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-bg-elevated/30">
                <span className="text-xs font-medium text-text-primary w-32 shrink-0 truncate uppercase">{name.replace(/_/g, ' ')}</span>
                <div className="flex-1 h-1.5 bg-border rounded-full overflow-hidden">
                  <div className={`h-full rounded-full ${c.confidence >= 0.85 ? 'bg-accent-green' : c.confidence >= 0.7 ? 'bg-accent-yellow' : 'bg-accent-coral'}`} style={{ width: `${Math.round(c.confidence * 100)}%` }} />
                </div>
                <span className={`text-xs font-mono font-bold tabular-nums ${c.confidence >= 0.85 ? 'text-accent-violet' : c.confidence >= 0.7 ? 'text-accent-yellow' : 'text-accent-coral'}`}>{Math.round(c.confidence * 100)}%</span>
                <div className="flex items-center gap-1 text-[11px] text-text-muted/70 font-mono">
                  <span title="OCR confidence">ocr:{Math.round(c.signals.ocr_confidence * 100)}</span>
                  <span title="Evidence match">match:{Math.round(c.signals.evidence_match * 100)}</span>
                  <span title="Format valid">{c.signals.format_valid ? '✓' : '✗'}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </SectionCard>
  )
}

function DataCardExport({ data, sessionId }: { data: Record<string, unknown>; sessionId: string }) {
  const formats = (data.formats as string[]) || []

  return (
    <SectionCard title="ERP Export"><div className="p-5 space-y-3">
      {formats.length === 0 ? <div className="text-sm text-text-muted italic">No exports generated</div> : (
        formats.map(fmt => (
          <div key={fmt} className="flex items-center justify-between px-4 py-2.5 bg-bg-elevated/30 rounded-lg">
            <div>
              <div className="text-sm text-text-primary font-medium">{fmt === 'ubl_xml' ? 'UBL 2.1 XML' : fmt === 'edi810' ? 'EDI 810' : 'CSV'}</div>
              <div className="text-xs text-text-muted">{fmt === 'ubl_xml' ? 'European e-invoice standard (EN 16931)' : fmt === 'edi810' ? 'US invoice format (ANSI X12)' : 'Configurable CSV'}</div>
            </div>
            <Button variant="default" size="sm" onClick={() => window.open(`/api/session/${sessionId}/export/${fmt === 'ubl_xml' ? 'ubl_xml' : fmt === 'edi810' ? 'edi810' : 'csv'}`)}>
              Download
            </Button>
          </div>
        ))
      )}
    </div></SectionCard>
  )
}

function DataCardVendor({ page }: { page: Record<string, unknown> | undefined }) {
  const match = page?.vendor_match as Record<string, unknown> | undefined
  const anomalies = (page?.vendor_anomalies as Array<Record<string, unknown>>) || []
  const isMatch = !!match

  return (
    <SectionCard title="Vendor Registry Lookup"><div className="p-5 space-y-3">
      {isMatch ? (
        <div className="flex items-center gap-2">
          <HugeiconsIcon icon={CheckmarkCircleIcon} className="size-4 text-accent-green" />
          <span className="text-sm text-text-primary font-medium">{match?.name as string || 'Matched'}</span>
          {match?._score != null && <span className="text-xs text-text-muted ml-auto">{Math.round((match._score as number) * 100)}% match</span>}
        </div>
      ) : (
        <div className="flex items-center gap-2">
          <HugeiconsIcon icon={AlertCircleIcon} className="size-4 text-accent-yellow" />
          <span className="text-sm text-text-muted">No matching vendor in registry</span>
        </div>
      )}
      {anomalies.length > 0 && (
        <div className="border-t border-border pt-3">
          <div className="text-xs font-semibold text-text-muted uppercase mb-2">Anomalies ({anomalies.length})</div>
          <div className="space-y-1.5">
            {anomalies.map((a, i) => (
              <div key={i} className={`flex items-start gap-2 px-3 py-1.5 rounded-lg text-xs ${a.severity === 'error' ? 'bg-accent-coral/10 text-accent-coral' : 'bg-accent-yellow/10 text-accent-yellow'}`}>
                <HugeiconsIcon icon={AlertCircleIcon} className="size-3.5 shrink-0 mt-0.5" />
                <span>{a.message as string}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div></SectionCard>
  )
}

function DataCardAnomaly({ page }: { page: Record<string, unknown> | undefined }) {
  const anomalies = (page?.anomalies as Array<Record<string, unknown>>) || []

  return (
    <SectionCard title="Anomaly Detection"><div className="p-5 space-y-3">
      {anomalies.length === 0 ? (
        <div className="flex items-center gap-2 text-sm text-text-muted">
          <HugeiconsIcon icon={CheckmarkCircleIcon} className="size-4 text-accent-green" />
          No anomalies detected
        </div>
      ) : (
        <div className="space-y-2">
          {anomalies.map((a, i) => (
            <div key={i} className={`flex items-start gap-2 px-4 py-2.5 rounded-lg ${a.severity === 'error' ? 'bg-accent-coral/10 border border-accent-coral/20' : 'bg-accent-yellow/10 border border-accent-yellow/20'}`}>
              <HugeiconsIcon icon={AlertCircleIcon} className={`size-4 shrink-0 mt-0.5 ${a.severity === 'error' ? 'text-accent-coral' : 'text-accent-yellow'}`} />
              <div className="flex-1 min-w-0">
                <div className={`text-sm ${a.severity === 'error' ? 'text-accent-coral' : 'text-accent-yellow'}`}>{a.type as string}</div>
                <div className="text-xs text-text-muted mt-0.5">{a.message as string}</div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div></SectionCard>
  )
}

function DataCardMultiTask({ data }: { data: Record<string, unknown> }) {
  const tasks = data.tasks as string[] | undefined
  const results = data.results as Record<string, unknown> | undefined
  if (!results || Object.keys(results).length === 0) {
    return <SectionCard title="Multi-Task NLP"><div className="p-5 text-sm text-text-muted italic">No multi-task results available</div></SectionCard>
  }
  return (
    <SectionCard title="Multi-Task NLP">
      <div className="p-4 space-y-4">
        {tasks && tasks.length > 0 && (
          <div className="flex flex-wrap gap-2">
            {tasks.map(t => (
              <span key={t} className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-xs font-medium bg-accent-violet/10 text-accent-violet border border-accent-violet/20">
                {t.replace(/_/g, ' ')}
              </span>
            ))}
          </div>
        )}
        {!!results.ner && (
          <div>
            <div className="text-xs font-semibold text-text-muted uppercase tracking-wider mb-2">Named Entities</div>
            <div className="grid grid-cols-2 gap-2">
              {Object.entries(results.ner as Record<string, unknown>).map(([k, v]) => (
                <div key={k} className="bg-bg-elevated/50 rounded-lg px-3 py-2">
                  <div className="text-[11px] text-text-muted capitalize">{k}</div>
                  <div className="text-xs font-mono text-text-primary mt-0.5">
                    {Array.isArray(v) ? v.join(', ') : String(v)}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
        {!!results.summarization && (
          <div>
            <div className="text-xs font-semibold text-text-muted uppercase tracking-wider mb-2">Summary</div>
            {!!(results.summarization as Record<string, unknown>).bullets && Array.isArray((results.summarization as Record<string, unknown>).bullets) && (
              <ul className="list-disc list-inside space-y-1 mb-2">
                {((results.summarization as Record<string, unknown>).bullets as string[]).map((b, i) => (
                  <li key={i} className="text-xs text-text-primary">{b}</li>
                ))}
              </ul>
            )}
            {!!(results.summarization as Record<string, unknown>).paragraph && (
              <p className="text-xs text-text-muted leading-relaxed">{(results.summarization as Record<string, unknown>).paragraph as string}</p>
            )}
          </div>
        )}
        {!!results.contract_kie && (
          <div>
            <div className="text-xs font-semibold text-text-muted uppercase tracking-wider mb-2">Contract Clauses</div>
            <div className="space-y-2">
              {Object.entries(results.contract_kie as Record<string, unknown>).map(([k, v]) => (
                <div key={k} className="bg-bg-elevated/50 rounded-lg px-3 py-2">
                  <div className="text-[11px] text-text-muted capitalize">{k.replace(/_/g, ' ')}</div>
                  <div className="text-xs text-text-primary mt-0.5">{(v as Record<string, unknown>)?.text as string || String(v)}</div>
                </div>
              ))}
            </div>
          </div>
        )}
        {!!results.clause_risk && (
          <div>
            <div className="text-xs font-semibold text-text-muted uppercase tracking-wider mb-2">Risk Scores</div>
            <div className="space-y-1">
              {((results.clause_risk as Record<string, unknown>).scores as Array<Record<string, unknown>> || []).map((s, i) => (
                <div key={i} className="flex items-center justify-between bg-bg-elevated/50 rounded-lg px-3 py-2">
                  <span className="text-xs text-text-muted capitalize">{(s.clause_type as string || '').replace(/_/g, ' ')}</span>
                  <span className={`text-xs font-mono font-semibold px-2 py-0.5 rounded-full ${
                    s.risk === 'high_risk' ? 'bg-accent-coral/15 text-accent-coral' :
                    s.risk === 'non_standard' ? 'bg-accent-yellow/15 text-accent-yellow' :
                    'bg-accent-green/15 text-accent-green'
                  }`}>{String(s.risk || '').replace(/_/g, ' ')}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </SectionCard>
  )
}

function DataCardKG({ page }: { page: Record<string, unknown> | undefined }) {
  const graph = page?.graph as Record<string, unknown> | null | undefined
  return (
    <div>{graph ? <KnowledgeGraphView graph={graph as unknown as Parameters<typeof KnowledgeGraphView>[0]['graph']} height={400} /> : (
      <SectionCard title=""><div className="p-5 text-center py-8"><HugeiconsIcon icon={AiNetworkIcon} className="size-10 text-text-muted mx-auto mb-3" /><p className="text-sm text-text-muted">No knowledge graph available</p></div></SectionCard>
    )}</div>
  )
}

function DataCardDocClassifier({ data }: { data: Record<string, unknown> }) {
  const docType = String(data.document_type || 'unknown')
  const confidence = Number(data.confidence || 0)
  const pages = (data.pages as Array<Record<string, unknown>> | undefined) || []
  const typeColors: Record<string, string> = {
    invoice: 'bg-accent-violet/15 text-accent-violet border-accent-violet/30',
    contract: 'bg-blue-500/15 text-blue-400 border-blue-500/30',
    purchase_order: 'bg-accent-coral/15 text-accent-coral border-accent-coral/30',
    delivery_note: 'bg-accent-green/15 text-accent-green border-accent-green/30',
    bank_statement: 'bg-accent-yellow/15 text-accent-yellow border-accent-yellow/30',
    id_card: 'bg-purple-500/15 text-purple-400 border-purple-500/30',
  }
  const colorClass = typeColors[docType] || 'bg-bg-elevated text-text-muted border-border'
  return (
    <div className="space-y-3">
      <SectionCard title="Document Type">
        <div className="p-5 space-y-4">
          <div className="flex items-center gap-3">
            <span className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-sm font-semibold border ${colorClass} capitalize`}>
              {docType === 'unknown' ? 'Unknown' : docType.replace(/_/g, ' ')}
            </span>
            <div className="flex items-center gap-2 text-xs text-text-muted">
              <div className="w-20 h-1.5 bg-border rounded-full overflow-hidden">
                <div className={`h-full rounded-full ${confidence >= 0.7 ? 'bg-accent-green' : confidence >= 0.4 ? 'bg-accent-yellow' : 'bg-accent-coral'}`}
                  style={{width: `${confidence * 100}%`}} />
              </div>
              <span className="font-mono">{(confidence * 100).toFixed(0)}%</span>
            </div>
          </div>
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div><div className="text-xs font-medium text-text-muted uppercase tracking-wider mb-0.5">Method</div><div className="text-sm text-text-primary">Keyword scoring + majority vote</div></div>
            <div><div className="text-xs font-medium text-text-muted uppercase tracking-wider mb-0.5">Routing applied</div><div className="text-sm text-text-primary">{docType !== 'unknown' ? 'Yes — target fields updated' : 'No — using defaults'}</div></div>
          </div>
        </div>
      </SectionCard>
      <SectionCard title="Per-Page Classification">
        {pages.length === 0 ? <div className="p-5 text-sm text-text-muted italic">No pages classified</div> : (
          <div className="divide-y divide-border">{pages.map(p => (
            <div key={p.page_number as number} className="flex items-center justify-between px-5 py-3 hover:bg-bg-elevated/30 transition-colors">
              <span className="text-sm text-text-primary font-medium">Page {p.page_number as number}</span>
              <div className="flex items-center gap-3">
                <span className="inline-flex items-center gap-1 px-2 py-0.5 text-xs font-medium rounded-full bg-bg-elevated text-text-primary capitalize">{(p.page_type as string) || 'Unknown'}</span>
                <span className="text-xs text-text-muted font-mono">{((p.confidence as number) * 100).toFixed(0)}%</span>
              </div>
            </div>
          ))}</div>
        )}
        <div className="px-5 py-3 border-t border-border bg-bg-elevated/50 rounded-b-xl text-xs text-text-muted">{pages.length} page{pages.length !== 1 ? 's' : ''} classified</div>
      </SectionCard>
    </div>
  )
}

function DataCardEmbedding({ data }: { data: Record<string, unknown> }) {
  const pages = (data.pages as Array<Record<string, unknown>> | undefined) || []
  return (
    <SectionCard title="Page Embeddings">
      <div className="p-5">
        <div className="text-xs text-text-muted mb-3">Model: <span className="font-mono text-text-primary font-medium">{String(data.model || 'Unknown')}</span></div>
        {pages.map(p => (
          <div key={p.page_number as number} className="flex items-center justify-between py-2 text-sm border-b border-border last:border-0">
            <span className="text-text-muted font-medium">Page {p.page_number as number}</span>
            <span className="text-xs text-text-muted font-mono tabular-nums">dim: {p.embedding_dim as number}</span>
          </div>
        ))}
      </div>
    </SectionCard>
  )
}

function DataCardCrossPage({ page }: { page: Record<string, unknown> | undefined }) {
  const entities = (page?.linked_entities as Array<Record<string, unknown>>) || []
  return (
    <SectionCard title="Cross-Page Resolution">
      {entities.length === 0 ? <div className="p-5 text-sm text-text-muted italic">No cross-page links</div> : (
        <div className="divide-y divide-border">{entities.map((e, i) => (
          <div key={i} className="px-5 py-3 hover:bg-bg-elevated/30 transition-colors">
            <div className="flex items-center gap-2 text-sm text-text-primary">
              <span className="font-medium">{String(e.supplier || e.entity || '')}</span>
              {!!e.address && <span className="text-text-muted">· {String(e.address)}</span>}
              {!!e.page && <span className="ml-auto text-xs text-text-muted font-mono">page {String(e.page)}</span>}
            </div>
          </div>
        ))}</div>
      )}
      {entities.length > 0 && <div className="px-5 py-3 border-t border-border bg-bg-elevated/50 rounded-b-xl text-xs text-text-muted">{entities.length} link{entities.length !== 1 ? 's' : ''}</div>}
    </SectionCard>
  )
}

function DataCardEval({ metrics }: { metrics: Record<string, unknown> | undefined }) {
  if (!metrics || Object.keys(metrics).length === 0) return <div className="bg-bg-surface border border-border rounded-xl p-5 text-sm text-text-muted italic">No evaluation data</div>
  const accuracy = metrics.accuracy as Record<string, unknown> | undefined
  const faithfulness = metrics.faithfulness as Record<string, unknown> | undefined
  const numericDelta = metrics.numeric_delta as Record<string, unknown> | undefined
  const formatCompliance = metrics.format_compliance as Record<string, unknown> | undefined
  const detectionRate = metrics.detection_rate as Record<string, unknown> | undefined
  const perField = accuracy?.per_field as Record<string, { count: number; exact_match: number; avg_token_f1: number; entries?: Array<Record<string, unknown>> }> | undefined

  const accScore = typeof accuracy?.score === 'number' ? accuracy.score : null
  const accExact = typeof accuracy?.exact_match === 'number' ? accuracy.exact_match : 0
  const accTotal = typeof accuracy?.total_fields === 'number' ? accuracy.total_fields : 0
  const accTokenF1 = typeof accuracy?.partial_token_f1 === 'number' ? accuracy.partial_token_f1 : null
  const faithScore = typeof faithfulness?.score === 'number' ? faithfulness.score : null
  const faithFul = typeof faithfulness?.faithful === 'number' ? faithfulness.faithful : 0
  const faithTotal = typeof faithfulness?.total === 'number' ? faithfulness.total : 0
  const ndScore = typeof numericDelta?.score === 'number' ? numericDelta.score : null
  const fmtScore = typeof formatCompliance?.score === 'number' ? formatCompliance.score : null
  const detScore = typeof detectionRate?.score === 'number' ? detectionRate.score : null
  const faithPerField: Record<string, number> = faithfulness?.per_field && typeof faithfulness.per_field === 'object' ? faithfulness.per_field as Record<string, number> : {}

  const gridItems: Array<{ key: string; label: string; score: number | null; subtitle: string; desc: string }> = []
  if (accScore != null || accTotal > 0)
    gridItems.push({ key: 'accuracy', label: METRIC_INFO.accuracy.label, score: accScore, subtitle: `${accExact}/${accTotal}`, desc: METRIC_INFO.accuracy.desc })
  if (faithScore != null || faithTotal > 0)
    gridItems.push({ key: 'faithfulness', label: METRIC_INFO.faithfulness.label, score: faithScore, subtitle: `${faithFul}/${faithTotal}`, desc: METRIC_INFO.faithfulness.desc })
  if (accTokenF1 != null)
    gridItems.push({ key: 'token_f1', label: METRIC_INFO.token_f1.label, score: accTokenF1, subtitle: 'partial', desc: METRIC_INFO.token_f1.desc })
  if (ndScore != null)
    gridItems.push({ key: 'numeric_delta', label: METRIC_INFO.numeric_delta.label, score: ndScore, subtitle: `${numericDelta?.count ?? 0} fields`, desc: METRIC_INFO.numeric_delta.desc })
  if (fmtScore != null)
    gridItems.push({ key: 'format_compliance', label: METRIC_INFO.format_compliance.label, score: fmtScore, subtitle: `${formatCompliance?.passed ?? 0}/${formatCompliance?.total ?? 0}`, desc: METRIC_INFO.format_compliance.desc })
  if (detScore != null)
    gridItems.push({ key: 'detection_rate', label: METRIC_INFO.detection_rate.label, score: detScore, subtitle: `${detectionRate?.detected ?? 0}/${detectionRate?.total ?? 0}`, desc: METRIC_INFO.detection_rate.desc })

  return (
    <div className="space-y-4">
      <div className={`grid gap-3 ${gridItems.length <= 3 ? 'grid-cols-3' : 'grid-cols-3'}`}>
        {gridItems.map(({ key, label, score, subtitle, desc }) => (
          <ScoreGauge key={key} label={label} score={score} subtitle={subtitle} info={desc} />
        ))}
      </div>
      {perField && Object.keys(perField).length > 0 && (
        <SectionCard title="Per-Field Accuracy"><div className="p-3 space-y-2">
          {Object.entries(perField).sort(([, a], [, b]) => a.exact_match - b.exact_match).map(([name, metric]) => (
            <FieldMetricRow key={name} name={name} metric={metric} />
          ))}
        </div></SectionCard>
      )}
      {Object.keys(faithPerField).length > 0 && (
        <SectionCard title="Faithfulness (value found in OCR)"><div className="p-3 grid grid-cols-2 gap-2">
          {Object.entries(faithPerField).map(([field, score]) => (
            <div key={field} className="flex items-center justify-between px-3 py-1.5 bg-bg-elevated/50 rounded-lg text-xs">
              <span className="text-text-muted">{field}</span>
              <span className={`font-mono font-semibold tabular-nums ${score >= 0.8 ? 'text-accent-violet' : score >= 0.5 ? 'text-accent-yellow' : 'text-accent-coral'}`}>{Math.round(score * 100)}%</span>
            </div>
          ))}
        </div></SectionCard>
      )}
    </div>
  )
}

/* ── Pipeline Step Data View ── */

export function PipelineStepDataView({ stepName, data, onFieldSelect, sessionId }: { stepName: string; data: Record<string, unknown>; onFieldSelect?: (field: string) => void; sessionId?: string }) {
  const pages = (data.pages as Array<Record<string, unknown>> | undefined) || []
  const page = pages[0]

  switch (stepName) {
    case 'ingestion': return <SectionCard title="Document Information"><IngestionDataContent data={data} /></SectionCard>
    case 'ocr': return <DataCardOCR page={page} />
    case 'vision_ocr': return <DataCardVisionOCR page={page} />
    case 'hybrid_ocr': return <DataCardHybridOCR page={page} />
    case 'document_graph': return <DataCardDocGraph page={page} />
    case 'end_to_end_vlm': return <DataCardE2EVLM page={page} />
    case 'document_classifier': return <DataCardDocClassifier data={data} />
    case 'embedding': return <DataCardEmbedding data={data} />
    case 'retrieval': return <DataCardRetrieval page={page} />
    case 'rag': return <DataCardRAG page={page} />
    case 'llm_extraction': return <DataCardLLM page={page} onFieldSelect={onFieldSelect} />
    case 'cross_page': return <DataCardCrossPage page={page} />
    case 'validation': return <DataCardValidation page={page} />
    case 'confidence_scoring': return <DataCardConfidence page={page} />
    case 'export': return <DataCardExport data={data} sessionId={sessionId || ''} />
    case 'vendor_lookup': return <DataCardVendor page={page} />
    case 'anomaly': return <DataCardAnomaly page={page} />
    case 'multi_task': return <DataCardMultiTask data={data} />
    case 'knowledge_graph': return <DataCardKG page={page} />
    case 'evaluation': return <DataCardEval metrics={data.metrics as Record<string, unknown> | undefined} />
    default: return <div className="text-sm text-text-muted italic p-4">No details available</div>
  }
}

/* ── Helpers ── */

function resultToReviewFields(r: PipelineResult): ReviewField[] {
  const fields: ReviewField[] = []
  const labelKeys = new Map<string, string>()
  const pageWidths = new Map<number, number>()
  const pageHeights = new Map<number, number>()
  for (const page of r.pages) {
    pageWidths.set(page.page_number, page.image_width || 1)
    pageHeights.set(page.page_number, page.image_height || 1)
    if (!page.predicted_annotations) continue
    for (const ann of page.predicted_annotations) {
      const existing = labelKeys.get(ann.label)
      if (!existing || ann.source === 'predicted') {
        labelKeys.set(ann.label, ann.text)
      }
    }
  }
  for (const page of r.pages) {
    for (const [key, val] of Object.entries(page.extracted_fields)) {
      const jsonVal = val as string | number | boolean | null | Record<string, unknown> | unknown[]
      let schema: ReviewFieldSchema = { type: 'string' }
      if (typeof jsonVal === 'number') schema = { type: 'number' }
      else if (typeof jsonVal === 'boolean') schema = { type: 'boolean' }
      else if (jsonVal === null) schema = { type: 'string' }
      const ann = page.predicted_annotations?.find(a => a.label === key)
      let location: ReviewLocation | undefined
      if (ann) {
        const pw = pageWidths.get(page.page_number) || 1
        const ph = pageHeights.get(page.page_number) || 1
        location = {
          page: page.page_number,
          area: {
            left: (ann.box[0] / pw) * 100,
            top: (ann.box[1] / ph) * 100,
            width: ((ann.box[2] - ann.box[0]) / pw) * 100,
            height: ((ann.box[3] - ann.box[1]) / ph) * 100,
          },
        }
      } else if (r.num_pages > 1) {
        location = { page: page.page_number, area: { left: 0, top: 0, width: 0, height: 0 } }
      }
      fields.push({
        key,
        schema,
        actual: jsonVal as string | number | boolean | null,
        expected: null,
        location,
      })
    }
  }
  return fields
}

function ocrDataToParsedOcrOutput(data: Record<string, unknown>): ParsedOcrOutput {
  const pages = (data.pages as Array<Record<string, unknown>>) || []
  const chunks = pages.map((page) => {
    const markdown = (page.markdown as string) || ''
    const text = (page.text as string) || ''
    const content = markdown || text
    const imageWidth = (page.image_width as number) || 1000
    const imageHeight = (page.image_height as number) || 1000
    const pageNumber = (page.page_number as number) || 1
    const boxes = (page.boxes as Array<Record<string, unknown>>) || []
    const avgConfidence = boxes.length > 0
      ? boxes.reduce((sum: number, b: Record<string, unknown>) => sum + (b.confidence as number || 0), 0) / boxes.length
      : 0.9

    const paragraphs = content.split(/\n\n+/).filter(Boolean)
    const blocks: ParsedOcrBlock[] = paragraphs.map((para, i) => {
      let type = 'text'
      if (para.startsWith('# ')) type = 'heading'
      else if (para.startsWith('|')) type = 'table'
      else if (para.startsWith('- ') || /^\d+\.\s/.test(para)) type = 'list'

      return {
        id: `ocr-p${pageNumber}-${i}`,
        type,
        content: para,
        metadata: {
          page: { number: pageNumber, width: imageWidth, height: imageHeight },
          avgOcrConfidence: avgConfidence,
        },
      }
    })

    return { blocks }
  })

  return { chunks }
}

function fmtExtractedVal(val: unknown): string {
  if (val == null) return ''
  if (typeof val === 'string' || typeof val === 'number' || typeof val === 'boolean') return String(val)
  if (Array.isArray(val)) {
    const primitives = val.filter(v => typeof v !== 'object' || v === null)
    if (primitives.length === val.length) return primitives.map(String).join(', ')
    return JSON.stringify(val)
  }
  return JSON.stringify(val)
}

/* ── Pipeline View ── */

export function PipelineView({
  sessionId, onDone, steps, setSteps, setError,
  selectedStep, onSelectStep, waiting, setWaiting, setNextStepName,
  error: _error, resultReady,
}: {
  sessionId: string
  onDone: (r: PipelineResult) => void
  steps: Record<string, StepState>
  setSteps: React.Dispatch<React.SetStateAction<Record<string, StepState>>>
  error: string | null
  setError: (e: string | null) => void
  selectedStep: string | null
  onSelectStep: (s: string | null) => void
  waiting: boolean
  setWaiting: React.Dispatch<React.SetStateAction<boolean>>
  setNextStepName: React.Dispatch<React.SetStateAction<string | null>>
  resultReady?: boolean
}) {
  void _error
  const fetchedResult = useRef(false)
  const onDoneRef = useRef(onDone)
  const onSelectStepRef = useRef(onSelectStep)
  const discardOnRerunRef = useRef<Record<string, string[]>>({})
  onDoneRef.current = onDone
  onSelectStepRef.current = onSelectStep
  const [resultsData, setResultsData] = useState<PipelineResult | null>(null)
  const [runningStep, setRunningStep] = useState<string | null>(null)

  // Build partial PipelineResult from available step data when core steps complete
  const partialBuilt = useRef(false)
  const prevSessionRef = useRef<string | null>(null)
  useEffect(() => { if (prevSessionRef.current !== sessionId) { partialBuilt.current = false; prevSessionRef.current = sessionId } })
  useEffect(() => {
    if (resultsData || resultReady || partialBuilt.current) return
    const vlmStep = steps['end_to_end_vlm']
    if (vlmStep?.status !== 'completed' || !vlmStep?.data) return
    const vlmData = vlmStep.data as Record<string, unknown>
    const vlmPages = (vlmData.pages as Array<Record<string, unknown>> | undefined) || []
    if (vlmPages.length === 0) return
    const resultPages = vlmPages.map(p => ({
      page_number: p.page_number as number || 1,
      extracted_fields: (p.fields || {}) as Record<string, string>,
      page_type: null,
      page_type_confidence: 0,
      ocr_word_count: 0,
      ocr_markdown: '',
      line_items: [],
      validation: null,
      knowledge_graph: null,
      ocr_text: '',
      ocr_boxes: [],
      image_width: 0,
      image_height: 0,
      image_path: '',
      retrieved_examples: [],
      rag_rules: [],
      rag_templates: [],
      last_prompt: '',
      ground_truth_annotations: [],
      predicted_annotations: [],
      extraction_evidence: {},
      page_fragments: [],
      linked_entities: [],
      session_id: sessionId,
    })) as unknown as PageResult[]
    const partial: PipelineResult = {
      session_id: sessionId,
      input_path: '',
      document_type: null,
      classified_type: 'unknown',
      classified_confidence: 0,
      pages: resultPages,
      num_pages: resultPages.length,
      timing: {},
      total_time: vlmStep.elapsed || 0,
      evaluation: null,
      errors: [],
    }
    partialBuilt.current = true
    setResultsData(partial)
    onDoneRef.current(partial)
  }, [steps, resultsData, resultReady, sessionId])
  const viewerRef = useRef<PDFViewerHandle>(null)
  const reviewFieldsRef = useRef<ReviewField[]>([])
  const [activeFieldKey, setActiveFieldKey] = useState<string | undefined>(undefined)
  const [hoverLocation, setHoverLocation] = useState<ReviewLocation | null>(null)

  const focusField = useCallback((field: ReviewField) => {
    setActiveFieldKey(field.key)
    if (field.location) {
      viewerRef.current?.scrollToPageArea(field.location.page, field.location.area)
    }
  }, [])

  const handleLocationHover = useCallback((location?: ReviewLocation) => {
    setHoverLocation(location ?? null)
    if (location) {
      viewerRef.current?.scrollToPageArea(location.page, location.area, { behavior: "auto" })
    }
  }, [])

  const handleQaFieldClick = useCallback((fieldKey: string) => {
    const field = reviewFieldsRef.current.find(f => f.key === fieldKey)
    if (field) focusField(field)
  }, [focusField])

  const [ocrPostCorrect] = useState(true)
  const [visionOcrPostCorrect] = useState(true)
  const [embeddingTextSource] = useState('auto')
  const [retrievalStrategy] = useState('hybrid')
  const [currentMode, setCurrentMode] = useState('end_to_end')
  const [selectedFields, setSelectedFields] = useState<string[]>(DEFAULT_FIELDS)
  const [infoTarget, setInfoTarget] = useState<{step: string; top: number; left: number; width: number} | null>(null)
  const [embeddingModels, setEmbeddingModels] = useState<Array<{id: string; name: string; provider: string}>>([])
  const [currentEmbeddingModel, setCurrentEmbeddingModel] = useState('e5')
  const [currentVlmModel, setCurrentVlmModel] = useState('gemma3:4b')

  useEffect(() => {
    fetch('/api/embedding/models').then(r => r.json()).then(d => { if (d.models?.length) setEmbeddingModels(d.models) }).catch(() => {})
  }, [])

  const enabledSteps = getEnabledSteps(currentMode)
  const enabledSet = new Set(enabledSteps)
  const completedCount = Object.entries(steps).filter(([k, s]) => enabledSet.has(k) && s.status === 'completed').length
  const anyPending = Object.values(steps).some(s => s.status === 'pending' || s.status === 'failed')
  const allStepsDone = Object.keys(steps).length > 0 && Object.keys(steps).every(s => steps[s]?.status === 'completed')
  const anyFailed = Object.values(steps).some(s => s.status === 'failed')
  const anyRunning = Object.values(steps).some(s => s.status === 'running')
  const pipelineFinished = allStepsDone || (anyFailed && !anyRunning)
  const totalSteps = enabledSteps.length
  const progressPct = Math.round((completedCount / totalSteps) * 100)
  const preprocSteps = getPreprocSteps(currentMode)

  // auto-fetch results when all pipeline steps are done
  useEffect(() => {
    if (!pipelineFinished || resultsData || resultReady) return
    let cancelled = false
    ;(async () => {
      for (let i = 0; i < 120 && !cancelled; i++) {
        try {
          const res = await fetch(`/api/result/${sessionId}`)
          if (res.ok) {
            const d = await res.json() as PipelineResult
            if (!cancelled) { setResultsData(d); onDoneRef.current(d) }
            return
          }
        } catch { /* retry */ }
        await new Promise(r => setTimeout(r, 2000))
      }
    })()
    return () => { cancelled = true }
  }, [pipelineFinished, resultsData, resultReady, sessionId])

  async function handleModeChange(mode: string) {
    setCurrentMode(mode)
    if (!sessionId) return
    const form = new FormData()
    form.append('mode', mode)
    form.append('target_fields', selectedFields.join(','))
    form.append('vlm_model', currentVlmModel)
    const res = await fetch(`/api/session/${sessionId}/config`, { method: 'POST', body: form })
    if (!res.ok) console.error('Mode change failed', await res.text())
  }

  async function handleRunMode(mode: string) {
    if (mode !== currentMode) await handleModeChange(mode)
    await runAll(mode)
  }

  async function handleFieldToggle(f: string) {
    const next = selectedFields.includes(f)
      ? selectedFields.filter(x => x !== f)
      : [...selectedFields, f]
    setSelectedFields(next)
    if (!sessionId) return
    const form = new FormData()
    form.append('mode', currentMode)
    form.append('target_fields', next.join(','))
    const res = await fetch(`/api/session/${sessionId}/config`, { method: 'POST', body: form })
    if (!res.ok) console.error('Config update failed', await res.text())
  }

  async function sendFields(fields: string[]) {
    setSelectedFields(fields)
    if (!sessionId) return
    const form = new FormData()
    form.append('mode', currentMode)
    form.append('target_fields', fields.join(','))
    form.append('vlm_model', currentVlmModel)
    const res = await fetch(`/api/session/${sessionId}/config`, { method: 'POST', body: form })
    if (!res.ok) console.error('Fields update failed', await res.text())
  }

  const sortedSteps = Object.entries(steps).sort(([, a], [, b]) => a.stepIndex - b.stepIndex)
  const lastCompleted = [...sortedSteps].reverse().find(([, s]) => s.status === 'completed')

  useEffect(() => {
    if (lastCompleted && !selectedStep) {
      onSelectStep(lastCompleted[0])
    }
  }, [lastCompleted?.[0], selectedStep, onSelectStep])

  useEffect(() => {
    if (!sessionId) return
    fetch('/api/pipeline/prereqs').then(r => r.json()).then(data => {
      if (data.discard_on_rerun) discardOnRerunRef.current = data.discard_on_rerun
    }).catch(() => {})
  }, [sessionId])

  useEffect(() => {
    let ws: WebSocket | null = null
    let reconnectTimer: ReturnType<typeof setTimeout> | null = null
    let pingInterval: ReturnType<typeof setInterval> | null = null
    let pollInterval: ReturnType<typeof setInterval> | null = null
    let reconnectAttempts = 0
    const MAX_RECONNECT = 5
    let cancelled = false

    function cleanup() {
      cancelled = true
      if (ws) { ws.close(); ws = null }
      if (reconnectTimer) { clearTimeout(reconnectTimer); reconnectTimer = null }
      if (pingInterval) { clearInterval(pingInterval); pingInterval = null }
      if (pollInterval) { clearInterval(pollInterval); pollInterval = null }
    }

    async function pollForResult() {
      if (resultReady) return
      for (let i = 0; i < 120 && !cancelled; i++) {
        await new Promise(r => setTimeout(r, 2000))
        try {
          const res = await fetch(`/api/result/${sessionId}`)
          if (res.ok) {
            const data: PipelineResult = await res.json()
            if (!cancelled && !resultReady) { setResultsData(data); onDone(data) }
            return
          }
          if (res.status === 404) { setError('Session expired — server restarted.'); cleanup(); return }
        } catch { /* continue */ }
      }
    }

    function connect() {
      if (cancelled) return
      if (reconnectAttempts >= MAX_RECONNECT) { setError('Cannot connect to server.'); cleanup(); return }
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
      ws = new WebSocket(`${protocol}//${window.location.host}/ws/${sessionId}`)
      ws.onopen = () => { reconnectAttempts = 0; pingInterval = setInterval(() => { if (ws?.readyState === WebSocket.OPEN) ws.send('ping') }, 25000) }
      ws.onmessage = (event) => {
        const msg = JSON.parse(event.data)
        if (msg.type === 'ping') return
        if (msg.type === 'progress') {
          const idx = STEP_ORDER.indexOf(msg.step)
          setSteps(prev => ({ ...prev, [msg.step]: { status: msg.status, elapsed: msg.elapsed, data: msg.data || {}, stepIndex: idx >= 0 ? idx : 999 } }))
          if (msg.status === 'running') setRunningStep(msg.step)
          if (msg.status === 'completed' || msg.status === 'failed') { setRunningStep(null); onSelectStep(msg.step) }
        } else if (msg.type === 'waiting') { setWaiting(true); setRunningStep(null)
        } else if (msg.type === 'completed' && !fetchedResult.current) { fetchedResult.current = true; setWaiting(false); ws?.close(); const r = msg.result as PipelineResult | undefined; if (r) { setResultsData(r); onDone(r) } else { pollForResult() }
        } else if (msg.type === 'stopped') { setWaiting(true); setRunningStep(null)
        } else if (msg.type === 'steps_discarded') { setSteps(prev => { const next = { ...prev }; for (const s of (msg.discarded as string[])) { const was = next[s] || { stepIndex: STEP_ORDER.indexOf(s) }; next[s] = { ...was, status: 'pending' as const, elapsed: 0 }; delete next[s].data }; return next }); setRunningStep(null); setWaiting(true)
        } else if (msg.type === 'error') { setError(msg.error); setWaiting(false); setRunningStep(null); ws?.close() }
      }
      ws.onclose = () => { if (pingInterval) clearInterval(pingInterval); ws = null; if (cancelled || fetchedResult.current) return; reconnectAttempts++; reconnectTimer = setTimeout(connect, 3000) }
    }
    connect()
    pollInterval = setInterval(async () => {
      try {
        const res = await fetch(`/api/status/${sessionId}`)
        if (res.status === 404) { setError('Session expired.'); cleanup(); return }
        if (res.ok) {
          const data = await res.json()
          if (data.status === 'completed' && !fetchedResult.current && !resultReady) { fetchedResult.current = true; pollForResult(); return }
          if (data.status === 'failed') { setError(data.error || 'Pipeline failed'); cleanup(); return }
          if (data.status === 'stopped') { setWaiting(true); setRunningStep(null) }
          if (data.progress) { setSteps(prev => { const next = { ...prev }; for (const [step, st] of Object.entries(data.progress)) { const stp = st as { status: StepState['status']; elapsed: number; data?: Record<string, unknown> }; const idx = STEP_ORDER.indexOf(step); const base = { stepIndex: idx >= 0 ? idx : 999, status: stp.status, elapsed: stp.elapsed }; if (!next[step]) { next[step] = { ...base, data: stp.data } } else { next[step] = { ...next[step], ...base, data: stp.data || next[step].data } } } return next }) }
          if (data.waiting_for_input) { setWaiting(true) }
        }
      } catch { /* ignore */ }
    }, 3000)
    return cleanup
  }, [sessionId, onDone, setSteps, setError, setWaiting, setNextStepName, onSelectStep])

  async function runStep(step: string, stepConfig?: Record<string, unknown>) {
    setWaiting(false); setRunningStep(step)
    const currentStatus = steps[step]?.status
    const isRerun = currentStatus === 'completed' || currentStatus === 'failed'
    if (isRerun) {
      const downstream = discardOnRerunRef.current[step] || getDownstream(step)
      setSteps(prev => {
        const next = { ...prev }
        for (const s of [step, ...downstream]) {
          const was = next[s] || { stepIndex: STEP_ORDER.indexOf(s) }
          next[s] = { ...was, status: 'pending' as const, elapsed: 0 }
          delete next[s].data
        }
        return next
      })
    }
    const url = isRerun ? `/api/pipeline/rerun/${sessionId}/${step}` : `/api/pipeline/continue/${sessionId}`
    const body: Record<string, unknown> = isRerun ? {} : { step }
    if (stepConfig && Object.keys(stepConfig).length > 0) {
      body.config = stepConfig
    }
    const opts: RequestInit = { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }
    try {
      const res = await fetch(url, opts)
      if (!res.ok) {
        const detail = await res.json().then(d => d.detail).catch(() => null)
        const msg = detail ? `${detail}` : `HTTP ${res.status}`
        setError(msg || 'Failed to run step')
        setRunningStep(null)
        return
      }
    } catch {
      setError('Network error — check server')
      setRunningStep(null)
    }
  }

  async function runAll(modeOverride?: string) {
    setWaiting(false); setRunningStep('__all__')
    try {
      const form = new FormData()
      form.append('mode', modeOverride || currentMode)
      const res = await fetch(`/api/pipeline/run-all/${sessionId}`, { method: 'POST', body: form })
      if (!res.ok) { setError(`Run all failed: HTTP ${res.status}`); setRunningStep(null) }
    } catch { setError('Network error — check server'); setRunningStep(null) }
  }

  async function stopPipeline() {
    setRunningStep(null)
    await fetch(`/api/pipeline/stop/${sessionId}`, { method: 'POST' })
  }

  const [sidebarOpen, setSidebarOpen] = useState(true)

  const [qaOpen, setQaOpen] = useState(false)
  const [qaMessages, setQaMessages] = useState<Array<{role: string; content: string; evidence?: Record<string, string>}>>([])
  const [qaInput, setQaInput] = useState('')
  const [qaLoading, setQaLoading] = useState(false)
  const [qaSystemPrompt, setQaSystemPrompt] = useState(DEFAULT_QA_PROMPT)
  const [qaShowPrompt, setQaShowPrompt] = useState(true)
  const qaEndRef = useRef<HTMLDivElement>(null)
  const qaMessagesRef = useRef(qaMessages)
  qaMessagesRef.current = qaMessages
  const [correctionEdits, setCorrectionEdits] = useState<Record<string, string>>({})
  const [correctionStatus, setCorrectionStatus] = useState<'idle' | 'saving' | 'saved' | 'error'>('idle')
  const [confirmedFields, setConfirmedFields] = useState<Set<string>>(new Set())
  const [correctionMsg, setCorrectionMsg] = useState('')

  const handleCorrectionSave = async () => {
    setCorrectionStatus('saving')
    setCorrectionMsg('')
    try {
      const res = await fetch(`/api/correct/${sessionId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ corrections: correctionEdits }),
      })
      if (!res.ok) throw new Error(await res.text())
      setCorrectionStatus('saved')
      setCorrectionMsg('Corrections saved successfully!')
      setCorrectionEdits({})
    } catch (e) {
      setCorrectionStatus('error')
      setCorrectionMsg(`Failed: ${e instanceof Error ? e.message : String(e)}`)
    }
  }

  const sendQuestion = useCallback(async () => {
    const q = qaInput.trim()
    if (!q || qaLoading) return
    setQaInput('')
    const prevMessages = qaMessagesRef.current
    setQaMessages(prev => [...prev, { role: 'user', content: q }])
    setQaLoading(true)
    try {
      const body: Record<string, unknown> = {
        question: q,
        messages: prevMessages.map(m => ({ role: m.role, content: m.content })),
      }
      if (qaSystemPrompt.trim()) body.system_prompt = qaSystemPrompt.trim()
      const res = await fetch(`/api/qa/${sessionId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })
      if (!res.ok) {
        const errData = await res.json().catch(() => ({}))
        throw new Error(errData.detail || `HTTP ${res.status}`)
      }
      const data = await res.json()
      const answer = typeof data.answer === 'string' ? data.answer : JSON.stringify(data.answer)
      setQaMessages(prev => [...prev, { role: 'assistant', content: answer, evidence: data.evidence }])
    } catch (e) {
      const msg = e instanceof Error ? e.message : 'Unknown error'
      setQaMessages(prev => [...prev, { role: 'assistant', content: `Sorry — ${msg}` }])
    }
    setQaLoading(false)
  }, [qaInput, qaLoading, qaSystemPrompt, sessionId])

  useEffect(() => {
    qaEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [qaMessages])

  const qaButton = sessionId && (
    <Button variant="default" size="icon-lg"
      onClick={() => setQaOpen(!qaOpen)}
      className={`fixed bottom-4 right-4 z-40 rounded-full shadow-lg ${
        qaOpen ? '!bg-bg-elevated !text-text-muted !shadow-none' : '!bg-accent-violet !text-white hover:!bg-accent-violet/80 !shadow-accent-violet/30'
      }`}>
      {qaOpen ? <HugeiconsIcon icon={Cancel01Icon} className="size-5" /> : <HugeiconsIcon icon={BubbleChatIcon} className="size-5" />}
    </Button>
  )

  const qaPanel = sessionId && qaOpen && steps['evaluation']?.status === 'completed' && (
        <div className="fixed bottom-4 right-4 w-[36rem] h-[42rem] bg-bg-surface border border-border rounded-2xl shadow-2xl flex flex-col overflow-hidden z-50 animate-scale-in">
          <div className="flex items-center justify-between px-4 py-3 border-b border-border shrink-0">
            <span className="text-sm font-semibold text-text-primary">Ask about this document</span>
            <div className="flex items-center gap-1">
              <Button variant="ghost" size="sm" onClick={() => setQaShowPrompt(!qaShowPrompt)}>
                {qaShowPrompt ? 'Hide' : 'Prompt'}
              </Button>
              <Button variant="ghost" size="icon-sm" onClick={() => setQaOpen(false)}>
                <HugeiconsIcon icon={Cancel01Icon} className="size-4" />
              </Button>
            </div>
          </div>
          {qaShowPrompt && (
            <div className="px-3 py-2 border-b border-border shrink-0">
              <div className="text-xs text-text-muted mb-1">System prompt (sent with each question):</div>
              <textarea
                value={qaSystemPrompt}
                onChange={e => setQaSystemPrompt(e.target.value)}
                rows={4}
                placeholder="Optional custom system prompt..."
                className="w-full bg-bg-base text-xs text-text-primary placeholder-text-muted px-2 py-1.5 rounded-lg border border-border focus:border-accent-violet focus:ring-1 focus:ring-accent-violet/30 outline-none resize-none transition-all"
              />
            </div>
          )}
          <div className="flex-1 overflow-y-auto p-3 space-y-3">
            {qaMessages.length === 0 && (
              <div className="text-xs text-text-muted text-center mt-12">
                Ask questions like:<br/>
                "What is the total amount?"<br/>
                "Who is the supplier?"<br/>
                "What is the invoice date?"
              </div>
            )}
            {qaMessages.map((msg, i) => (
              <div key={i} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                <div className={`max-w-[85%] px-3 py-2 rounded-xl text-sm ${
                  msg.role === 'user'
                    ? 'bg-accent-violet/20 text-accent-violet border border-accent-violet/30'
                    : 'bg-bg-elevated text-text-muted border border-border/50'
                }`}>
                  <QAMessage content={msg.content} evidence={msg.evidence} onFieldClick={handleQaFieldClick} />
                </div>
              </div>
            ))}
            {qaLoading && (
              <div className="flex justify-start">
                <div className="bg-bg-elevated text-text-muted px-3 py-2 rounded-xl text-sm border border-border/50 flex items-center gap-2">
                  <HugeiconsIcon icon={Loading01Icon} className="size-3 animate-spin" /> Thinking...
                </div>
              </div>
            )}
            <div ref={qaEndRef} />
          </div>
          <div className="px-3 py-2 border-t border-border shrink-0">
            <form onSubmit={e => { e.preventDefault(); sendQuestion() }}
              className="flex items-center gap-2">
              <input
                type="text"
                value={qaInput}
                onChange={e => setQaInput(e.target.value)}
                placeholder="Ask a question..."
                disabled={qaLoading}
                className="flex-1 bg-bg-base text-sm text-text-primary placeholder-text-muted px-3 py-2 rounded-xl border border-border focus:border-accent-violet focus:ring-1 focus:ring-accent-violet/30 outline-none transition-all disabled:opacity-50"
              />
              <Button variant="default" size="icon-sm" type="submit" disabled={qaLoading || !qaInput.trim()}>
                <HugeiconsIcon icon={MailSend01Icon} className="size-4" />
              </Button>
            </form>
          </div>
        </div>
    )

  const reviewFields = useMemo(() => resultsData ? resultToReviewFields(resultsData) : [], [resultsData])
  reviewFieldsRef.current = reviewFields

  const ocrParsedOutput = useMemo(() => {
    if (selectedStep !== 'ocr' || !steps.ocr?.data) return null
    return ocrDataToParsedOcrOutput(steps.ocr.data as Record<string, unknown>)
  }, [selectedStep, steps.ocr?.data])

  const ocrBlocks = useMemo(() => ocrParsedOutput ? getOcrBlocks(ocrParsedOutput) : [], [ocrParsedOutput])

  const [activeOcrBlockId, setActiveOcrBlockId] = useState<string | undefined>(undefined)

  const focusOcrBlock = useCallback((block: OcrBlock) => {
    setActiveOcrBlockId(block.id)
    const area = blockToArea(block)
    viewerRef.current?.scrollToPageArea(block.page, {
      left: Number.parseFloat(String(area.left ?? 0)),
      top: Number.parseFloat(String(area.top ?? 0)),
      width: Number.parseFloat(String(area.width ?? 0)),
      height: Number.parseFloat(String(area.height ?? 0)),
    }, { behavior: "auto" })
  }, [])

  return (
    <div className="h-screen flex relative">
      <PipelineSidebar
        collapsed={!sidebarOpen} onToggle={() => setSidebarOpen(!sidebarOpen)}
        currentMode={currentMode} onModeChange={handleModeChange} onRunMode={handleRunMode}
        selectedFields={selectedFields} onFieldToggle={handleFieldToggle} sendFields={sendFields}
        embeddingModels={embeddingModels}
        currentEmbeddingModel={currentEmbeddingModel} setCurrentEmbeddingModel={setCurrentEmbeddingModel}
        currentVlmModel={currentVlmModel} setCurrentVlmModel={setCurrentVlmModel}
        vlmModels={AVAILABLE_VLM_MODELS}
        progressPct={progressPct} completedCount={completedCount} totalSteps={totalSteps}
      />
      {!sidebarOpen && (
        <Button variant="ghost" size="icon-sm" onClick={() => setSidebarOpen(true)}
          className="absolute left-0 top-1/2 -translate-y-1/2 z-10">
          <HugeiconsIcon icon={ChevronRightIcon} className="size-4" />
        </Button>
      )}
      <div className="flex-1 flex flex-col overflow-hidden">
        <div className="shrink-0 bg-bg-surface/60 border-b border-border px-4 py-3 space-y-3 max-h-48 overflow-y-auto">
          {preprocSteps.length > 0 && (
            <div>
              <StepHeader label="Preprocessing" />
              <div className="flex flex-wrap gap-2">
                {preprocSteps.map(name => (
                  <StepCard key={name}
                    name={name} step={steps[name]}
                    runningStep={runningStep} selectedStep={selectedStep}
                    onSelectStep={onSelectStep}
                    infoTarget={infoTarget} setInfoTarget={setInfoTarget}
                    ocrPostCorrect={ocrPostCorrect} visionOcrPostCorrect={visionOcrPostCorrect}
                    embeddingTextSource={embeddingTextSource} retrievalStrategy={retrievalStrategy}
                    waiting={waiting}
                    runStep={runStep} stopPipeline={stopPipeline}
                  />
                ))}
              </div>
            </div>
          )}
          {STEP_GROUPS.slice(1).map((group) => {
            const groupSteps = group.steps.filter(s => enabledSet.has(s) && s !== 'review')
            if (groupSteps.length === 0 && !(group.label === 'Extraction & Validation' && enabledSet.has('review'))) return null
            return (
              <div key={group.label}>
                <StepHeader label={group.label} />
                <div className="flex flex-wrap gap-2">
                  {groupSteps.map(name => (
                    <StepCard key={name}
                      name={name} step={steps[name]}
                      runningStep={runningStep} selectedStep={selectedStep}
                      onSelectStep={onSelectStep}
                      infoTarget={infoTarget} setInfoTarget={setInfoTarget}
                      ocrPostCorrect={ocrPostCorrect} visionOcrPostCorrect={visionOcrPostCorrect}
                      embeddingTextSource={embeddingTextSource} retrievalStrategy={retrievalStrategy}
                      waiting={waiting}
                      runStep={runStep} stopPipeline={stopPipeline}
                    />
                  ))}
                  {group.label === 'Extraction & Validation' && enabledSet.has('review') && (
                    <div
                      onClick={() => onSelectStep(selectedStep === 'review' ? null : 'review')}
                      className={`flex items-center gap-2 px-3 py-2 rounded-lg border transition-all cursor-pointer shrink-0 group ${
                        selectedStep === 'review'
                          ? 'border-accent-violet/40 bg-accent-violet/12 ring-1 ring-accent-violet/30 border-l-accent-violet border-l-2'
                          : 'border-border/40 bg-bg-surface/60 hover:bg-bg-elevated/60 border-l-transparent border-l-2'
                      }`}>
                      <div className="w-5 h-5 flex items-center justify-center shrink-0">
                        <HugeiconsIcon icon={PencilIcon} className="size-3.5 text-accent-violet/70" />
                      </div>
                      <span className="text-xs font-medium truncate flex-1 text-text-muted">Human Review</span>
                    </div>
                  )}
                  {group.label === 'Evaluation' && (resultsData || steps['end_to_end_vlm']?.status === 'completed' || steps['llm_extraction']?.status === 'completed') && (
                    <div
                      onClick={() => onSelectStep(selectedStep === '__results__' ? null : '__results__')}
                      className={`flex items-center gap-2 px-3 py-2 rounded-lg border transition-all cursor-pointer shrink-0 group ${
                        selectedStep === '__results__'
                          ? 'border-accent-violet/40 bg-accent-violet/12 ring-1 ring-accent-violet/30 border-l-accent-violet border-l-2'
                          : 'border-border/40 bg-bg-surface/60 hover:bg-bg-elevated/60 border-l-[#16A34A]/60 border-l-2'
                      } ${!resultsData ? 'opacity-60' : ''}`}>
                      <div className="w-5 h-5 flex items-center justify-center shrink-0">
                        {resultsData ? (
                          <HugeiconsIcon icon={CheckmarkCircleIcon} className="size-3.5 text-accent-green" />
                        ) : (
                          <HugeiconsIcon icon={Loading01Icon} className="size-3.5 animate-spin text-accent-violet" />
                        )}
                      </div>
                      <span className={`text-xs font-medium truncate flex-1 ${resultsData ? 'text-text-primary' : 'text-accent-violet'}`}>
                        {resultsData ? 'Results' : 'Processing…'}
                      </span>
                      {resultsData && (
                        <span className="text-[10px] font-mono text-text-muted tabular-nums shrink-0">
                          {resultsData.num_pages}p · {resultsData.total_time.toFixed(1)}s
                        </span>
                      )}
                    </div>
                  )}
                </div>
              </div>
            )
          })}
        </div>
        <div className="shrink-0 bg-bg-surface border-b border-border px-4 py-2 flex items-center gap-2">
          {(waiting || anyPending) && !runningStep && (
            <Button variant="default" size="sm" onClick={() => runAll()}>
              <HugeiconsIcon icon={FastForwardIcon} className="size-3.5" />
              Run All
            </Button>
          )}
          {runningStep && (
            <Button variant="destructive" size="sm" onClick={stopPipeline}
              className="shadow-sm">
              <HugeiconsIcon icon={Cancel01Icon} className="size-3.5" />
              Stop{runningStep !== '__all__' ? ` ${STEP_LABELS[runningStep] || runningStep}` : ''}
            </Button>
          )}
          {runningStep === '__all__' && (
            <span className="text-xs text-accent-violet">Running all steps...</span>
          )}
        </div>

        <div className="flex flex-1 overflow-hidden">
          <div className="w-3/5 flex flex-col border-r border-border/50 overflow-hidden">
            {(() => {
              const pdfUrl = `/api/session/${sessionId}/pdf`
              if (selectedStep === '__results__' && resultsData) {
                const activeLocation = hoverLocation ?? (() => {
                  const f = reviewFieldsRef.current.find(ff => ff.key === activeFieldKey)
                  return f?.location
                })()
                return (
                  <PDFViewer
                    ref={viewerRef}
                    file={pdfUrl}
                    className="flex-1 min-h-0"
                    showToolbar showDownload showRotateControls
                    renderPageOverlay={({ pageNumber }) =>
                      activeLocation && activeLocation.page === pageNumber
                        ? <HumanReviewHighlight location={activeLocation} />
                        : null
                    }
                  />
                )
              }
              if (selectedStep === 'ocr' && steps.ocr?.data) {
                return (
                  <PDFViewer
                    ref={viewerRef}
                    file={pdfUrl}
                    className="flex-1 min-h-0"
                    showToolbar showDownload showRotateControls
                    renderPageOverlay={({ pageNumber }) =>
                      ocrBlocks.filter(b => b.page === pageNumber).map(block => (
                        <OcrBlockOverlay
                          key={block.id}
                          block={block}
                          isActive={block.id === activeOcrBlockId}
                        />
                      ))
                    }
                  />
                )
              }
              return (
                <PDFViewer
                  file={pdfUrl}
                  className="flex-1 min-h-0"
                  showToolbar showDownload showRotateControls
                />
              )
            })()}
          </div>
          <div className="flex-1 p-4 bg-bg-base flex flex-col overflow-y-auto">
            {(() => {
              const sn = selectedStep
              if (sn === 'review') {
                const isEndToEnd = currentMode === 'end_to_end'
                const extractionStep = isEndToEnd ? 'end_to_end_vlm' : 'llm_extraction'
                const extractionData = steps[extractionStep]?.data as Record<string, unknown> | undefined
                const vlmPages = (extractionData?.pages as Array<Record<string, unknown>> | undefined) || []
                const vlmFields = vlmPages[0]?.fields as Record<string, unknown> | undefined
                const revFields = resultsData ? resultToReviewFields(resultsData) : (vlmFields ? Object.entries(vlmFields).map(([key, value]) => ({ key, value, label: key.replace(/_/g, ' ') })) : [])
                if (revFields.length === 0) {
                  return (
                    <div className="flex flex-col items-center justify-center h-full text-sm text-text-muted gap-2">
                      <HugeiconsIcon icon={InformationCircleIcon} className="size-8 text-border" />
                      <span>Run the pipeline first to review extracted fields</span>
                    </div>
                  )
                }
                const firstFields = vlmFields || (resultsData?.pages?.[0]?.extracted_fields as Record<string, unknown> ?? {})
                const fieldKeys = Object.keys(firstFields)
                return (
                  <div className="min-h-0 flex-1 flex flex-col">
                    <div className="shrink-0 px-1 py-2 border-b border-border">
                      <div className="text-sm font-semibold text-text-primary">Human Review</div>
                      <div className="text-xs text-text-muted">Review and correct extracted fields</div>
                    </div>
                    <div className="flex-1 overflow-y-auto space-y-2 p-3">
                      {fieldKeys.map(fk => {
                        const rawVal = firstFields[fk]
                        const isEditing = fk in correctionEdits
                        const displayValue = isEditing ? correctionEdits[fk] : fmtExtractedVal(rawVal)
                        const editValue = isEditing ? correctionEdits[fk] : fmtExtractedVal(rawVal)
                        const isConfirmed = confirmedFields.has(fk)
                        return (
                          <div key={fk} className="bg-bg-surface border border-border rounded-lg p-3 space-y-1">
                            <div className="flex items-center justify-between">
                              <span className="text-xs font-semibold text-text-muted uppercase">{fk.replace(/_/g, ' ')}</span>
                              <div className="flex items-center gap-1">
                                <button
                                  onClick={() => setConfirmedFields(prev => { const n = new Set(prev); if (n.has(fk)) n.delete(fk); else n.add(fk); return n })}
                                  className={`text-[11px] px-2 py-0.5 rounded font-medium transition-colors ${
                                    isConfirmed ? 'bg-accent-green/15 text-accent-green' : 'bg-bg-elevated text-text-muted hover:text-accent-green'
                                  }`}>
                                  {isConfirmed ? '✓ Confirmed' : 'Confirm'}
                                </button>
                                <button
                                  onClick={() => {
                                    if (isEditing) {
                                      setCorrectionEdits(prev => { const n = { ...prev }; delete n[fk]; return n })
                                    } else {
                                      setCorrectionEdits(prev => ({ ...prev, [fk]: fmtExtractedVal(rawVal) }))
                                    }
                                  }}
                                  className="text-[11px] px-2 py-0.5 rounded font-medium bg-bg-elevated text-text-muted hover:text-accent-violet transition-colors">
                                  {isEditing ? 'Cancel' : 'Edit'}
                                </button>
                              </div>
                            </div>
                            {isEditing ? (
                              <input
                                value={editValue}
                                onChange={e => setCorrectionEdits(prev => ({ ...prev, [fk]: e.target.value }))}
                                className="w-full text-xs bg-bg-elevated border border-border rounded px-2 py-1.5 outline-none focus:border-accent-violet transition-colors font-mono"
                              />
                            ) : (
                              <div className="text-xs text-text-primary font-mono">{displayValue}</div>
                            )}
                          </div>
                        )
                      })}
                    </div>
                    {Object.keys(correctionEdits).length > 0 && (
                      <div className="shrink-0 border-t border-border bg-bg-surface px-4 py-3 flex items-center gap-3">
                        <Button variant="default" size="sm" onClick={handleCorrectionSave}
                          disabled={correctionStatus === 'saving'}
                          loading={correctionStatus === 'saving'}
                          className="bg-accent-violet hover:bg-accent-violet/80 text-white">
                          {correctionStatus === 'saving' ? <><HugeiconsIcon icon={Loading01Icon} className="size-3 animate-spin" /> Saving...</> : 'Save Corrections'}
                        </Button>
                        {correctionStatus === 'saved' && <span className="text-xs text-accent-green">{correctionMsg}</span>}
                        {correctionStatus === 'error' && <span className="text-xs text-accent-coral">{correctionMsg}</span>}
                      </div>
                    )}
                  </div>
                )
              }
              if (sn === '__results__') {
                if (!resultsData) {
                  return (
                    <div className="flex flex-col items-center justify-center h-full text-sm text-text-muted gap-2">
                      <HugeiconsIcon icon={Loading01Icon} className="size-8 animate-spin text-accent-violet" />
                      <span>Loading results...</span>
                    </div>
                  )
                }
                const r = resultsData
                const firstPageFields = r.pages?.[0]?.extracted_fields as Record<string, unknown> ?? {}
                const correctionFieldKeys = Object.keys(firstPageFields)
                return (
                  <Tabs defaultValue="fields" className="min-h-0 flex-1 flex flex-col">
                    <div className="flex items-center justify-between gap-3 border-b border-border pb-2 shrink-0">
                      <TabsList variant="line">
                        <TabsTrigger value="fields" className="text-xs">Fields</TabsTrigger>
                        <TabsTrigger value="validation" className="text-xs">Validation</TabsTrigger>
                      </TabsList>
                    </div>

                    <TabsContent value="fields" className="min-h-0 flex-1 flex flex-col overflow-hidden mt-2">
                      <HumanReviewPanel
                        fields={reviewFields}
                        activeFieldKey={activeFieldKey}
                        showExpected={false}
                        className="flex-1 min-h-0"
                        onFieldFocus={focusField}
                        onLocationHover={handleLocationHover}
                      />
                      <div className="shrink-0 border-t border-border bg-bg-surface px-4 py-2.5">
                        <div className="space-y-1">
                          {correctionFieldKeys.map(fk => {
                            const rawVal = firstPageFields[fk]
                            const isEditing = fk in correctionEdits
                            const displayValue = isEditing ? correctionEdits[fk] : fmtExtractedVal(rawVal)
                            const editValue = isEditing ? correctionEdits[fk] : fmtExtractedVal(rawVal)
                            return (
                              <div key={fk} className="flex items-center gap-2 group">
                                <span className="text-xs text-text-muted w-28 shrink-0 truncate uppercase">{fk.replace(/_/g, ' ')}</span>
                                <div className="flex-1 min-w-0">
                                  {isEditing ? (
                                    <input
                                      value={editValue}
                                      onChange={e => setCorrectionEdits(prev => ({ ...prev, [fk]: e.target.value }))}
                                      className="w-full text-xs bg-bg-elevated border border-border rounded px-2 py-1 outline-none focus:border-accent-violet transition-colors"
                                    />
                                  ) : (
                                    <span className="text-xs text-text-primary font-mono">{displayValue}</span>
                                  )}
                                </div>
                                <Button variant="ghost" size="icon-sm"
                                  onClick={() => {
                                    if (isEditing) {
                                      setCorrectionEdits(prev => { const next = { ...prev }; delete next[fk]; return next })
                                    } else {
                                      setCorrectionEdits(prev => ({ ...prev, [fk]: fmtExtractedVal(rawVal) }))
                                    }
                                  }}
                                  className="text-text-muted hover:text-text-primary opacity-0 group-hover:opacity-100 shrink-0"
                                  title={isEditing ? 'Cancel edit' : 'Edit field'}>
                                  {isEditing ? <HugeiconsIcon icon={Cancel01Icon} className="size-3" /> : <HugeiconsIcon icon={PencilIcon} className="size-3" />}
                                </Button>
                              </div>
                            )
                          })}
                        </div>
                        {Object.keys(correctionEdits).length > 0 && (
                          <div className="flex items-center gap-2 mt-2 pt-2 border-t border-border">
                            <Button variant="default" size="sm" onClick={handleCorrectionSave}
                              disabled={correctionStatus === 'saving'}
                              loading={correctionStatus === 'saving'}>
                              Save Corrections
                            </Button>
                            {correctionStatus === 'saved' && <span className="text-xs text-accent-green">{correctionMsg}</span>}
                            {correctionStatus === 'error' && <span className="text-xs text-accent-coral">{correctionMsg}</span>}
                          </div>
                        )}
                      </div>
                    </TabsContent>

                    <TabsContent value="validation" className="min-h-0 flex-1 overflow-auto mt-2">
                      <div className="space-y-3">
                        {/* HITL Routing: fields needing human review */}
                        {r.pages.some(p => p.field_confidence) && r.pages.map(p => {
                          const fc = p.field_confidence
                          if (!fc || Object.keys(fc).length === 0) return null
                          const needsReview = Object.entries(fc).filter(([_, v]) => v.needs_review && !_.startsWith('_'))
                          const allScores = Object.values(fc).filter(v => v.confidence > 0).map(v => v.confidence)
                          const overall = allScores.length > 0 ? allScores.reduce((a, b) => a + b, 0) / allScores.length : 0
                          return (
                            <SectionCard key={p.page_number} title={`HITL Routing — Page ${p.page_number}`}>
                              <div className="px-4 py-3 border-b border-border flex items-center justify-between">
                                <span className="text-xs font-semibold text-text-muted uppercase tracking-wider">Overall Confidence</span>
                                <span className={`text-sm font-bold font-mono tabular-nums ${
                                  overall >= 0.85 ? 'text-accent-green' : overall >= 0.7 ? 'text-accent-yellow' : 'text-accent-coral'
                                }`}>{Math.round(overall * 100)}%</span>
                              </div>
                              {needsReview.length > 0 && (
                                <div className="px-4 py-2 border-b border-border bg-accent-coral/5">
                                  <span className="text-xs font-semibold text-accent-coral">{needsReview.length} field{needsReview.length !== 1 ? 's' : ''} need{needsReview.length === 1 ? 's' : ''} review</span>
                                </div>
                              )}
                              <div className="divide-y divide-border">
                                {Object.entries(fc).filter(([k]) => !k.startsWith('_')).map(([k, v]) => (
                                  <div key={k} className="flex items-center justify-between px-4 py-2 hover:bg-bg-elevated/30 transition-colors">
                                    <div className="flex items-center gap-2 min-w-0">
                                      <div className={`w-2 h-2 rounded-full shrink-0 ${
                                        v.needs_review ? 'bg-accent-coral' :
                                        v.level === 'high' ? 'bg-accent-green' : 'bg-accent-yellow'
                                      }`} />
                                      <span className="text-xs text-text-primary truncate">{k.replace(/_/g, ' ')}</span>
                                    </div>
                                    <div className="flex items-center gap-2 shrink-0">
                                      {v.needs_review && (
                                        <span className="text-[10px] px-1.5 py-0.5 rounded-full bg-accent-coral/10 text-accent-coral font-medium">Review</span>
                                      )}
                                      <span className={`text-xs font-mono tabular-nums font-medium ${
                                        v.confidence >= 0.85 ? 'text-accent-green' : v.confidence >= 0.7 ? 'text-accent-yellow' : 'text-accent-coral'
                                      }`}>
                                        {Math.round(v.confidence * 100)}%
                                      </span>
                                    </div>
                                  </div>
                                ))}
                              </div>
                            </SectionCard>
                          )
                        })}
                        {r.evaluation && Object.keys(r.evaluation).length > 0 && (() => {
                          const docType = (r.document_type || 'invoice').replace(/_/g, ' ')
                          const catMetrics = CATEGORY_METRICS[r.document_type || 'invoice'] || CATEGORY_METRICS.invoice
                          const acc = r.evaluation?.accuracy as Record<string, unknown> | undefined
                          const faith = r.evaluation?.faithfulness as Record<string, unknown> | undefined
                          const nd = r.evaluation?.numeric_delta as Record<string, unknown> | undefined
                          const fmt = r.evaluation?.format_compliance as Record<string, unknown> | undefined
                          const det = r.evaluation?.detection_rate as Record<string, unknown> | undefined
                          const metricMap: Record<string, { score: number | null; subtitle: string }> = {
                            accuracy: { score: typeof acc?.score === 'number' ? acc.score : null, subtitle: `${acc?.exact_match ?? 0}/${acc?.total_fields ?? 0}` },
                            faithfulness: { score: typeof faith?.score === 'number' ? faith.score : null, subtitle: `${faith?.faithful ?? 0}/${faith?.total ?? 0}` },
                            token_f1: { score: typeof acc?.partial_token_f1 === 'number' ? acc.partial_token_f1 : null, subtitle: 'partial' },
                            numeric_delta: { score: typeof nd?.score === 'number' ? nd.score : null, subtitle: `${nd?.count ?? 0} fields` },
                            format_compliance: { score: typeof fmt?.score === 'number' ? fmt.score : null, subtitle: `${fmt?.passed ?? 0}/${fmt?.total ?? 0}` },
                            detection_rate: { score: typeof det?.score === 'number' ? det.score : null, subtitle: `${det?.detected ?? 0}/${det?.total ?? 0}` },
                          }
                          const relevant = catMetrics.filter(k => metricMap[k]?.score != null)
                          if (relevant.length === 0) return null
                          return (
                            <SectionCard title={`Extraction Quality — ${docType}`}>
                              <div className="p-4">
                                <div className="grid grid-cols-3 gap-3">
                                  {relevant.map(key => {
                                    const info = METRIC_INFO[key]
                                    const m = metricMap[key]
                                    return <ScoreGauge key={key} label={info?.label ?? key} score={m!.score} subtitle={m!.subtitle} info={info?.desc} />
                                  })}
                                </div>
                              </div>
                            </SectionCard>
                          )
                        })()}
                        {r.pages.some(p => p.validation) ? (
                        (() => {
                          const validPages = r.pages.filter(p => p.validation?.is_valid).length
                          const totalPages = r.pages.filter(p => p.validation).length
                          const totalErrors = r.pages.reduce((s, p) => s + (p.validation?.error_count ?? 0), 0)
                          const totalWarnings = r.pages.reduce((s, p) => s + (p.validation?.warning_count ?? 0), 0)
                          const overallScore = totalErrors + totalWarnings === 0 ? 1 : Math.max(0, 1 - (totalErrors * 0.3 + totalWarnings * 0.1))
                          return (
                            <div className="space-y-3">
                              <div className="grid grid-cols-4 gap-3 shrink-0">
                                <div className="bg-bg-surface border border-border rounded-xl p-3">
                                  <div className="text-xs font-semibold text-text-muted uppercase">Pages</div>
                                  <div className="text-lg font-bold font-mono tabular-nums text-text-primary">{validPages}/{totalPages}</div>
                                </div>
                                <div className="bg-bg-surface border border-border rounded-xl p-3">
                                  <div className="text-xs font-semibold text-text-muted uppercase">Errors</div>
                                  <div className="text-lg font-bold font-mono tabular-nums text-accent-coral">{totalErrors}</div>
                                </div>
                                <div className="bg-bg-surface border border-border rounded-xl p-3">
                                  <div className="text-xs font-semibold text-text-muted uppercase">Warnings</div>
                                  <div className="text-lg font-bold font-mono tabular-nums text-accent-yellow">{totalWarnings}</div>
                                </div>
                                <div className="bg-bg-surface border border-border rounded-xl p-3">
                                  <ScoreGauge label="Score" score={overallScore} subtitle={`${Math.round(overallScore * 100)}%`} />
                                </div>
                              </div>
                              {r.pages.map(p => {
                                if (!p.validation) return null
                                const errors = p.validation.issues.filter(i => i.severity === 'error')
                                const warnings = p.validation.issues.filter(i => i.severity !== 'error')
                                return (
                                  <SectionCard key={p.page_number} title={`Page ${p.page_number}`}>
                                    <div className="space-y-2">
                                      <div className="flex items-center gap-2">
                                        {p.validation.is_valid ? (
                                          <><HugeiconsIcon icon={CheckmarkCircleIcon} className="size-4 text-accent-green" /><span className="text-sm text-text-primary">Valid</span></>
                                        ) : (
                                          <><HugeiconsIcon icon={AlertCircleIcon} className="size-4 text-accent-coral" /><span className="text-sm text-text-primary">{p.validation.error_count} error{p.validation.error_count !== 1 ? 's' : ''}, {p.validation.warning_count} warning{p.validation.warning_count !== 1 ? 's' : ''}</span></>
                                        )}
                                      </div>
                                      {errors.length > 0 && (
                                        <div>
                                          <div className="text-xs font-semibold text-text-muted uppercase tracking-wider mb-1">Errors</div>
                                          <div className="space-y-1">
                                            {errors.map((issue, i) => (
                                              <div key={i} className="flex items-start gap-2 bg-accent-coral/10 border border-accent-coral/20 rounded-lg px-3 py-2">
                                                <HugeiconsIcon icon={AlertCircleIcon} className="size-3.5 text-accent-coral mt-0.5 shrink-0" />
                                                <div className="flex-1 min-w-0">
                                                  <div className="text-xs text-text-primary">{issue.message}</div>
                                                  <div className="text-[11px] text-text-muted font-mono mt-0.5">{issue.rule}</div>
                                                  {issue.fields.length > 0 && (
                                                    <div className="flex flex-wrap gap-1.5 mt-1.5">
                                                      {issue.fields.map(f => {
                                                        const rField = reviewFieldsRef.current.find(rf => rf.key === f)
                                                        return (
                                                          <span key={f} onClick={() => rField && focusField(rField)}
                                                            className="inline-flex items-center gap-1 px-2 py-0.5 text-[11px] font-medium rounded-full bg-bg-elevated text-text-muted font-mono cursor-pointer hover:bg-accent-violet/20 hover:text-accent-violet transition-colors">
                                                            {f}
                                                            {p.extracted_fields[f] != null && <span className="text-text-muted/60">={String(p.extracted_fields[f]).slice(0, 20)}{String(p.extracted_fields[f]).length > 20 ? '…' : ''}</span>}
                                                          </span>
                                                        )
                                                      })}
                                                    </div>
                                                  )}
                                                </div>
                                              </div>
                                            ))}
                                          </div>
                                        </div>
                                      )}
                                      {warnings.length > 0 && (
                                        <div>
                                          <div className="text-xs font-semibold text-text-muted uppercase tracking-wider mb-1">Warnings</div>
                                          <div className="space-y-1">
                                            {warnings.map((issue, i) => (
                                              <div key={i} className="flex items-start gap-2 bg-accent-yellow/10 border border-accent-yellow/20 rounded-lg px-3 py-2">
                                                <HugeiconsIcon icon={AlertCircleIcon} className="size-3.5 text-accent-yellow mt-0.5 shrink-0" />
                                                <div className="flex-1 min-w-0">
                                                  <div className="text-xs text-text-primary">{issue.message}</div>
                                                  <div className="text-[11px] text-text-muted font-mono mt-0.5">{issue.rule}</div>
                                                  {issue.fields.length > 0 && (
                                                    <div className="flex flex-wrap gap-1.5 mt-1.5">
                                                      {issue.fields.map(f => {
                                                        const rField = reviewFieldsRef.current.find(rf => rf.key === f)
                                                        return (
                                                          <span key={f} onClick={() => rField && focusField(rField)}
                                                            className="inline-flex items-center gap-1 px-2 py-0.5 text-[11px] font-medium rounded-full bg-bg-elevated text-text-muted font-mono cursor-pointer hover:bg-accent-violet/20 hover:text-accent-violet transition-colors">
                                                            {f}
                                                            {p.extracted_fields[f] != null && <span className="text-text-muted/60">={String(p.extracted_fields[f]).slice(0, 20)}{String(p.extracted_fields[f]).length > 20 ? '…' : ''}</span>}
                                                          </span>
                                                        )
                                                      })}
                                                    </div>
                                                  )}
                                                </div>
                                              </div>
                                            ))}
                                          </div>
                                        </div>
                                      )}
                                      {errors.length === 0 && warnings.length === 0 && (
                                        <div className="text-xs text-text-muted italic">All checks passed</div>
                                      )}
                                    </div>
                                  </SectionCard>
                                )
                              })}
                            </div>
                          )
                        })()
                        ) : (
                          <div className="flex items-center justify-center h-32 text-sm text-text-muted">No validation data</div>
                        )}
                      </div>
                    </TabsContent>
                  </Tabs>
                )
              }
              if (sn === 'review') {
                if (!resultsData) {
                  return (
                    <div className="flex flex-col items-center justify-center h-full text-sm text-text-muted gap-2">
                      <HugeiconsIcon icon={InformationCircleIcon} className="size-8 text-border" />
                      <span>Run the pipeline first to review extracted fields</span>
                    </div>
                  )
                }
                const r = resultsData
                const allFields = r.pages.reduce((acc, p) => ({
                  ...acc,
                  ...(p.extracted_fields as Record<string, unknown> ?? {})
                }), {} as Record<string, unknown>)
                const reviewFieldKeys = Object.keys(allFields).filter(k => !k.startsWith('LINE/'))
                const lineFieldsPerPage: Array<{ page: number; fields: Array<{ key: string; val: unknown }> }> = []
                for (const p of r.pages) {
                  const ef = p.extracted_fields as Record<string, unknown> ?? {}
                  const lineKeys = Object.keys(ef).filter(k => k.startsWith('LINE/'))
                  if (lineKeys.length > 0) {
                    lineFieldsPerPage.push({
                      page: p.page_number,
                      fields: lineKeys.map(k => ({ key: k, val: ef[k] })),
                    })
                  }
                }
                return (
                  <div className="flex flex-col h-full">
                    <div className="text-xs font-semibold text-text-muted uppercase tracking-wider mb-3 shrink-0">Human Review — Correct Extracted Fields</div>
                    <div className="flex-1 min-h-0 overflow-y-auto space-y-2">
                      {reviewFieldKeys.map(fk => {
                        const isEditing = fk in correctionEdits
                        const displayValue = isEditing ? correctionEdits[fk] : fmtExtractedVal(allFields[fk])
                        return (
                          <div key={fk} className="flex items-start gap-3 px-5 py-3 hover:bg-bg-elevated/30 transition-colors group rounded-lg border border-border/50">
                            <div className="w-32 shrink-0">
                              <span className="text-[13px] font-medium text-text-muted uppercase tracking-wider">{fk.replace(/_/g, ' ')}</span>
                            </div>
                            <div className="flex-1 min-w-0">
                              {isEditing ? (
                                <textarea
                                  className="w-full text-sm text-text-primary font-mono bg-bg-elevated px-3 py-1.5 rounded-lg border border-border focus:border-accent-violet focus:ring-1 focus:ring-accent-violet/30 outline-none resize-y min-h-[2.5rem]"
                                  style={{ fontFamily: "'JetBrains Mono', monospace" }}
                                  value={displayValue}
                                  onChange={e => setCorrectionEdits(prev => ({ ...prev, [fk]: e.target.value }))}
                                  rows={2}
                                />
                              ) : (
                                <div className="text-sm text-text-primary font-mono bg-bg-elevated/50 px-3 py-1.5 rounded-lg border border-border/50">
                                  {fmtExtractedVal(allFields[fk])}
                                </div>
                              )}
                            </div>
                            <Button variant="ghost" size="icon-sm"
                              onClick={() => {
                                if (isEditing) {
                                  setCorrectionEdits(prev => { const next = { ...prev }; delete next[fk]; return next })
                                } else {
                                  const raw = allFields[fk]
                                  setCorrectionEdits(prev => ({ ...prev, [fk]: fmtExtractedVal(raw) }))
                                }
                              }}
                              className="text-text-muted hover:text-text-primary opacity-0 group-hover:opacity-100 shrink-0"
                              title={isEditing ? 'Cancel edit' : 'Edit field'}>
                              {isEditing ? <HugeiconsIcon icon={Cancel01Icon} className="size-3.5" /> : <HugeiconsIcon icon={PencilIcon} className="size-3.5" />}
                            </Button>
                          </div>
                        )
                      })}
                      {lineFieldsPerPage.length > 0 && (
                        <div className="mt-3 pt-3 border-t border-border">
                          <div className="text-xs font-semibold text-text-muted uppercase tracking-wider mb-3">Line Item Fields</div>
                          <div className="space-y-3">
                            {lineFieldsPerPage.map(({ page, fields }) => (
                              <div key={`line-p${page}`} className="rounded-lg border border-border/50 overflow-hidden">
                                <div className="bg-bg-elevated/30 px-4 py-1.5 border-b border-border">
                                  <span className="text-xs font-semibold text-text-muted">Page {page}</span>
                                </div>
                                <div className="divide-y divide-border/40">
                                  {fields.map(({ key, val }) => {
                                    const fk = `p${page}__${key}`
                                    const isEditing = fk in correctionEdits
                                    const displayValue = isEditing ? correctionEdits[fk] : fmtExtractedVal(val)
                                    return (
                                      <div key={fk} className="flex items-start gap-3 px-4 py-2.5 hover:bg-bg-elevated/20 transition-colors group">
                                        <div className="w-32 shrink-0">
                                          <span className="text-xs font-medium text-text-muted uppercase tracking-wider">{key.replace(/_/g, ' ')}</span>
                                        </div>
                                        <div className="flex-1 min-w-0">
                                          {isEditing ? (
                                            <textarea
                                              className="w-full text-xs text-text-primary font-mono bg-bg-elevated px-2.5 py-1.5 rounded-md border border-border focus:border-accent-violet focus:ring-1 focus:ring-accent-violet/30 outline-none resize-y min-h-[2rem]"
                                              style={{ fontFamily: "'JetBrains Mono', monospace" }}
                                              value={displayValue}
                                              onChange={e => setCorrectionEdits(prev => ({ ...prev, [fk]: e.target.value }))}
                                              rows={1}
                                            />
                                          ) : (
                                            <div className="text-xs text-text-primary font-mono bg-bg-elevated/50 px-2.5 py-1.5 rounded-md border border-border/50">
                                              {fmtExtractedVal(val)}
                                            </div>
                                          )}
                                        </div>
                                        <Button variant="ghost" size="icon-sm"
                                          onClick={() => {
                                            if (isEditing) {
                                              setCorrectionEdits(prev => { const next = { ...prev }; delete next[fk]; return next })
                                            } else {
                                              const raw = val
                                              setCorrectionEdits(prev => ({ ...prev, [fk]: fmtExtractedVal(raw) }))
                                            }
                                          }}
                                          className="text-text-muted hover:text-text-primary opacity-0 group-hover:opacity-100 shrink-0"
                                          title={isEditing ? 'Cancel edit' : 'Edit field'}>
                                          {isEditing ? <HugeiconsIcon icon={Cancel01Icon} className="size-3" /> : <HugeiconsIcon icon={PencilIcon} className="size-3" />}
                                        </Button>
                                      </div>
                                    )
                                  })}
                                </div>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                    {Object.keys(correctionEdits).length > 0 && (
                      <div className="shrink-0 border-t border-border bg-bg-surface px-4 py-3 flex items-center gap-3">
                        <Button variant="default" size="sm" onClick={handleCorrectionSave}
                          disabled={correctionStatus === 'saving'}
                          loading={correctionStatus === 'saving'}
                          className="bg-accent-violet hover:bg-accent-violet/80 text-white">
                          {correctionStatus === 'saving' ? <><HugeiconsIcon icon={Loading01Icon} className="size-3 animate-spin" /> Saving...</> : 'Save Corrections'}
                        </Button>
                        {correctionStatus === 'saved' && <span className="text-xs text-accent-green">{correctionMsg}</span>}
                        {correctionStatus === 'error' && <span className="text-xs text-accent-coral">{correctionMsg}</span>}
                      </div>
                    )}
                  </div>
                )
              }
              if (!sn) {
                return (
                  <div className="flex flex-col items-center justify-center h-full text-sm text-text-muted gap-2">
                    <HugeiconsIcon icon={InformationCircleIcon} className="size-8 text-border" />
                    <span>Select a completed step to view details</span>
                  </div>
                )
              }
              const stepName = sn as string
              const st = steps[stepName]
              if (st?.status !== 'completed' || !st?.data) {
                return (
                  <div className="flex flex-col items-center justify-center h-full text-sm text-text-muted gap-2">
                    <HugeiconsIcon icon={InformationCircleIcon} className="size-8 text-border" />
                    <span>Select a completed step to view details</span>
                  </div>
                )
              }
              if (stepName === 'ocr') {
                return (
                  <OcrBlocksPanel
                    blocks={ocrBlocks}
                    activeBlockId={activeOcrBlockId}
                    onBlockFocus={focusOcrBlock}
                  />
                )
              }
              return (
              <div className="space-y-3">
                {st?.elapsed != null && (
                  <div className="flex items-center gap-1.5 text-xs text-text-muted bg-bg-surface rounded-lg px-3 py-1.5 border border-border w-fit">
                    <HugeiconsIcon icon={Clock01Icon} className="size-3" />
                    {fmtTime(st.elapsed)}
                  </div>
                )}
                <ErrorBoundary key={stepName}>
                  <PipelineStepDataView stepName={stepName} data={st!.data as Record<string, unknown>} sessionId={sessionId} />
                </ErrorBoundary>
              </div>
              )
            })()}
          </div>
        </div>
      </div>
      {infoTarget && createPortal(
        <div style={{ position: 'fixed', top: infoTarget.top + 4, left: infoTarget.left, zIndex: 9999 }}>
          <div style={{ width: infoTarget.width }}>
            <StepInfoTooltip step={infoTarget.step} onClose={() => setInfoTarget(null)} />
          </div>
        </div>,
        document.body
      )}
      {qaPanel}
      {qaButton}
    </div>
  )
}
