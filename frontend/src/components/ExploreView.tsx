import { Button } from "@/components/ui/button"
import { useState, useEffect, useRef, useCallback } from 'react'
import { createPortal } from 'react-dom'
import { HugeiconsIcon } from "@hugeicons/react"
import {
  CheckmarkCircleIcon, Loading01Icon, AlertCircleIcon, EyeIcon, EyeOffIcon,
  CodeIcon, AiNetworkIcon, ChevronLeftIcon, ChevronRightIcon, Clock01Icon,
  InformationCircleIcon, BarChartIcon, Cancel01Icon, PencilIcon, SaveIcon, BubbleChatIcon, MailSend01Icon,
  Image01Icon,
} from "@hugeicons/core-free-icons"
import KnowledgeGraphView from './KnowledgeGraph'
import MarkdownPreview from './MarkdownPreview'
import { JsonViewer } from './JsonViewer'
import { CompareView } from './CompareView'
import { QAMessage } from './QAMessage'
import { ScoreGauge, FieldMetricRow, IssueRow } from './ScoreGauge'
import { StepInfoTooltip } from './StepInfoTooltip'
import { STEP_LABELS, STEP_ORDER, STEP_GROUPS, fmtTime, DEFAULT_QA_PROMPT, findFieldInOcr } from './constants'
import { PDFViewer } from '@/components/ui/pdf-viewer'
import type { PipelineResult, PageResult, LineItem } from '../types'

/* ── Explorer Section Card ── */

function ExploreSectionCard({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="bg-bg-surface border border-border rounded-xl overflow-hidden">
      <div className="flex items-center justify-between px-5 py-3 border-b border-border">
        <h3 className="text-sm font-semibold text-text-primary">{title}</h3>
      </div>
      {children}
    </div>
  )
}

/* ── Explorer detail components ── */

function ExploreIngestionDetail({ result }: { result: PipelineResult }) {
  const [showFragments, setShowFragments] = useState(false)
  const fragments = result.pages[0]?.page_fragments || []
  return (
    <div className="space-y-3">
      <ExploreSectionCard title="Document Information">
        <div className="p-5 grid grid-cols-2 gap-4">
          <div><div className="text-xs font-medium text-text-muted uppercase tracking-wider mb-0.5">Type</div><div className="text-sm font-medium text-text-primary">{result.document_type || 'Unknown'}</div></div>
          <div><div className="text-xs font-medium text-text-muted uppercase tracking-wider mb-0.5">Pages</div><div className="text-sm font-medium text-text-primary">{result.num_pages}</div></div>
          <div><div className="text-xs font-medium text-text-muted uppercase tracking-wider mb-0.5">Input</div><div className="text-sm text-text-primary truncate font-mono">{result.input_path}</div></div>
        </div>
        <div className="px-5 py-3 border-t border-border bg-bg-elevated/50 rounded-b-xl flex items-center gap-2 text-xs text-text-muted">
          <HugeiconsIcon icon={CheckmarkCircleIcon} className="size-3.5 text-accent-green" />Ingested successfully
        </div>
      </ExploreSectionCard>
      {fragments.length > 0 && (
        <ExploreSectionCard title="">
          <Button variant="ghost" onClick={() => setShowFragments(!showFragments)} className="w-full flex items-center justify-between px-5 py-3">
            <div className="flex items-center gap-2"><HugeiconsIcon icon={CodeIcon} className="size-4 text-text-muted" /><h3 className="text-sm font-semibold text-text-primary">Page Fragments</h3><span className="inline-flex items-center gap-1 px-2 py-0.5 text-xs font-medium rounded-full bg-bg-elevated text-text-muted text-xs">{fragments.length}</span></div>
            <HugeiconsIcon icon={ChevronRightIcon} className={`size-4 text-text-muted transition-transform duration-200 ${showFragments ? 'rotate-90' : ''}`} />
          </Button>
          {showFragments && (
            <div className="px-5 pb-4 space-y-2 max-h-96 overflow-y-auto animate-fade-in">
              {fragments.map((f, i) => (
                <div key={i} className="flex items-start gap-3 text-xs">
                  <span className="w-16 shrink-0 text-text-muted font-mono text-right tabular-nums">#{f.reading_order}</span>
                  <span className={`w-14 shrink-0 px-1.5 py-0.5 rounded text-center font-medium text-xs ${f.fragment_type === 'title' ? 'bg-purple-500/10 text-purple-400' : f.fragment_type === 'table' ? 'bg-blue-500/10 text-blue-400' : f.fragment_type === 'field' ? 'bg-accent-yellow/10 text-accent-yellow' : 'bg-bg-elevated text-text-muted'}`}>{f.fragment_type}</span>
                  <span className="text-text-muted font-mono break-all line-clamp-2 text-[13px]">{JSON.stringify(f.content).slice(0, 120)}</span>
                </div>
              ))}
            </div>
          )}
        </ExploreSectionCard>
      )}
    </div>
  )
}

function ExploreOcrDetail({ page }: { page: PageResult }) {
  const [showFull, setShowFull] = useState(false)
  const [mode, setMode] = useState<'plain' | 'raw' | 'preview'>('preview')
  const text = page.ocr_text || ''
  const markdown = page.ocr_markdown || ''
  const modes = [
    { key: 'preview', label: 'Preview' },
    { key: 'raw', label: 'Formatted' },
    { key: 'plain', label: 'Plain' },
  ] as const
  const content = mode === 'plain' ? text : markdown
  const displayContent = showFull ? content : content.slice(0, 3000)
  return (
    <ExploreSectionCard title="OCR Text Transcription">
      <div className="flex items-center justify-between px-5 py-2.5 border-b border-border bg-bg-elevated/50">
        <div className="flex items-center gap-3 text-xs text-text-muted">
          <HugeiconsIcon icon={Image01Icon} className="size-3.5 text-accent-violet/70" /><span>{page.ocr_word_count} words</span>
          <div className="flex gap-0.5 ml-1 rounded-md bg-bg-elevated border border-accent-violet/30 overflow-hidden">
            {modes.map(m => (
              <Button variant="ghost" size="xs" key={m.key} onClick={() => setMode(m.key)}
                className={`${
                  mode === m.key ? 'bg-accent-violet/20 text-accent-violet' : 'text-text-muted hover:text-text-primary'
                }`}>
                {m.label}
              </Button>
            ))}
          </div>
        </div>
        {content.length > 3000 && (
          <Button variant="link" size="sm" onClick={() => setShowFull(!showFull)} className="flex items-center gap-1 text-xs text-accent-violet font-medium">
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
    </ExploreSectionCard>
  )
}

function ExploreVisionOcrDetail({ page }: { page: PageResult }) {
  const [showFull, setShowFull] = useState(false)
  const [mode, setMode] = useState<'raw' | 'plain'>('raw')
  const text = page.vlm_text || ''
  const markdown = page.vlm_markdown || ''
  const content = mode === 'plain' ? text : markdown
  const displayContent = showFull ? content : content.slice(0, 3000)
  return (
    <ExploreSectionCard title="Vision OCR (VLM)">
      <div className="flex items-center justify-between px-5 py-2.5 border-b border-border bg-bg-elevated/50">
        <div className="flex items-center gap-3 text-xs text-text-muted">
          <HugeiconsIcon icon={EyeIcon} className="size-3.5 text-accent-violet/70" /><span>gemma3:4b VLM</span>
          <div className="flex gap-0.5 ml-1 rounded-md bg-bg-elevated border border-accent-violet/30 overflow-hidden">
            {([{ key: 'raw', label: 'Formatted' }, { key: 'plain', label: 'Plain' }] as const).map(m => (
              <Button variant="ghost" size="xs" key={m.key} onClick={() => setMode(m.key)}
                className={`${
                  mode === m.key ? 'bg-accent-violet/20 text-accent-violet' : 'text-text-muted hover:text-text-primary'
                }`}>
                {m.label}
              </Button>
            ))}
          </div>
        </div>
        {content.length > 3000 && (
          <Button variant="link" size="sm" onClick={() => setShowFull(!showFull)} className="flex items-center gap-1 text-xs text-accent-violet font-medium">
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
    </ExploreSectionCard>
  )
}

function ExploreRetrievalDetail({ page }: { page: PageResult }) {
  const examples = page.retrieved_examples || []
  const [expanded, setExpanded] = useState<number | null>(null)
  return (
    <ExploreSectionCard title="Retrieved Examples">
      {examples.length === 0 ? <div className="p-5 text-sm text-text-muted italic">No examples retrieved</div> : (
        <div className="divide-y divide-border">{examples.map((ex, i) => (
          <div key={i} className="hover:bg-bg-elevated/30 transition-colors">
            <div className="px-5 py-3">
              <div className="flex items-center justify-between mb-2">
                <span className="text-[13px] font-semibold text-accent-violet bg-accent-violet/10 px-2 py-0.5 rounded-md">Example {i + 1}</span>
                {ex.source && <span className="text-xs text-text-muted font-mono">{ex.source}</span>}
              </div>
              <pre className="text-xs text-text-muted font-mono leading-relaxed whitespace-pre-wrap line-clamp-3 mb-2" style={{ fontFamily: "'JetBrains Mono', monospace" }}>{ex.ocr_text || ''}</pre>
              {ex.fields && Object.keys(ex.fields).length > 0 && (
                <div className="flex flex-wrap gap-1.5 mb-2">{Object.entries(ex.fields).map(([k, v]) => (
                  <span key={k} className="text-xs bg-bg-elevated text-text-muted px-1.5 py-0.5 rounded font-mono">{k}: {String(v).slice(0, 30)}</span>
                ))}</div>
              )}
              {ex.image_path && (
                <Button variant="link" size="sm" onClick={() => setExpanded(expanded === i ? null : i)} className="text-[13px] text-accent-violet font-medium">
                  {expanded === i ? 'Hide source' : 'Show source document'}
                </Button>
              )}
              {ex.image_path && expanded === i && (
                <div className="mt-2 border border-border rounded-lg overflow-hidden animate-fade-in">
                  <img src={`/api/image/${encodeURIComponent(ex.image_path)}`} alt="Source" className="w-full h-auto max-h-72 object-contain bg-bg-base" />
                </div>
              )}
            </div>
          </div>
        ))}</div>
      )}
    </ExploreSectionCard>
  )
}

function ExploreRagDetail({ page }: { page: PageResult }) {
  const rawRules = (page.rag_rules || []) as Array<Record<string, unknown>>
  const rawTemplates = (page.rag_templates || []) as Array<Record<string, unknown>>
  return (
    <ExploreSectionCard title="RAG Context">
      <div className="p-5 space-y-4">
        <div><div className="text-[13px] font-semibold text-text-muted uppercase tracking-wider mb-2">Field Rules ({rawRules.length})</div>
          {rawRules.length === 0 ? <p className="text-xs text-text-muted italic">No rules</p> : (
            <div className="space-y-1.5">{rawRules.map((r, i) => (
              <div key={i} className="text-xs bg-bg-elevated/50 px-3 py-2 rounded-lg border border-border/50 space-y-0.5">
                <span className="font-semibold text-text-primary">{String(r.field_name ?? '')}</span>
                <p className="text-text-muted">{String(r.description ?? '')}</p>
                {Array.isArray(r.format_patterns) && r.format_patterns.length > 0 && <p className="text-text-muted text-xs font-mono">Formats: {(r.format_patterns as string[]).join(', ')}</p>}
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
    </ExploreSectionCard>
  )
}

function ExploreLlmDetail({ page, onFieldSelect }: { page: PageResult; onFieldSelect?: (f: string) => void }) {
  const [showPrompt, setShowPrompt] = useState(false)
  const [edits, setEdits] = useState<Record<string, string>>({})
  const [saving, setSaving] = useState(false)
  const fields = page.extracted_fields || {}
  const prompt = page.last_prompt || ''

  const handleEdit = (field: string, val: string) => {
    setEdits(prev => ({ ...prev, [field]: val }))
  }

  const handleSave = async () => {
    setSaving(true)
    const body = JSON.stringify({ corrections: edits })
    try {
      const res = await fetch(`/api/correct/${page.session_id || ''}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body,
      })
      if (!res.ok) throw new Error('Save failed')
      setEdits({})
    } catch (e) {
      console.error('Failed to save corrections:', e)
    }
    setSaving(false)
  }

  const hasEdits = Object.keys(edits).length > 0

  return (
    <div className="space-y-3">
      <ExploreSectionCard title="Extracted Fields">
        {Object.keys(fields).length === 0 ? <div className="p-5 text-sm text-text-muted italic">No fields extracted</div> : (
          <div className="divide-y divide-border">{Object.entries(fields).map(([field, value]) => {
            const isEditing = field in edits
            const displayValue = isEditing ? edits[field] : (typeof value === 'object' && value !== null ? JSON.stringify(value) : String(value ?? ''))
            return (
              <div key={field} className="flex items-start gap-3 px-5 py-3 hover:bg-bg-elevated/30 transition-colors group">
                <div className="w-32 shrink-0"><span className="text-[13px] font-medium text-text-muted uppercase tracking-wider">{field.replace(/_/g, ' ')}</span></div>
                <div className="flex-1 min-w-0">
                  {isEditing ? (
                    <textarea
                      className="w-full text-sm text-text-primary font-mono bg-bg-elevated px-3 py-1.5 rounded-lg border border-border focus:border-accent-violet focus:ring-1 focus:ring-accent-violet/30 outline-none resize-y min-h-[2.5rem]"
                      style={{ fontFamily: "'JetBrains Mono', monospace" }}
                      value={displayValue}
                      onChange={e => handleEdit(field, e.target.value)}
                      rows={2}
                    />
                  ) : Array.isArray(value) ? (
                    <div className="space-y-1">{(value as Array<Record<string, unknown>>).map((item, i) => (
                      <Button key={i} variant="ghost" onClick={() => onFieldSelect?.(field)}
                        className="text-sm text-text-primary font-mono bg-bg-elevated/50 px-3 py-1.5 rounded-lg border border-border/50 hover:bg-accent-violet/10 hover:border-accent-violet/40 text-left w-full"
                        style={{ fontFamily: "'JetBrains Mono', monospace" }}>{JSON.stringify(item)}</Button>
                    ))}</div>
                  ) : (
                    <Button variant="ghost" onClick={() => onFieldSelect?.(field)}
                      className="text-sm text-text-primary font-mono bg-bg-elevated/50 px-3 py-1.5 rounded-lg border border-border/50 hover:bg-accent-violet/10 hover:border-accent-violet/40 text-left w-full"
                      style={{ fontFamily: "'JetBrains Mono', monospace" }}>
                      {typeof value === 'object' && value !== null ? JSON.stringify(value) : String(value)}
                    </Button>
                  )}
                </div>
                <Button
                  variant="ghost" size="icon-sm"
                  onClick={() => {
                    if (isEditing) {
                      setEdits(prev => { const next = { ...prev }; delete next[field]; return next })
                    } else {
                      handleEdit(field, typeof value === 'object' && value !== null ? JSON.stringify(value) : String(value ?? ''))
                    }
                  }}
                  className="text-text-muted hover:text-text-primary opacity-0 group-hover:opacity-100"
                  title={isEditing ? 'Cancel edit' : 'Edit field'}
                >
                  {isEditing ? <HugeiconsIcon icon={Cancel01Icon} className="size-3.5" /> : <HugeiconsIcon icon={PencilIcon} className="size-3.5" />}
                </Button>
                {!isEditing && (page.validation?.issues?.filter(i => i.fields?.includes(field)).length ? <HugeiconsIcon icon={AlertCircleIcon} className="size-4 text-accent-yellow shrink-0 mt-0.5" /> : value ? <HugeiconsIcon icon={CheckmarkCircleIcon} className="size-4 text-accent-violet shrink-0 mt-0.5" /> : null)}
              </div>
            )
          })}</div>
        )}
        {hasEdits && (
          <div className="px-5 py-3 border-t border-border bg-bg-elevated/50 rounded-b-xl flex items-center gap-3">
            <Button variant="default" size="sm" onClick={handleSave} disabled={saving}
              className="bg-accent-violet hover:bg-accent-violet/80 text-white">
              {saving ? <HugeiconsIcon icon={Loading01Icon} className="size-3 animate-spin" /> : <HugeiconsIcon icon={SaveIcon} className="size-3" />}
              {saving ? 'Saving...' : 'Save Corrections'}
            </Button>
            <Button variant="ghost" size="sm" onClick={() => setEdits({})} disabled={saving}>
              Cancel all
            </Button>
          </div>
        )}
      </ExploreSectionCard>
      {prompt && (
        <ExploreSectionCard title="">
          <Button variant="ghost" onClick={() => setShowPrompt(!showPrompt)} className="w-full flex items-center justify-between px-5 py-3">
            <div className="flex items-center gap-2"><HugeiconsIcon icon={CodeIcon} className="size-4 text-text-muted" /><h3 className="text-sm font-semibold text-text-primary">LLM Prompt</h3></div>
            <div className="flex items-center gap-1 text-xs text-text-muted">{showPrompt ? <HugeiconsIcon icon={EyeOffIcon} className="size-3" /> : <HugeiconsIcon icon={EyeIcon} className="size-3" />}{showPrompt ? 'Hide' : 'Show'}</div>
          </Button>
          {showPrompt && (
            <div className="px-5 pb-4 animate-fade-in">
              <pre className="text-xs text-text-muted font-mono leading-relaxed whitespace-pre-wrap bg-bg-base p-4 rounded-lg border border-border max-h-96 overflow-y-auto" style={{ fontFamily: "'JetBrains Mono', monospace" }}>{prompt}</pre>
            </div>
          )}
        </ExploreSectionCard>
      )}
    </div>
  )
}

function ExploreValidationDetail({ page }: { page: PageResult }) {
  const v = page.validation
  if (!v) return <div className="text-sm text-text-muted italic p-4">No validation results</div>
  const allIssues = v.issues || []
  const errors = allIssues.filter(i => i.severity === 'error')
  const warnings = allIssues.filter(i => i.severity === 'warning')
  return (
    <ExploreSectionCard title="Field Validation">
      <div className="flex items-center gap-3 px-5 py-3 border-b border-border">
        <div className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium ${v.is_valid ? 'bg-accent-green/10 text-accent-green' : 'bg-accent-yellow/10 text-accent-yellow'}`}>
          {v.is_valid ? <HugeiconsIcon icon={CheckmarkCircleIcon} className="size-3" /> : <HugeiconsIcon icon={AlertCircleIcon} className="size-3" />}{v.is_valid ? 'Valid' : 'Issues'}
        </div>
      </div>
      {allIssues.length === 0 ? <div className="p-5 text-sm text-text-muted italic">All checks passed</div> : (
        <div className="divide-y divide-border">
          {errors.map((issue, i) => <IssueRow key={`e-${i}`} issue={issue} type="error" />)}
          {warnings.map((issue, i) => <IssueRow key={`w-${i}`} issue={issue} type="warning" />)}
        </div>
      )}
    </ExploreSectionCard>
  )
}

function ExploreKgDetail({ page }: { page: PageResult }) {
  const graph = page.knowledge_graph
  return (
    <div>{graph ? <KnowledgeGraphView graph={graph} height={400} /> : (
      <ExploreSectionCard title=""><div className="p-5 text-center py-8"><HugeiconsIcon icon={AiNetworkIcon} className="size-10 text-text-muted mx-auto mb-3" /><p className="text-sm text-text-muted">No knowledge graph available</p></div></ExploreSectionCard>
    )}</div>
  )
}

function ExploreEmbeddingDetail({ page }: { page: PageResult }) {
  return (
    <ExploreSectionCard title="Page Embedding">
      <div className="p-5 space-y-2">
        <div className="flex items-center justify-between text-sm"><span className="text-text-muted">OCR word count</span><span className="text-text-primary font-mono tabular-nums">{page.ocr_word_count}</span></div>
        <div className="flex items-center justify-between text-sm"><span className="text-text-muted">Image dimensions</span><span className="text-text-primary font-mono tabular-nums">{page.image_width} × {page.image_height}</span></div>
      </div>
    </ExploreSectionCard>
  )
}

function ExploreCrossPageDetail({ page }: { page: PageResult }) {
  const entities = page.linked_entities || []
  return (
    <ExploreSectionCard title="Cross-Page Resolution">
      {entities.length === 0 ? <div className="p-5 text-sm text-text-muted italic">No cross-page links</div> : (
        <div className="divide-y divide-border">{entities.map((e, i) => (
          <div key={i} className="px-5 py-3 hover:bg-bg-elevated/30 transition-colors">
            <div className="flex items-center gap-2 text-sm text-text-primary">
              <span className="font-medium">{String(e.supplier || '')}</span>
              {!!e.address && <span className="text-text-muted">· {String(e.address)}</span>}
              {!!e.page && <span className="ml-auto text-xs text-text-muted font-mono">page {String(e.page)}</span>}
            </div>
          </div>
        ))}</div>
      )}
    </ExploreSectionCard>
  )
}

function ExploreEvalDetail({ result }: { result: PipelineResult }) {
  const metrics = result.evaluation || {} as Record<string, unknown>
  const accuracy = metrics.accuracy as Record<string, unknown> | undefined
  const faithfulness = metrics.faithfulness as Record<string, unknown> | undefined
  const perField = accuracy?.per_field as Record<string, { count: number; exact_match: number; avg_token_f1: number; entries?: Array<Record<string, unknown>> }> | undefined

  if (Object.keys(metrics).length === 0) return <div className="bg-bg-surface border border-border rounded-xl p-5 text-sm text-text-muted italic">No evaluation data</div>

  const accScore = typeof accuracy?.score === 'number' ? accuracy.score : null
  const accExact = typeof accuracy?.exact_match === 'number' ? accuracy.exact_match : 0
  const accTotal = typeof accuracy?.total_fields === 'number' ? accuracy.total_fields : 0
  const accTokenF1 = typeof accuracy?.partial_token_f1 === 'number' ? accuracy.partial_token_f1 : null
  const faithScore = typeof faithfulness?.score === 'number' ? faithfulness.score : null
  const faithFul = typeof faithfulness?.faithful === 'number' ? faithfulness.faithful : 0
  const faithTotal = typeof faithfulness?.total === 'number' ? faithfulness.total : 0
  const faithPerField: Record<string, number> = faithfulness?.per_field && typeof faithfulness.per_field === 'object' ? faithfulness.per_field as Record<string, number> : {}

  const timing = result.timing || {}
  const totalTime = result.total_time || 0
  const numPages = result.num_pages || 1
  const numFields = accTotal || 0
  const throughput = totalTime > 0 ? `${(numPages / totalTime).toFixed(2)} pg/s, ${numFields > 0 ? `${(numFields / totalTime).toFixed(2)} fld/s` : ''}` : '—'
  const maxTime = Math.max(...Object.values(timing), 0.001)

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-4 gap-3">
        <ScoreGauge label="Exact Match" score={accScore} subtitle={`${accExact}/${accTotal}`} />
        <ScoreGauge label="Token F1" score={accTokenF1} subtitle="partial credit" />
        <ScoreGauge label="Faithfulness" score={faithScore} subtitle={`${faithFul}/${faithTotal}`} />
        <div className="bg-bg-surface border border-border rounded-xl p-4">
          <div className="text-xs font-semibold text-text-muted uppercase tracking-wider mb-1">Throughput</div>
          <div className="flex items-baseline gap-2">
            <span className="text-2xl font-bold font-mono tabular-nums text-accent-violet">{totalTime.toFixed(1)}s</span>
            <span className="text-xs text-text-muted">total</span>
          </div>
          <div className="mt-1 text-xs text-text-muted font-mono">{throughput}</div>
        </div>
      </div>

      {Object.keys(timing).length > 0 && (
        <ExploreSectionCard title="Step Timing">
          <div className="p-4 space-y-2">
            {Object.entries(timing).sort(([, a], [, b]) => (b as number) - (a as number)).map(([step, t]) => {
              const widthPct = Math.max((t as number) / maxTime * 100, 2)
              return (
                <div key={step} className="flex items-center gap-3">
                  <span className="text-[13px] font-medium text-text-muted min-w-[8rem] truncate">{STEP_LABELS[step] || step}</span>
                  <div className="flex-1 h-2 bg-border rounded-full overflow-hidden">
                    <div className="h-full rounded-full bg-accent-violet/60" style={{ width: `${widthPct}%` }} />
                  </div>
                  <span className="text-[13px] font-mono text-text-muted tabular-nums min-w-[4rem] text-right">{t.toFixed(1)}s</span>
                </div>
              )
            })}
            <div className="flex items-center gap-3 pt-2 border-t border-border">
              <span className="text-[13px] font-semibold text-text-primary min-w-[8rem]">Total</span>
              <div className="flex-1" />
              <span className="text-[13px] font-mono font-semibold text-text-primary tabular-nums min-w-[4rem] text-right">{totalTime.toFixed(1)}s</span>
            </div>
          </div>
        </ExploreSectionCard>
      )}

      {perField && Object.keys(perField).length > 0 && (
        <ExploreSectionCard title="Per-Field Accuracy (F1 → exact match)"><div className="p-3 space-y-2">
          {Object.entries(perField).sort(([, a], [, b]) => a.avg_token_f1 - b.avg_token_f1).map(([name, metric]) => (
            <FieldMetricRow key={name} name={name} metric={metric} />
          ))}
        </div></ExploreSectionCard>
      )}

      {Object.keys(faithPerField).length > 0 && (
        <ExploreSectionCard title="Faithfulness by Field"><div className="p-3 grid grid-cols-2 gap-2">
          {Object.entries(faithPerField).sort(([, a], [, b]) => b - a).map(([field, score]) => (
            <div key={field} className="flex items-center justify-between px-3 py-1.5 bg-bg-elevated/50 rounded-lg text-xs">
              <span className="text-text-muted">{field}</span>
              <span className={`font-mono font-semibold tabular-nums ${score >= 0.8 ? 'text-accent-violet' : score >= 0.5 ? 'text-accent-yellow' : 'text-accent-coral'}`}>{Math.round(score * 100)}%</span>
            </div>
          ))}
        </div></ExploreSectionCard>
      )}
    </div>
  )
}

/* ── renderStepDetail ── */

export function renderStepDetail(step: string, page: PageResult, result: PipelineResult, onFieldSelect?: (f: string) => void) {
  switch (step) {
    case 'ingestion': return <ExploreIngestionDetail result={result} />
    case 'ocr': return <ExploreOcrDetail page={page} />
    case 'vision_ocr': return <ExploreVisionOcrDetail page={page} />
    case 'hybrid_ocr': return <div className="text-sm text-text-muted">Hybrid OCR: PaddleOCR layout + VLM text. See Pipeline view for details.</div>
    case 'document_graph': return <div className="text-sm text-text-muted">Document Graph: spatial graph from OCR boxes. See Pipeline view for details.</div>
    case 'end_to_end_vlm': return <div className="text-sm text-text-muted">End-to-End VLM: direct field extraction. See Pipeline view for details.</div>
    case 'document_classifier': return <div className="text-sm text-text-muted">Document classification ran after ingestion. Per-page types and document type set. Check Pipeline view for details.</div>
    case 'embedding': return <ExploreEmbeddingDetail page={page} />
    case 'retrieval': return <ExploreRetrievalDetail page={page} />
    case 'rag': return <ExploreRagDetail page={page} />
    case 'llm_extraction': return <ExploreLlmDetail page={page} onFieldSelect={onFieldSelect} />
    case 'validation': return <ExploreValidationDetail page={page} />
    case 'cross_page': return <ExploreCrossPageDetail page={page} />
    case 'knowledge_graph': return <ExploreKgDetail page={page} />
    case 'evaluation': return <ExploreEvalDetail result={result} />
    default: return <div className="text-sm text-text-muted">Select a step to view details</div>
  }
}

/* ── Explore View ── */

export function ExploreView({ result, selectedStep, setSelectedStep, activePage, setActivePage }: {
  result: PipelineResult
  selectedStep: string | null
  setSelectedStep: (s: string) => void
  activePage: number
  setActivePage: (n: number) => void
}) {
  const [infoTarget, setInfoTarget] = useState<{step: string; top: number; left: number; width: number} | null>(null)

  const hasStep = (s: string) => {
    if (s === 'evaluation') return result.evaluation != null
    if (s === 'knowledge_graph') return result.pages.some(p => !!p.knowledge_graph?.nodes?.length)
    return result.timing?.[s] != null
  }
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const groupedStepNames = new Set(STEP_GROUPS.slice(1).flatMap(g => g.steps))
  const preprocSteps = STEP_ORDER.filter(s => hasStep(s) && !groupedStepNames.has(s))
  const [qaOpen, setQaOpen] = useState(false)
  const [qaMessages, setQaMessages] = useState<Array<{role: string; content: string; evidence?: Record<string, string>}>>([])
  const [qaInput, setQaInput] = useState('')
  const [qaLoading, setQaLoading] = useState(false)
  const [qaSystemPrompt, setQaSystemPrompt] = useState(DEFAULT_QA_PROMPT)
  const [qaShowPrompt, setQaShowPrompt] = useState(true)
  const qaEndRef = useRef<HTMLDivElement>(null)
  const qaMessagesRef = useRef(qaMessages)
  qaMessagesRef.current = qaMessages
  const [selectedField, setSelectedField] = useState<string | null>(null)
  const [jsonViewOpen, setJsonViewOpen] = useState(false)
  const [compareOpen, setCompareOpen] = useState(false)
  const [exploreTab, setExploreTab] = useState<'fields' | 'line_items' | 'validation' | 'raw'>('fields')

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
      const res = await fetch(`/api/qa/${result.session_id}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })
      if (!res.ok) {
        const errBody = await res.json().catch(() => ({}))
        throw new Error(errBody.detail || `QA request failed (${res.status})`)
      }
      const data = await res.json()
      const answer = typeof data.answer === 'string' ? data.answer : JSON.stringify(data.answer)
      setQaMessages(prev => [...prev, { role: 'assistant', content: answer, evidence: data.evidence }])
    } catch (e) {
      const msg = e instanceof Error ? e.message : 'Unknown error'
      setQaMessages(prev => [...prev, { role: 'assistant', content: `Sorry — ${msg}` }])
    }
    setQaLoading(false)
  }, [qaInput, qaLoading, qaSystemPrompt, result.session_id])

  useEffect(() => {
    qaEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [qaMessages])

  return (
    <div className="h-full flex relative">
      <div className={`h-full bg-bg-surface border-r border-border flex flex-col transition-all duration-200 ${sidebarOpen ? 'w-80' : 'w-0 overflow-hidden'}`}>
        <div className="flex items-center justify-between px-4 py-3 border-b border-border shrink-0">
          <div className="flex items-center gap-2">
            <div className="text-xs font-semibold text-text-muted uppercase tracking-wider">Results</div>
            {result.total_time != null && (
              <span className="text-xs text-text-muted tabular-nums flex items-center gap-1">
                <HugeiconsIcon icon={Clock01Icon} className="size-2.5" />{fmtTime(result.total_time)}
              </span>
            )}
          </div>
          <div className="flex items-center gap-1">
            <Button variant="ghost" size="sm" onClick={() => setCompareOpen(true)}
              className="text-xs text-text-muted hover:text-text-primary">
              <HugeiconsIcon icon={BarChartIcon} className="size-3 inline mr-1 -mt-0.5" />Compare
            </Button>
            <Button variant="ghost" size="sm" onClick={() => setJsonViewOpen(true)}
              className="text-xs text-text-muted hover:text-text-primary">
              {'{ }'} JSON
            </Button>
            <Button variant="ghost" size="icon-sm" onClick={() => setSidebarOpen(false)}
            className="text-text-muted hover:text-text-primary">
            <HugeiconsIcon icon={ChevronLeftIcon} className="size-4" />
          </Button>
          </div>
        </div>
        <div className="flex-1 overflow-y-auto px-3 py-3 space-y-4">
          {preprocSteps.length > 0 && (
            <div>
              <div className="text-xs font-semibold text-text-muted uppercase tracking-wider mb-1.5">Preprocessing</div>
              <div className="space-y-1">
                {preprocSteps.map(name => {
                    const isSel = selectedStep === name
                    return (
                      <Button key={name}
                        variant={isSel ? 'outline' : 'ghost'}
                        onClick={() => setSelectedStep(isSel ? '' : name)}
                        className={`w-full flex items-center gap-1.5 px-2.5 py-2 text-left relative ${
                          isSel
                            ? 'bg-accent-violet/12 ring-1 ring-accent-violet/40'
                            : 'bg-bg-surface/60 hover:bg-bg-elevated/60 border border-border/30'
                        }`}>
                        <div className={`flex items-center justify-center w-5 h-5 rounded-full text-xs font-bold shrink-0 ${
                          isSel ? 'bg-accent-violet/20 text-accent-violet' : 'bg-bg-elevated text-text-muted'
                        }`}>{STEP_ORDER.indexOf(name) + 1}</div>
                        <span className={`text-xs font-medium truncate flex-1 ${
                          isSel ? 'text-accent-violet' : 'text-text-primary'
                        }`}>{STEP_LABELS[name]}</span>
                        <Button variant="ghost" size="icon-xs" onClick={e => { e.stopPropagation(); const rect = (e.currentTarget as HTMLElement).getBoundingClientRect(); setInfoTarget(infoTarget?.step === name ? null : { step: name, top: rect.bottom, left: rect.left, width: Math.max(rect.width, 280) }) }}
                          className="text-text-muted hover:text-text-primary shrink-0">
                          <HugeiconsIcon icon={InformationCircleIcon} className="size-3" />
                        </Button>
                        {result.timing?.[name] != null && (
                          <span className="text-xs text-text-muted tabular-nums shrink-0">{fmtTime(result.timing[name])}</span>
                        )}
                      </Button>
                    )
                  })}
                </div>
              </div>
          )}
          {STEP_GROUPS.slice(1).map(group => {
            const availableSteps = group.steps.filter(hasStep)
            if (availableSteps.length === 0) return null
            return (
              <div key={group.label}>
                <div className="text-xs font-semibold text-text-muted uppercase tracking-wider mb-1.5">{group.label}</div>
                <div className="space-y-1">
                  {availableSteps.map(name => {
                    const isSel = selectedStep === name
                    return (
                      <Button key={name}
                        variant={isSel ? 'outline' : 'ghost'}
                        onClick={() => setSelectedStep(isSel ? '' : name)}
                        className={`w-full flex items-center gap-1.5 px-2.5 py-2 text-left relative ${
                          isSel
                            ? 'bg-accent-violet/12 ring-1 ring-accent-violet/40'
                            : 'bg-bg-surface/60 hover:bg-bg-elevated/60 border border-border/30'
                        }`}>
                        <div className={`flex items-center justify-center w-5 h-5 rounded-full text-xs font-bold shrink-0 ${
                          isSel ? 'bg-accent-violet/20 text-accent-violet' : 'bg-bg-elevated text-text-muted'
                        }`}>{STEP_ORDER.indexOf(name) + 1}</div>
                        <span className={`text-xs font-medium truncate flex-1 ${
                          isSel ? 'text-accent-violet' : 'text-text-primary'
                        }`}>{STEP_LABELS[name]}</span>
                        <Button variant="ghost" size="icon-xs" onClick={e => { e.stopPropagation(); const rect = (e.currentTarget as HTMLElement).getBoundingClientRect(); setInfoTarget(infoTarget?.step === name ? null : { step: name, top: rect.bottom, left: rect.left, width: Math.max(rect.width, 280) }) }}
                          className="text-text-muted hover:text-text-primary shrink-0">
                          <HugeiconsIcon icon={InformationCircleIcon} className="size-3" />
                        </Button>
                        {result.timing?.[name] != null && (
                          <span className="text-xs text-text-muted tabular-nums shrink-0">{fmtTime(result.timing[name])}</span>
                        )}
                      </Button>
                    )
                  })}
                </div>
              </div>
            )
          })}
        </div>
      </div>

      {!sidebarOpen && (
        <Button variant="ghost" size="icon-sm" onClick={() => setSidebarOpen(true)}
          className="absolute left-0 top-1/2 -translate-y-1/2 w-6 h-12 bg-bg-elevated/80 border border-border rounded-r-lg flex items-center justify-center text-text-muted hover:text-text-primary z-10">
          <HugeiconsIcon icon={ChevronRightIcon} className="size-4" />
        </Button>
      )}

      <div className="flex-1 overflow-y-auto">
        <div className="flex min-h-full">
          <div className="w-2/5 shrink-0 bg-bg-surface/50 flex flex-col border-r border-border/50">
            <PDFViewer
              file={`/api/session/${result.session_id}/pdf`}
              className="flex-1 min-h-0"
              showToolbar
              showDownload
              showRotateControls
              renderPageOverlay={({ pageNumber }) => {
                const page = result.pages.find(p => p.page_number === pageNumber)
                if (!page) return null
                const iw = page.image_width || 1
                const ih = page.image_height || 1
                return (
                  <>
                    {page.ocr_boxes?.map((b, i) => (
                      <div key={`ocr-${i}`}
                        className="absolute border border-blue-400/60 bg-blue-400/15 pointer-events-none"
                        style={{
                          left: `${(b.box[0] / iw) * 100}%`,
                          top: `${(b.box[1] / ih) * 100}%`,
                          width: `${((b.box[2] - b.box[0]) / iw) * 100}%`,
                          height: `${((b.box[3] - b.box[1]) / ih) * 100}%`,
                        }}
                      />
                    ))}
                    {page.predicted_annotations?.map((a, i) => (
                      <div key={`pred-${i}`}
                        className={`absolute border-2 pointer-events-auto cursor-pointer ${selectedField === a.label ? 'border-yellow-400 bg-yellow-400/20' : ''}`}
                        style={{
                          left: `${(a.box[0] / iw) * 100}%`,
                          top: `${(a.box[1] / ih) * 100}%`,
                          width: `${((a.box[2] - a.box[0]) / iw) * 100}%`,
                          height: `${((a.box[3] - a.box[1]) / ih) * 100}%`,
                          borderColor: a.color || '#a29bfe',
                          background: a.color ? `${a.color}33` : '#a29bfe33',
                        }}
                        onClick={() => setSelectedField(a.label)}
                        title={`${a.label}: ${a.text}`}
                      />
                    ))}
                    {page.ground_truth_annotations?.map((a, i) => (
                      <div key={`gt-${i}`}
                        className="absolute border-2 border-dashed pointer-events-none"
                        style={{
                          left: `${(a.box[0] / iw) * 100}%`,
                          top: `${(a.box[1] / ih) * 100}%`,
                          width: `${((a.box[2] - a.box[0]) / iw) * 100}%`,
                          height: `${((a.box[3] - a.box[1]) / ih) * 100}%`,
                          borderColor: a.color || '#2ed573',
                          background: a.color ? `${a.color}44` : '#2ed57344',
                        }}
                      />
                    ))}
                    {(() => {
                      if (!selectedField) return null
                      const hlBoxes = (page.predicted_annotations || []).filter(a => a.label === selectedField)
                      const fallback = hlBoxes.length > 0 ? hlBoxes : findFieldInOcr(selectedField, page.extracted_fields, page.ocr_boxes)
                      return fallback.map((a, i) => (
                        <div key={`hl-${i}`}
                          className="absolute border-2 border-yellow-400 animate-pulse pointer-events-none"
                          style={{
                            left: `${(a.box[0] / iw) * 100 - 0.5}%`,
                            top: `${(a.box[1] / ih) * 100 - 0.5}%`,
                            width: `${((a.box[2] - a.box[0]) / iw) * 100 + 1}%`,
                            height: `${((a.box[3] - a.box[1]) / ih) * 100 + 1}%`,
                          }}
                        />
                      ))
                    })()}
                  </>
                )
              }}
            />
          </div>

          <div className="w-3/5 shrink-0 p-4 bg-bg-base overflow-y-auto">
            {result.errors.length > 0 && (
              <div className="mb-4 flex items-center gap-2 px-4 py-2.5 bg-accent-coral/15 border border-accent-coral/30 rounded-xl text-sm text-accent-coral animate-fade-in">
                <HugeiconsIcon icon={AlertCircleIcon} className="size-4 shrink-0 text-accent-coral" />
                <span className="font-medium">{result.errors.length} error{result.errors.length !== 1 ? 's' : ''}</span>
                {result.errors.map((e, i) => <span key={i} className="text-accent-coral ml-1">{e}</span>)}
              </div>
            )}

            {result.num_pages > 1 && (
              <div className="flex items-center gap-2 mb-4">
                <Button variant="ghost" size="icon-sm" onClick={() => setActivePage(Math.max(0, activePage - 1))} disabled={activePage === 0}
                  className="text-text-muted hover:text-text-primary disabled:opacity-30">
                  <HugeiconsIcon icon={ChevronLeftIcon} className="size-4" />
                </Button>
                <span className="text-xs text-text-muted tabular-nums">Page {activePage + 1} / {result.num_pages}</span>
                <Button variant="ghost" size="icon-sm" onClick={() => setActivePage(Math.min(result.num_pages - 1, activePage + 1))} disabled={activePage >= result.num_pages - 1}
                  className="text-text-muted hover:text-text-primary disabled:opacity-30">
                  <HugeiconsIcon icon={ChevronRightIcon} className="size-4" />
                </Button>
              </div>
            )}

            {(() => {
              const page = result.pages[activePage]
              const fields = page?.extracted_fields || {}
              const validation = page?.validation
              const displayFields = Object.fromEntries(
                Object.entries(fields).filter(([k]) => !k.startsWith('LINE/') && k !== 'line_items')
              )
              const hasFields = Object.keys(displayFields).length > 0

              return (
                <>
                  <div className="flex items-center gap-1 mb-4 border-b border-border">
                    {(['fields', 'line_items', 'validation', 'raw'] as const).map(tab => (
                      <Button key={tab} variant={exploreTab === tab ? 'ghost' : 'ghost'} size="sm" onClick={() => setExploreTab(tab)}
                        className={`text-xs font-semibold rounded-t-lg ${
                          exploreTab === tab
                            ? 'text-accent-violet border-b-2 border-accent-violet bg-accent-violet/5'
                            : 'text-text-muted hover:text-text-primary'
                        }`}>
                        {tab === 'fields' ? 'Fields' : tab === 'line_items' ? 'Line Items' : tab === 'validation' ? 'Validation' : 'Raw JSON'}
                      </Button>
                    ))}
                    <div className="flex-1" />
                    <Button variant="default" size="sm" onClick={() => { const blob = new Blob([JSON.stringify(result, null, 2)], {type: 'application/json'}); const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = `result-${result.session_id}.json`; a.click() }}
                      className="text-xs font-semibold text-text-primary bg-accent-violet hover:bg-accent-violet/80">
                      Export JSON
                    </Button>
                  </div>

                  {exploreTab === 'fields' && (
                    <div className="space-y-2">
                      {!hasFields ? (
                        <div className="text-sm text-text-muted italic p-4 text-center">No fields extracted for this page</div>
                      ) : (
                        <div className="overflow-x-auto">
                          <table className="w-full text-sm">
                            <thead>
                              <tr className="border-b border-border">
                                <th className="text-left py-2 px-3 text-xs font-semibold text-text-muted uppercase tracking-wider">Field</th>
                                <th className="text-left py-2 px-3 text-xs font-semibold text-text-muted uppercase tracking-wider">Value</th>
                                <th className="text-left py-2 px-3 text-xs font-semibold text-text-muted uppercase tracking-wider">Confidence</th>
                                <th className="text-left py-2 px-3 text-xs font-semibold text-text-muted uppercase tracking-wider">Evidence</th>
                                <th className="text-left py-2 px-3 text-xs font-semibold text-text-muted uppercase tracking-wider">Page</th>
                              </tr>
                            </thead>
                            <tbody>
                              {Object.entries(displayFields).map(([field, value]) => {
                                const evidence = page.extraction_evidence?.[field]
                                const conf = 0.9
                                const confColor = conf >= 0.85 ? 'bg-accent-green' : conf >= 0.6 ? 'bg-accent-yellow' : 'bg-accent-coral'
                                return (
                                  <tr key={field} className="border-b border-border/50 hover:bg-bg-elevated/30 transition-colors">
                                    <td className="py-2.5 px-3">
                                      <Button variant="link" size="sm" onClick={() => setSelectedField(field)}
                                        className="text-xs font-semibold text-accent-violet">
                                        {field.replace(/_/g, ' ')}
                                      </Button>
                                    </td>
                                    <td className="py-2.5 px-3">
                                      <span className="text-xs text-text-primary font-mono">{typeof value === 'object' ? JSON.stringify(value) : String(value)}</span>
                                    </td>
                                    <td className="py-2.5 px-3">
                                      <div className="flex items-center gap-2">
                                        <div className="w-16 h-1.5 bg-border rounded-full overflow-hidden">
                                          <div className={`h-full rounded-full ${confColor}`} style={{width: `${conf * 100}%`}} />
                                        </div>
                                        <span className="text-xs text-text-muted font-mono">{Math.round(conf * 100)}%</span>
                                      </div>
                                    </td>
                                    <td className="py-2.5 px-3">
                                      <span className="text-xs text-text-muted font-mono max-w-[120px] truncate inline-block">{evidence || '—'}</span>
                                    </td>
                                    <td className="py-2.5 px-3">
                                      <span className="text-xs text-text-muted">{activePage + 1}</span>
                                    </td>
                                  </tr>
                                )
                              })}
                            </tbody>
                          </table>
                        </div>
                      )}
                    </div>
                  )}

                  {exploreTab === 'validation' && (
                    <div className="space-y-4">
                      {!validation && !page?.field_confidence ? (
                        <div className="text-sm text-text-muted italic p-4 text-center">No validation data available</div>
                      ) : (
                        <>
                          {/* HITL Routing: fields needing human review */}
                          {page?.field_confidence && (
                            <div className="bg-bg-surface border border-border rounded-xl overflow-hidden">
                              <div className="flex items-center justify-between px-4 py-2.5 border-b border-border">
                                <span className="text-xs font-semibold text-text-muted uppercase tracking-wider">HITL Routing</span>
                                {page?.overall_confidence != null && (
                                  <span className={`text-xs font-semibold tabular-nums ${
                                    page.overall_confidence >= 0.85 ? 'text-accent-green' : page.overall_confidence >= 0.7 ? 'text-accent-yellow' : 'text-accent-coral'
                                  }`}>
                                    {Math.round(page.overall_confidence * 100)}%
                                  </span>
                                )}
                              </div>
                              <div className="divide-y divide-border">
                                {Object.entries(page.field_confidence).filter(([k]) => !k.startsWith('_')).map(([k, v]) => (
                                  <div key={k} className="flex items-center justify-between px-4 py-2 hover:bg-bg-elevated/30 transition-colors">
                                    <div className="flex items-center gap-2 min-w-0">
                                      <div className={`w-1.5 h-1.5 rounded-full shrink-0 ${
                                        v.needs_review ? 'bg-accent-coral' :
                                        v.level === 'high' ? 'bg-accent-green' :
                                        v.level === 'medium' ? 'bg-accent-yellow' : 'bg-accent-coral'
                                      }`} />
                                      <span className="text-xs text-text-primary truncate">{k.replace(/_/g, ' ')}</span>
                                    </div>
                                    <div className="flex items-center gap-2 shrink-0">
                                      {v.needs_review && (
                                        <span className="text-[10px] px-1.5 py-0.5 rounded-full bg-accent-coral/10 text-accent-coral font-medium">Review</span>
                                      )}
                                      <span className={`text-xs font-mono tabular-nums ${
                                        v.confidence >= 0.85 ? 'text-accent-green' : v.confidence >= 0.7 ? 'text-accent-yellow' : 'text-accent-coral'
                                      }`}>
                                        {Math.round(v.confidence * 100)}%
                                      </span>
                                    </div>
                                  </div>
                                ))}
                              </div>
                            </div>
                          )}

                          {/* Validation issues */}
                          {validation && (
                            <div className="bg-bg-surface border border-border rounded-xl overflow-hidden">
                              <div className={`inline-flex items-center gap-1.5 px-3 py-1.5 mt-3 ml-4 rounded-xl text-xs font-semibold ${
                                validation.is_valid ? 'bg-accent-green/15 text-accent-green' : 'bg-accent-coral/15 text-accent-coral'
                              }`}>
                                {validation.is_valid ? 'All checks passed' : `${(validation.issues || []).length} issues found`}
                              </div>
                              <div className="p-4 space-y-1">
                                {(validation.issues || []).map((issue, i) => (
                                  <div key={i} className="flex items-start gap-2 px-3 py-2 bg-bg-surface/50 rounded-lg border border-border">
                                    {issue.severity === 'error' ? (
                                      <HugeiconsIcon icon={AlertCircleIcon} className="size-4 text-accent-coral mt-0.5 shrink-0" />
                                    ) : (
                                      <HugeiconsIcon icon={AlertCircleIcon} className="size-4 text-accent-yellow mt-0.5 shrink-0" />
                                    )}
                                    <div className="flex-1 min-w-0">
                                      <div className="text-xs text-text-primary">{issue.message}</div>
                                      <div className="text-xs text-text-muted font-mono mt-0.5">{issue.rule}</div>
                                    </div>
                                  </div>
                                ))}
                              </div>
                            </div>
                          )}
                        </>
                      )}
                    </div>
                  )}

                  {exploreTab === 'line_items' && (
                    <div>
                      {!page?.line_items?.length ? (
                        <div className="text-sm text-text-muted italic p-4 text-center">No line items extracted for this page</div>
                      ) : (
                        <div className="overflow-x-auto">
                          <table className="w-full text-sm">
                            <thead>
                              <tr className="border-b border-border">
                                <th className="text-left py-2 px-3 text-xs font-semibold text-text-muted uppercase tracking-wider">#</th>
                                <th className="text-left py-2 px-3 text-xs font-semibold text-text-muted uppercase tracking-wider">Description</th>
                                <th className="text-right py-2 px-3 text-xs font-semibold text-text-muted uppercase tracking-wider">Qty</th>
                                <th className="text-right py-2 px-3 text-xs font-semibold text-text-muted uppercase tracking-wider">Unit Price</th>
                                <th className="text-right py-2 px-3 text-xs font-semibold text-text-muted uppercase tracking-wider">VAT</th>
                                <th className="text-right py-2 px-3 text-xs font-semibold text-text-muted uppercase tracking-wider">Total</th>
                              </tr>
                            </thead>
                            <tbody>
                              {page.line_items.map((item: LineItem, i: number) => (
                                <tr key={i} className="border-b border-border/50 hover:bg-bg-elevated/30 transition-colors">
                                  <td className="py-2 px-3 text-xs text-text-muted">{i + 1}</td>
                                  <td className="py-2 px-3 text-xs text-text-primary font-medium">{item.description || '—'}</td>
                                  <td className="py-2 px-3 text-xs text-text-muted text-right font-mono">{item.quantity || '—'}</td>
                                  <td className="py-2 px-3 text-xs text-text-muted text-right font-mono">{item.unit_price ? `${Number(item.unit_price).toFixed(2)}` : '—'}</td>
                                  <td className="py-2 px-3 text-xs text-text-muted text-right font-mono">{item.vat_rate || '—'}</td>
                                  <td className="py-2 px-3 text-xs text-text-primary text-right font-mono font-semibold">{item.total ? `${Number(item.total).toFixed(2)}` : '—'}</td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                          <div className="text-xs text-text-muted mt-2 px-3 py-2 bg-bg-elevated/30 rounded-lg">
                            {page.line_items.length} line item{page.line_items.length !== 1 ? 's' : ''} on page {activePage + 1}
                          </div>
                        </div>
                      )}
                    </div>
                  )}

                  {exploreTab === 'raw' && (
                    <div>
                      {!page ? (
                        <div className="text-sm text-text-muted italic p-4 text-center">No data available</div>
                      ) : (
                        <div className="bg-bg-surface border border-border rounded-xl overflow-hidden">
                          <div className="flex items-center justify-between px-4 py-2 border-b border-border">
                            <span className="text-xs font-semibold text-text-muted uppercase tracking-wider">Page {activePage + 1} Raw Fields</span>
                          </div>
                          <pre className="text-xs text-text-muted font-mono leading-relaxed whitespace-pre-wrap p-4 max-h-96 overflow-y-auto"
                            style={{ fontFamily: "'JetBrains Mono', monospace" }}>
                            {JSON.stringify({
                              page_number: page.page_number,
                              extracted_fields: page.extracted_fields,
                              ocr_word_count: page.ocr_word_count,
                            }, null, 2)}
                          </pre>
                        </div>
                      )}
                    </div>
                  )}
                </>
              )
            })()}
          </div>
        </div>
      </div>

      {qaOpen && (
        <div className="fixed bottom-4 right-4 w-[36rem] h-[42rem] bg-bg-surface border border-border rounded-2xl shadow-2xl flex flex-col overflow-hidden z-50 animate-scale-in">
          <div className="flex items-center justify-between px-4 py-3 border-b border-border shrink-0">
            <span className="text-sm font-semibold text-text-primary">Ask about this document</span>
            <div className="flex items-center gap-1">
              <Button variant="ghost" size="sm" onClick={() => setQaShowPrompt(!qaShowPrompt)}
                className="text-xs text-text-muted hover:text-text-primary">
                {qaShowPrompt ? 'Hide' : 'Prompt'}
              </Button>
              <Button variant="ghost" size="icon-sm" onClick={() => setQaOpen(false)}
                className="text-text-muted hover:text-text-primary">
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
                  <QAMessage content={msg.content} evidence={msg.evidence} onFieldClick={setSelectedField} />
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
              <Button variant="default" size="icon-sm" type="submit" disabled={qaLoading || !qaInput.trim()}
                className="bg-accent-violet hover:bg-accent-violet/80 text-white disabled:opacity-30">
                <HugeiconsIcon icon={MailSend01Icon} className="size-4" />
              </Button>
            </form>
          </div>
        </div>
      )}

      <Button variant="ghost" size="icon-lg" onClick={() => setQaOpen(!qaOpen)}
        className={`fixed bottom-4 right-4 w-12 h-12 rounded-full shadow-lg z-40 ${
          qaOpen ? 'bg-bg-elevated text-text-muted' : 'bg-accent-violet text-white hover:bg-accent-violet/80 shadow-accent-violet/30'
        }`}>
        {qaOpen ? <HugeiconsIcon icon={Cancel01Icon} className="size-5" /> : <HugeiconsIcon icon={BubbleChatIcon} className="size-5" />}
      </Button>

      {jsonViewOpen && (
        <JsonViewer page={result.pages[activePage]} result={result} onClose={() => setJsonViewOpen(false)} />
      )}
      {compareOpen && (
        <CompareView sessionId={result.session_id} onClose={() => setCompareOpen(false)} />
      )}
      {infoTarget && createPortal(
        <div style={{ position: 'fixed', top: infoTarget.top + 4, left: infoTarget.left, zIndex: 9999 }}>
          <div style={{ width: infoTarget.width }}>
            <StepInfoTooltip step={infoTarget.step} onClose={() => setInfoTarget(null)} />
          </div>
        </div>,
        document.body
      )}
    </div>
  )
}
