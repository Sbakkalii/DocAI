import { useState, useEffect } from 'react'
import { HugeiconsIcon } from "@hugeicons/react"
import { File01Icon, Loading01Icon, AiLockIcon, AiIdeaIcon, Analytics01Icon, Edit01Icon, AiBrain01Icon, AiChipIcon, AiMagicIcon, AiScanIcon, BadgeCheckIcon } from "@hugeicons/core-free-icons"
import { Button } from '@/components/ui/button'
import { FileUpload } from '@/components/ui/file-upload'
import { DEFAULT_FIELDS } from './constants'

function StepPill({ label, icon, color }: { label: string; icon: typeof AiBrain01Icon; color: string }) {
  return (
    <div className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg border text-xs font-medium transition-all hover:scale-105 ${color}`}>
      <HugeiconsIcon icon={icon} className="size-3.5" />
      {label}
    </div>
  )
}

export function UploadShell({ onStart }: {
  onStart: (sid: string, name: string) => void
}) {
  const [uploading, setUploading] = useState(false)
  const [ollamaModels, setOllamaModels] = useState<Array<{name: string; size_gb: number; parameter_size: string}>>([])

  const [dsDocs, setDsDocs] = useState<Array<Record<string, unknown>>>([])
  const [dsPage, setDsPage] = useState(1)
  const [dsTotal, setDsTotal] = useState(0)
  const uploadTargetFields = DEFAULT_FIELDS

  useEffect(() => {
    fetch('/api/ollama/models')
      .then(r => r.json())
      .then(d => { if (d.models?.length) setOllamaModels(d.models) })
      .catch(() => {})
  }, [])

  useEffect(() => {
    fetch(`/api/dataset/documents?page=${dsPage}&per_page=18&per_model=1`)
      .then(r => r.json())
      .then(d => { setDsDocs(d.documents || []); setDsTotal(d.total_pages || 1) })
      .catch(() => {})
  }, [dsPage])

  async function handleFile(file: File) {
    setUploading(true)
    try {
      const form = new FormData()
      form.append('file', file)
      if (uploadTargetFields.length > 0) form.append('target_fields', uploadTargetFields.join(','))
      const res = await fetch('/api/upload', { method: 'POST', body: form })
      if (!res.ok) throw new Error((await res.json()).detail || 'Upload failed')
      const data = await res.json()
      onStart(data.session_id, file.name)
    } catch (e: unknown) {
      alert(e instanceof Error ? e.message : 'Upload failed')
    } finally {
      setUploading(false)
    }
  }

  async function handleDatasetDoc(doc: Record<string, unknown>) {
    setUploading(true)
    try {
      const modelName = (doc.model as string || '').replace('invoice_dataset_', '')
      let targetFields = ''
      try {
        const fieldsRes = await fetch(`/api/dataset/model-fields/${modelName}`)
        if (fieldsRes.ok) {
          const fieldsData = await fieldsRes.json()
          targetFields = Object.keys(fieldsData.fields || {}).join(',')
        }
      } catch {}
      const form = new FormData()
      form.append('path', doc.image_path as string)
      form.append('filename', doc.filename as string)
      if (targetFields) form.append('target_fields', targetFields)
      form.append('mode', 'end_to_end')
      const res = await fetch('/api/dataset/load', { method: 'POST', body: form })
      if (!res.ok) throw new Error('Failed to load document')
      const data = await res.json()
      onStart(data.session_id, doc.filename as string)
    } catch (e: unknown) {
      alert(e instanceof Error ? e.message : 'Failed to load document')
    } finally {
      setUploading(false)
    }
  }

  return (
    <div className="h-full flex items-start justify-center overflow-y-auto p-6 pt-8">
      <div className="w-full max-w-5xl mx-auto space-y-8">

        {/* ── Hero ── */}
        <div className="text-center">
          <div className="relative inline-flex mb-5">
            <div className="w-16 h-16 rounded-2xl flex items-center justify-center bg-gradient-to-br from-accent-violet via-purple-500 to-accent-violet-dark text-white shadow-xl shadow-accent-violet/25 animate-in zoom-in-95 duration-300">
              <HugeiconsIcon icon={AiBrain01Icon} className="size-8" />
            </div>
            <div className="absolute -bottom-1 -right-1 w-6 h-6 rounded-full bg-accent-green/20 border-2 border-bg-base flex items-center justify-center">
              <HugeiconsIcon icon={BadgeCheckIcon} className="size-3.5 text-accent-green" />
            </div>
          </div>
          <h1 className="text-2xl font-bold text-text-primary mb-2 bg-gradient-to-r from-accent-violet via-purple-400 to-accent-violet bg-clip-text text-transparent">
            Agentic Document Intelligence
          </h1>
          <p className="text-sm text-text-muted max-w-lg mx-auto leading-relaxed">
            Local-first VLM-powered document extraction pipeline — no cloud, no API keys.
            Upload a document and let AI extract, validate, and export your data.
          </p>

          {/* ── Quick Stats ── */}
          <div className="flex items-center justify-center gap-4 mt-4 flex-wrap">
            <div className="flex items-center gap-1.5 text-xs text-text-muted/70 bg-bg-elevated/30 px-3 py-1.5 rounded-full border border-border/30">
              <HugeiconsIcon icon={AiChipIcon} className="size-3.5 text-accent-violet" />
              <span>{ollamaModels.length} models available</span>
            </div>
            <div className="flex items-center gap-1.5 text-xs text-text-muted/70 bg-bg-elevated/30 px-3 py-1.5 rounded-full border border-border/30">
              <HugeiconsIcon icon={AiScanIcon} className="size-3.5 text-blue-400" />
              <span>4 extraction modes</span>
            </div>
            <div className="flex items-center gap-1.5 text-xs text-text-muted/70 bg-bg-elevated/30 px-3 py-1.5 rounded-full border border-border/30">
              <HugeiconsIcon icon={BadgeCheckIcon} className="size-3.5 text-emerald-400" />
              <span>6 document types</span>
            </div>
          </div>
        </div>

        {/* ── Pipeline Flow ── */}
        <div className="bg-gradient-to-br from-bg-surface/80 to-bg-surface/40 border border-border/40 rounded-xl p-5 backdrop-blur-sm">
          <div className="text-xs font-semibold text-text-muted uppercase tracking-wider mb-4 flex items-center gap-2">
            <HugeiconsIcon icon={AiMagicIcon} className="size-3.5 text-accent-violet" />
            Pipeline Architecture
          </div>
          <div className="flex items-center justify-center gap-1.5 flex-wrap">
            <StepPill label="Upload" icon={File01Icon} color="bg-bg-elevated/60 text-text-muted border-border/40" />
            <span className="text-text-muted/30 text-xs">→</span>
            <StepPill label="VLM Extract" icon={AiBrain01Icon} color="bg-accent-violet/15 text-accent-violet border-accent-violet/30" />
            <span className="text-text-muted/30 text-xs">→</span>
            <StepPill label="Classify" icon={AiScanIcon} color="bg-blue-500/10 text-blue-400 border-blue-500/25" />
            <span className="text-text-muted/30 text-xs">→</span>
            <StepPill label="Validate" icon={BadgeCheckIcon} color="bg-teal-500/10 text-teal-400 border-teal-500/25" />
            <span className="text-text-muted/30 text-xs">→</span>
            <StepPill label="Score" icon={Analytics01Icon} color="bg-emerald-500/10 text-emerald-400 border-emerald-500/25" />
            <span className="text-text-muted/30 text-xs">→</span>
            <StepPill label="Review" icon={Edit01Icon} color="bg-amber-500/10 text-amber-400 border-amber-500/25" />
            <span className="text-text-muted/30 text-xs">→</span>
            <StepPill label="Export" icon={File01Icon} color="bg-bg-elevated/60 text-text-muted border-border/40" />
          </div>
          <div className="flex items-center justify-center gap-1.5 mt-2">
            <span className="text-[10px] text-text-muted/40 w-[76px] text-right">optional ↓</span>
            <span className="text-text-muted/15 text-xs">╰</span>
            <StepPill label="OCR + RAG" icon={AiChipIcon} color="bg-bg-elevated/30 text-text-muted/50 border-border/20" />
          </div>
        </div>

        {/* ── Feature Cards ── */}
        <div className="grid grid-cols-4 gap-3">
          {[
            { icon: AiLockIcon, color: 'from-accent-violet/20 to-accent-violet/5', iconBg: 'bg-accent-violet/15', iconColor: 'text-accent-violet', title: 'Private & Local', desc: 'All runs on your machine with open-source LLMs. No data ever leaves your computer.' },
            { icon: AiIdeaIcon, color: 'from-blue-500/20 to-blue-500/5', iconBg: 'bg-blue-500/15', iconColor: 'text-blue-400', title: 'VLM-First', desc: 'Vision-language models extract fields directly from images in a single pass.' },
            { icon: Analytics01Icon, color: 'from-emerald-500/20 to-emerald-500/5', iconBg: 'bg-emerald-500/15', iconColor: 'text-emerald-400', title: 'Multi-Task NLP', desc: 'NER, summarization, contract analysis, risk scoring — all in one pipeline.' },
            { icon: Edit01Icon, color: 'from-amber-500/20 to-amber-500/5', iconBg: 'bg-amber-500/15', iconColor: 'text-amber-400', title: 'Human Review', desc: 'Review, correct, and confirm each field before exporting results.' },
          ].map((card, i) => (
            <div key={i}
              className="group relative bg-bg-surface/40 border border-border/30 rounded-xl p-4 space-y-2.5 transition-all duration-200 hover:bg-bg-surface/60 hover:border-border/60 hover:shadow-sm hover:-translate-y-0.5">
              <div className={`w-9 h-9 rounded-lg ${card.iconBg} flex items-center justify-center transition-transform duration-200 group-hover:scale-110`}>
                <HugeiconsIcon icon={card.icon} className={`size-4.5 ${card.iconColor}`} />
              </div>
              <div className="text-sm font-semibold text-text-primary">{card.title}</div>
              <div className="text-xs text-text-muted leading-relaxed">{card.desc}</div>
            </div>
          ))}
        </div>

        {/* ── Upload & Dataset ── */}
        <div className="grid grid-cols-2 gap-6">
          <div className="flex flex-col gap-4">
            <h2 className="text-sm font-semibold text-text-primary flex items-center gap-2">
              <div className="w-6 h-6 rounded-lg bg-accent-violet/15 flex items-center justify-center">
                <HugeiconsIcon icon={File01Icon} className="size-3.5 text-accent-violet" />
              </div>
              Upload from Computer
            </h2>
            {uploading ? (
              <div className="flex flex-col items-center justify-center gap-3 min-h-[220px] border-2 border-dashed rounded-xl bg-bg-surface">
                <HugeiconsIcon icon={Loading01Icon} className="size-8 animate-spin text-accent-violet" />
                <p className="text-sm text-text-muted">Uploading document...</p>
              </div>
            ) : (
              <FileUpload
                accept=".jpg,.jpeg,.png,.tiff,.pdf"
                title="Click to upload or drop a document"
                description="JPG, PNG, TIFF, PDF — up to 100 MB"
                multiple={false}
                showFileList={false}
                showBorderBeam={true}
                onFilesAccepted={(files) => { if (files[0]) handleFile(files[0]) }}
                className="min-h-[220px]"
              />
            )}
          </div>

          <div className="flex flex-col gap-4">
            <h2 className="text-sm font-semibold text-text-primary flex items-center gap-2">
              <div className="w-6 h-6 rounded-lg bg-accent-violet/15 flex items-center justify-center">
                <HugeiconsIcon icon={AiScanIcon} className="size-3.5 text-accent-violet" />
              </div>
              Browse Dataset
            </h2>
            <div className="bg-bg-surface border border-border rounded-xl p-3 min-h-[220px]">
              {dsDocs.length === 0 ? (
                <div className="flex items-center justify-center h-[220px] text-xs text-text-muted">
                  Loading documents...
                </div>
              ) : (
                <div className="grid grid-cols-3 gap-2">
                  {dsDocs.map(doc => (
                    <button key={doc.id as string} onClick={() => handleDatasetDoc(doc)}
                      className="group bg-bg-surface border border-border rounded-lg p-1.5 text-left hover:border-accent-violet hover:shadow-sm transition-all duration-200">
                      <div className="aspect-[3/4] bg-bg-base rounded mb-1 flex items-center justify-center overflow-hidden">
                        {doc.image_path ? (
                          <img src={`/api/image/${encodeURIComponent(doc.image_path as string)}`} alt=""
                            className="w-full h-full object-cover transition-transform duration-200 group-hover:scale-105" />
                        ) : (
                          <HugeiconsIcon icon={File01Icon} className="size-5 text-text-muted" />
                        )}
                      </div>
                      <div className="text-xs text-text-muted truncate leading-tight">{doc.filename as string}</div>
                      <div className="text-[13px] text-text-muted/60 leading-tight">{(doc.model as string).replace('invoice_dataset_', '')}</div>
                    </button>
                  ))}
                </div>
              )}
              {dsTotal > 1 && (
                <div className="flex items-center justify-center gap-2 mt-3 text-xs">
                  <Button variant="outline" size="sm" disabled={dsPage <= 1} onClick={() => setDsPage(p => p - 1)}>Prev</Button>
                  <span className="text-text-muted tabular-nums">{dsPage} / {dsTotal}</span>
                  <Button variant="outline" size="sm" disabled={dsPage >= dsTotal} onClick={() => setDsPage(p => p + 1)}>Next</Button>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* ── Available Models ── */}
        {ollamaModels.length > 0 && (
          <div className="bg-bg-surface/40 border border-border/30 rounded-xl p-4">
            <div className="text-xs font-semibold text-text-muted uppercase tracking-wider mb-3 flex items-center gap-2">
              <HugeiconsIcon icon={AiChipIcon} className="size-3.5 text-accent-violet" />
              Available Models
            </div>
            <div className="flex flex-wrap gap-2">
              {ollamaModels.map(m => (
                <div key={m.name}
                  className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-bg-elevated/40 border border-border/30 text-xs">
                  <span className="text-text-primary font-medium">{m.name}</span>
                  <span className="text-text-muted/60">{m.size_gb.toFixed(1)} GB</span>
                  {m.parameter_size && (
                    <span className="text-text-muted/40">{m.parameter_size}</span>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
