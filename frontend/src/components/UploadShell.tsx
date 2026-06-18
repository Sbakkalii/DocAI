import { useState, useEffect } from 'react'
import { HugeiconsIcon } from "@hugeicons/react"
import { File01Icon, Loading01Icon, AiLockIcon, AiIdeaIcon, Analytics01Icon, Edit01Icon } from "@hugeicons/core-free-icons"
import { Button } from '@/components/ui/button'
import { FileUpload } from '@/components/ui/file-upload'
import { DEFAULT_FIELDS } from './constants'

export function UploadShell({ onStart }: {
  onStart: (sid: string, name: string) => void
}) {
  const [uploading, setUploading] = useState(false)

  const [dsDocs, setDsDocs] = useState<Array<Record<string, unknown>>>([])
  const [dsPage, setDsPage] = useState(1)
  const [dsTotal, setDsTotal] = useState(0)
  const uploadTargetFields = DEFAULT_FIELDS

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

  function StepBox({ label, color, small }: { label: string; color: string; small?: boolean }) {
    return (
      <div className={`${small ? 'px-2 py-1 text-[10px]' : 'px-3 py-1.5 text-xs'} rounded-lg border font-medium ${color}`}>
        {label}
      </div>
    )
  }

  function Arrow() {
    return <span className="text-text-muted/40 text-xs">→</span>
  }

  return (
    <div className="h-full flex items-start justify-center overflow-y-auto p-6 pt-10">
      <div className="w-full max-w-5xl mx-auto space-y-6">
        {/* Hero */}
        <div className="text-center mb-2">
          <div className="w-14 h-14 rounded-2xl mx-auto mb-4 flex items-center justify-center bg-gradient-to-br from-accent-violet to-accent-violet-dark text-white shadow-lg shadow-accent-violet/20">
            <HugeiconsIcon icon={File01Icon} className="size-7" />
          </div>
          <h1 className="text-xl font-semibold text-text-primary mb-1">Agentic Document Intelligence</h1>
          <p className="text-sm text-text-muted">Local-first VLM-powered document extraction pipeline — no cloud, no API keys</p>
        </div>

        {/* Architecture Flow */}
        <div className="bg-bg-surface/60 border border-border/50 rounded-xl p-5">
          <div className="text-xs font-semibold text-text-muted mb-4">Pipeline Architecture</div>
          <div className="flex items-center justify-center gap-1.5 flex-wrap">
            <StepBox label="Upload" color="bg-bg-elevated/60 text-text-muted border-border/40" />
            <Arrow />
            <StepBox label="VLM" color="bg-accent-violet/15 text-accent-violet border-accent-violet/30" />
            <Arrow />
            <StepBox label="Classify" color="bg-blue-500/10 text-blue-400 border-blue-500/25" />
            <Arrow />
            <StepBox label="Validate" color="bg-teal-500/10 text-teal-400 border-teal-500/25" />
            <Arrow />
            <StepBox label="Score" color="bg-emerald-500/10 text-emerald-400 border-emerald-500/25" />
            <Arrow />
            <StepBox label="Review" color="bg-amber-500/10 text-amber-400 border-amber-500/25" />
            <Arrow />
            <StepBox label="Export" color="bg-bg-elevated/60 text-text-muted border-border/40" />
          </div>
          <div className="flex items-center justify-center gap-1.5 mt-1.5">
            <span className="text-[10px] text-text-muted/50 w-[72px] text-right">optional ↓</span>
            <span className="text-text-muted/20 text-xs">╰</span>
            <StepBox label="OCR + RAG" color="bg-bg-elevated/40 text-text-muted/60 border-border/30" small />
          </div>
        </div>

        {/* Feature Highlights */}
        <div className="grid grid-cols-4 gap-3">
          <div className="bg-bg-surface/40 border border-border/30 rounded-xl p-4 space-y-2">
            <div className="w-8 h-8 rounded-lg bg-accent-violet/15 flex items-center justify-center">
              <HugeiconsIcon icon={AiLockIcon} className="size-4 text-accent-violet" />
            </div>
            <div className="text-sm font-semibold text-text-primary">Private &amp; Local</div>
            <div className="text-xs text-text-muted leading-relaxed">All runs on your machine with open-source LLMs. No data ever leaves.</div>
          </div>
          <div className="bg-bg-surface/40 border border-border/30 rounded-xl p-4 space-y-2">
            <div className="w-8 h-8 rounded-lg bg-blue-500/15 flex items-center justify-center">
              <HugeiconsIcon icon={AiIdeaIcon} className="size-4 text-blue-400" />
            </div>
            <div className="text-sm font-semibold text-text-primary">VLM-First</div>
            <div className="text-xs text-text-muted leading-relaxed">Vision-language models extract fields directly from images in a single pass.</div>
          </div>
          <div className="bg-bg-surface/40 border border-border/30 rounded-xl p-4 space-y-2">
            <div className="w-8 h-8 rounded-lg bg-emerald-500/15 flex items-center justify-center">
              <HugeiconsIcon icon={Analytics01Icon} className="size-4 text-emerald-400" />
            </div>
            <div className="text-sm font-semibold text-text-primary">Multi-Task NLP</div>
            <div className="text-xs text-text-muted leading-relaxed">NER, summarization, contract analysis, risk scoring — all in one pipeline.</div>
          </div>
          <div className="bg-bg-surface/40 border border-border/30 rounded-xl p-4 space-y-2">
            <div className="w-8 h-8 rounded-lg bg-amber-500/15 flex items-center justify-center">
              <HugeiconsIcon icon={Edit01Icon} className="size-4 text-amber-400" />
            </div>
            <div className="text-sm font-semibold text-text-primary">Human Review</div>
            <div className="text-xs text-text-muted leading-relaxed">Review, correct, and confirm each field before exporting results.</div>
          </div>
        </div>

        {/* Upload & Dataset */}
        <div className="grid grid-cols-2 gap-6">
          <div className="flex flex-col gap-4">
            <h2 className="text-sm font-semibold text-text-primary flex items-center gap-2">
              <HugeiconsIcon icon={File01Icon} className="size-4 text-accent-violet" />
              Upload from Computer
            </h2>
            {uploading ? (
              <div className="flex flex-col items-center justify-center gap-3 min-h-[200px] border-2 border-dashed rounded-xl bg-bg-surface">
                <HugeiconsIcon icon={Loading01Icon} className="size-8 animate-spin text-accent-violet" />
                <p className="text-sm text-text-muted">Uploading document...</p>
              </div>
            ) : (
              <FileUpload
                accept=".jpg,.jpeg,.png,.tiff,.pdf"
                title="Click to upload or drop a document"
                description="JPG, PNG, TIFF, PDF"
                multiple={false}
                showFileList={false}
                showBorderBeam={true}
                onFilesAccepted={(files) => { if (files[0]) handleFile(files[0]) }}
                className="min-h-[200px]"
              />
            )}
          </div>

          <div className="flex flex-col gap-4">
            <h2 className="text-sm font-semibold text-text-primary flex items-center gap-2">
              <HugeiconsIcon icon={File01Icon} className="size-4 text-accent-violet" />
              Browse Dataset
            </h2>
            <div className="bg-bg-surface border border-border rounded-xl p-3 min-h-[200px]">
              {dsDocs.length === 0 ? (
                <div className="flex items-center justify-center h-[200px] text-xs text-text-muted">
                  Loading documents...
                </div>
              ) : (
                <div className="grid grid-cols-3 gap-2">
                  {dsDocs.map(doc => (
                    <Button key={doc.id as string} variant="ghost" render={<div/>} onClick={() => handleDatasetDoc(doc)}
                      className="group bg-bg-surface border border-border rounded-lg p-1.5 text-left hover:border-accent-violet hover:shadow-sm !inline-flex !flex-col !gap-0 !h-auto !w-full !whitespace-normal !rounded-lg !p-1.5 !text-left">
                      <div className="aspect-[3/4] bg-bg-base rounded mb-1 flex items-center justify-center overflow-hidden">
                        {doc.image_path ? (
                          <img src={`/api/image/${encodeURIComponent(doc.image_path as string)}`} alt=""
                            className="w-full h-full object-cover" />
                        ) : (
                          <HugeiconsIcon icon={File01Icon} className="size-5 text-text-muted" />
                        )}
                      </div>
                      <div className="text-xs text-text-muted truncate leading-tight">{doc.filename as string}</div>
                      <div className="text-[13px] text-text-muted leading-tight">{(doc.model as string).replace('invoice_dataset_', '')}</div>
                    </Button>
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
      </div>
    </div>
  )
}
