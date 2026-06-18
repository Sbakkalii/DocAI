import { Button } from "@/components/ui/button"
import { useState, useCallback } from 'react'
import { HugeiconsIcon } from "@hugeicons/react"
import { Cancel01Icon, Download01Icon, Loading01Icon, PencilIcon, CheckIcon } from "@hugeicons/core-free-icons"
import type { PageResult, PipelineResult } from '../types'

interface Props {
  page: PageResult
  result: PipelineResult
  onClose: () => void
}

function formatJSON(obj: unknown): string {
  try {
    return JSON.stringify(obj, null, 2)
  } catch {
    return String(obj)
  }
}

export function JsonViewer({ page, result, onClose }: Props) {
  const [activeTab, setActiveTab] = useState<'page' | 'full'>('page')
  const [editing, setEditing] = useState(false)
  const [editText, setEditText] = useState('')
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState('')
  const [copied, setCopied] = useState(false)

  const pageData = {
    page_number: page.page_number,
    image_path: page.image_path,
    extracted_fields: page.extracted_fields,
    validation: page.validation,
    ocr_word_count: page.ocr_word_count,
    extraction_evidence: page.extraction_evidence,
    predicted_annotations: page.predicted_annotations,
  }

  const currentData = activeTab === 'page' ? pageData : result

  const jsonText = formatJSON(currentData)

  const handleStartEdit = () => {
    setEditText(jsonText)
    setEditing(true)
    setError('')
  }

  const handleCancelEdit = () => {
    setEditing(false)
    setEditText('')
    setError('')
  }

  const handleSave = useCallback(async () => {
    setSaving(true)
    setError('')
    try {
      let parsed: Record<string, unknown>
      try {
        parsed = JSON.parse(editText)
      } catch {
        setError('Invalid JSON — please fix syntax errors')
        setSaving(false)
        return
      }

      const corrections = parsed?.extracted_fields as Record<string, unknown> | undefined
      if (!corrections || typeof corrections !== 'object') {
        setError('JSON must contain an "extracted_fields" object')
        setSaving(false)
        return
      }

      const res = await fetch(`/api/correct/${result.session_id}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ corrections }),
      })
      if (!res.ok) {
        const errData = await res.json().catch(() => ({}))
        throw new Error(errData.detail || `HTTP ${res.status}`)
      }
      setEditing(false)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Save failed')
    }
    setSaving(false)
  }, [editText, result.session_id])

  const handleDownload = () => {
    const blob = new Blob([formatJSON(result)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `result-${result.session_id}.json`
    a.click()
    URL.revokeObjectURL(url)
  }

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(jsonText)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    } catch { /* ignore */ }
  }

  return (
    <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-50 p-4" onClick={onClose}>
      <div className="bg-bg-surface border border-border rounded-2xl shadow-2xl w-full max-w-3xl max-h-[85vh] flex flex-col"
        onClick={e => e.stopPropagation()}>
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-3 border-b border-border shrink-0">
          <div className="flex items-center gap-3">
            <h3 className="text-sm font-semibold text-text-primary">JSON Output</h3>
            <div className="flex bg-bg-elevated rounded-lg p-0.5 gap-0.5">
              <Button variant={activeTab === 'page' ? 'default' : 'ghost'} size="sm" onClick={() => setActiveTab('page')}
                className={`text-xs ${activeTab === 'page' ? 'shadow-sm' : ''}`}>
                Current Page
              </Button>
              <Button variant={activeTab === 'full' ? 'default' : 'ghost'} size="sm" onClick={() => setActiveTab('full')}
                className={`text-xs ${activeTab === 'full' ? 'shadow-sm' : ''}`}>
                Full Result
              </Button>
            </div>
          </div>
          <div className="flex items-center gap-1.5">
            <Button variant="ghost" size="icon-sm" onClick={handleDownload}
              className="text-text-muted hover:text-text-primary"
              title="Download JSON">
              <HugeiconsIcon icon={Download01Icon} className="size-4" />
            </Button>
            {editing ? (
              <>
                <Button variant="default" size="sm" onClick={handleSave} disabled={saving}
                  className="bg-accent-green hover:bg-accent-green/80 text-white">
                  {saving ? <HugeiconsIcon icon={Loading01Icon} className="size-3 animate-spin" /> : <HugeiconsIcon icon={CheckIcon} className="size-3" />}
                  Save
                </Button>
                <Button variant="ghost" size="sm" onClick={handleCancelEdit} disabled={saving}>
                  Cancel
                </Button>
              </>
            ) : (
              <Button variant="ghost" size="icon-sm" onClick={handleStartEdit}
                className="text-text-muted hover:text-text-primary"
                title="Edit fields">
                <HugeiconsIcon icon={PencilIcon} className="size-4" />
              </Button>
            )}
            <Button variant="ghost" size="icon-sm" onClick={onClose}
              className="text-text-muted hover:text-text-primary">
              <HugeiconsIcon icon={Cancel01Icon} className="size-4" />
            </Button>
          </div>
        </div>

        {/* Body */}
        <div className="flex-1 overflow-y-auto p-4">
          {error && (
            <div className="mb-3 px-3 py-2 bg-accent-coral/20 border border-accent-coral/30 rounded-lg text-xs text-accent-coral">
              {error}
            </div>
          )}
          {editing ? (
            <textarea
              value={editText}
              onChange={e => setEditText(e.target.value)}
              className="w-full h-full min-h-[50vh] bg-bg-base text-xs text-text-primary font-mono p-4 rounded-lg border border-border focus:border-accent-violet focus:ring-1 focus:ring-accent-violet/30 outline-none resize-none"
              style={{ fontFamily: "'JetBrains Mono', monospace", tabSize: 2 }}
              spellCheck={false}
            />
          ) : (
            <pre className="text-xs text-text-primary font-mono leading-relaxed whitespace-pre-wrap bg-bg-base p-4 rounded-lg border border-border max-h-[60vh] overflow-y-auto"
              style={{ fontFamily: "'JetBrains Mono', monospace" }}>
              {jsonText}
            </pre>
          )}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between px-5 py-2.5 border-t border-border shrink-0">
          <span className="text-xs text-text-muted">
            {editing ? 'Edit the extracted_fields values above and save to apply corrections' : 'Read-only view — click the pencil to edit'}
          </span>
          <Button variant="ghost" size="sm" onClick={handleCopy}>
            {copied ? 'Copied!' : 'Copy'}
          </Button>
        </div>
      </div>
    </div>
  )
}
