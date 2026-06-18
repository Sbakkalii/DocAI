import { Button } from "@/components/ui/button"
import { useState, useEffect } from 'react'
import { HugeiconsIcon } from "@hugeicons/react"
import { Loading01Icon, File01Icon } from "@hugeicons/core-free-icons"

export function DatasetView() {
  const [stats, setStats] = useState<Record<string, unknown> | null>(null)
  const [loading, setLoading] = useState(true)
  const [docs, setDocs] = useState<Array<Record<string, unknown>>>([])
  const [totalDocPages, setTotalDocPages] = useState(1)
  const [docPage, setDocPage] = useState(1)
  const [docModel, setDocModel] = useState<string>('')
  const [previewDoc, setPreviewDoc] = useState<Record<string, unknown> | null>(null)
  const [previewFields, setPreviewFields] = useState<Array<Record<string, unknown>>>([])
  const [tab, setTab] = useState<'overview' | 'documents'>('overview')

  useEffect(() => {
    fetch('/api/dataset/stats')
      .then(r => r.json())
      .then(d => { setStats(d); setLoading(false) })
      .catch(() => setLoading(false))
  }, [])

  useEffect(() => {
    const isPerModel = !docModel
    const url = isPerModel
      ? `/api/dataset/documents?page=${docPage}&per_model=1`
      : `/api/dataset/documents?model=${docModel}&page=${docPage}&per_page=12`
    fetch(url)
      .then(r => r.json())
      .then(d => {
        setDocs(d.documents || [])
        if (isPerModel) {
          setTotalDocPages(d.total_pages || 1)
        } else {
          setTotalDocPages(Math.ceil((d.total || 0) / (d.per_page || 12)))
        }
      })
      .catch(() => {})
  }, [docPage, docModel])

  if (loading) return (
    <div className="h-full flex items-center justify-center">
      <HugeiconsIcon icon={Loading01Icon} className="size-6 animate-spin text-accent-violet" />
    </div>
  )

  if (!stats) return (
    <div className="h-full flex items-center justify-center text-sm text-text-muted">
      Failed to load dataset info
    </div>
  )

  const fieldCounts = stats.field_counts as Record<string, number> || {}
  const fieldCoverage = stats.field_coverage as Record<string, number> || {}
  const topValues = stats.top_values as Record<string, Array<{value: string; count: number}>> || {}
  const models = stats.models as Record<string, {images: number; annotations: number; field_counts: Record<string, number>}> || {}
  const totalImages = stats.total_images as number || 0
  const totalAnns = stats.total_annotations as number || 0
  const avgValLen = stats.avg_value_length as number || 0
  const totalModels = stats.total_models as number || 0

  const maxFieldCount = Math.max(...Object.values(fieldCounts), 1)
  const maxCoverage = totalModels

  async function loadPreview(doc: Record<string, unknown>) {
    setPreviewDoc(doc)
    try {
      const tsvPath = doc.annotation_path as string
      const res = await fetch(`/api/dataset/annotations?path=${encodeURIComponent(tsvPath)}`)
      const data = await res.json()
      setPreviewFields(data.annotations || [])
    } catch {
      setPreviewFields([])
    }
  }

  return (
    <div className="h-full overflow-y-auto p-6 space-y-6">
      <div className="flex items-center gap-3 border-b border-border pb-3">
        <Button variant="ghost" size="sm" onClick={() => setTab('overview')}
          className={`font-semibold pb-1 border-b-2 ${
            tab === 'overview' ? 'text-accent-violet border-accent-violet' : 'text-text-muted border-transparent hover:text-text-primary'
          }`}>Overview</Button>
        <Button variant="ghost" size="sm" onClick={() => setTab('documents')}
          className={`font-semibold pb-1 border-b-2 ${
            tab === 'documents' ? 'text-accent-violet border-accent-violet' : 'text-text-muted border-transparent hover:text-text-primary'
          }`}>Documents ({totalImages})</Button>
      </div>

      {tab === 'overview' && (
        <div className="space-y-6 max-w-5xl">
          <div className="grid grid-cols-4 gap-3">
            <div className="bg-bg-surface border border-border rounded-xl p-4">
              <div className="text-xs font-semibold text-text-muted uppercase tracking-wider mb-1">Documents</div>
              <div className="text-2xl font-bold text-text-primary tabular-nums">{totalImages}</div>
              <div className="text-xs text-text-muted mt-1">{totalModels} model directories</div>
            </div>
            <div className="bg-bg-surface border border-border rounded-xl p-4">
              <div className="text-xs font-semibold text-text-muted uppercase tracking-wider mb-1">Annotations</div>
              <div className="text-2xl font-bold text-accent-violet tabular-nums">{totalAnns.toLocaleString()}</div>
              <div className="text-xs text-text-muted mt-1">{avgValLen} avg chars/val</div>
            </div>
            <div className="bg-bg-surface border border-border rounded-xl p-4">
              <div className="text-xs font-semibold text-text-muted uppercase tracking-wider mb-1">Unique Fields</div>
              <div className="text-2xl font-bold text-accent-green tabular-nums">{Object.keys(fieldCounts).length}</div>
              <div className="text-xs text-text-muted mt-1">across all documents</div>
            </div>
            <div className="bg-bg-surface border border-border rounded-xl p-4">
              <div className="text-xs font-semibold text-text-muted uppercase tracking-wider mb-1">Field Coverage</div>
              <div className="text-2xl font-bold text-accent-yellow tabular-nums">{Object.keys(fieldCoverage).length}</div>
              <div className="text-xs text-text-muted mt-1">fields in {totalModels} dirs</div>
            </div>
          </div>

          <div className="bg-bg-surface border border-border rounded-xl p-5">
            <h3 className="text-sm font-semibold text-text-primary mb-4">Field Frequency</h3>
            <div className="space-y-2">
              {Object.entries(fieldCounts).sort(([,a], [,b]) => (b as number) - (a as number)).map(([field, count]) => {
                const pct = (count as number) / maxFieldCount * 100
                return (
                  <div key={field} className="flex items-center gap-3">
                    <span className="text-xs text-text-muted min-w-[8rem] truncate">{field}</span>
                    <div className="flex-1 h-3 bg-bg-elevated rounded-full overflow-hidden">
                      <div className="h-full rounded-full bg-accent-violet/60" style={{ width: `${Math.max(pct, 2)}%` }} />
                    </div>
                    <span className="text-xs text-text-muted font-mono tabular-nums min-w-[4rem] text-right">{(count as number).toLocaleString()}</span>
                    <span className="text-xs text-text-muted min-w-[4rem]">{((count as number) / totalImages * 100).toFixed(0)}% docs</span>
                  </div>
                )
              })}
            </div>
          </div>

          <div className="bg-bg-surface border border-border rounded-xl p-5">
            <h3 className="text-sm font-semibold text-text-primary mb-4">Cross-Model Field Coverage</h3>
            <p className="text-xs text-text-muted mb-3">How many model directories contain each field (out of {totalModels})</p>
            <div className="space-y-2">
              {Object.entries(fieldCoverage).sort(([,a], [,b]) => (b as number) - (a as number)).map(([field, count]) => {
                const pct = (count as number) / maxCoverage * 100
                return (
                  <div key={field} className="flex items-center gap-3">
                    <span className="text-xs text-text-muted min-w-[8rem] truncate">{field}</span>
                    <div className="flex-1 h-2 bg-border rounded-full overflow-hidden">
                      <div className={`h-full rounded-full ${count === totalModels ? 'bg-accent-violet/60' : count as number > totalModels / 2 ? 'bg-accent-yellow/60' : 'bg-accent-coral/60'}`}
                        style={{ width: `${Math.max(pct, 2)}%` }} />
                    </div>
                    <span className="text-xs text-text-muted font-mono tabular-nums">{count}/{totalModels}</span>
                  </div>
                )
              })}
            </div>
          </div>

          <div className="bg-bg-surface border border-border rounded-xl p-5">
            <h3 className="text-sm font-semibold text-text-primary mb-4">Most Frequent Values by Field</h3>
            <div className="grid grid-cols-2 gap-4">
              {Object.entries(topValues).sort().map(([field, values]) => (
                <div key={field} className="bg-bg-elevated/50 rounded-lg p-3">
                  <h4 className="text-xs font-semibold text-text-primary mb-2">{field}</h4>
                  <div className="space-y-1">
                    {(values as Array<{value: string; count: number}>).slice(0, 6).map((v, i) => (
                      <div key={i} className="flex items-center justify-between text-[13px]">
                        <span className="text-text-muted truncate mr-2">{v.value}</span>
                        <span className="text-text-muted font-mono shrink-0">{v.count}</span>
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="bg-bg-surface border border-border rounded-xl p-5">
            <h3 className="text-sm font-semibold text-text-primary mb-4">Per-Model Statistics</h3>
            <div className="grid grid-cols-3 gap-3">
              {Object.entries(models).sort().map(([modelName, m]) => (
                <div key={modelName} className="bg-bg-elevated/50 rounded-lg p-3">
                  <h4 className="text-xs font-semibold text-text-primary mb-1">{modelName.replace('invoice_dataset_', '')}</h4>
                  <div className="text-[13px] text-text-muted">
                    <div>{m.images} images</div>
                    <div>{m.annotations} annotation files</div>
                    <div>{Object.keys(m.field_counts || {}).length} field types</div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {tab === 'documents' && (
        <div className="space-y-4">
          <div className="flex items-center gap-2 flex-wrap">
            <Button variant={!docModel ? 'default' : 'ghost'} size="sm" onClick={() => { setDocModel(''); setDocPage(1) }}
              className={`text-xs`}>
              All Models
            </Button>
            {Array.from({length: 9}, (_, i) => `model_${i + 1}`).map(m => (
              <Button key={m} variant={docModel === m ? 'default' : 'ghost'} size="sm" onClick={() => { setDocModel(m); setDocPage(1) }}
                className={`text-xs`}>
                {m.replace('model_', 'Model ')}
              </Button>
            ))}
          </div>

          <div className={`grid ${docModel ? 'grid-cols-4' : 'grid-cols-9'} gap-3`}>
            {(docs as Array<Record<string, unknown>>).map(doc => (
              <Button key={doc.id as string} variant="ghost" onClick={() => loadPreview(doc)}
                className={`bg-bg-surface border rounded-lg p-2 text-left transition-all hover:border-accent-violet ${
                  previewDoc?.id === doc.id ? 'border-accent-violet ring-1 ring-accent-violet/30' : 'border-border'
                }`}>
                <div className="aspect-[3/4] bg-bg-elevated rounded mb-1.5 flex items-center justify-center overflow-hidden">
                  {doc.image_path ? (
                    <img src={`/api/image/${encodeURIComponent(doc.image_path as string)}`} alt=""
                      className="w-full h-full object-cover" />
                  ) : (
                    <HugeiconsIcon icon={File01Icon} className="size-6 text-text-muted" />
                  )}
                </div>
                <div className="text-xs text-text-muted truncate">{doc.filename as string}</div>
                <div className="text-[13px] text-text-muted">{(doc.model as string).replace('invoice_dataset_', '')}</div>
              </Button>
            ))}
          </div>

          {totalDocPages > 1 && (
            <div className="flex items-center justify-center gap-2 text-xs">
              <Button variant="secondary" size="sm" disabled={docPage <= 1} onClick={() => setDocPage(p => p - 1)}>Prev</Button>
              <span className="text-text-muted">Page {docPage} of {totalDocPages}</span>
              <Button variant="secondary" size="sm" disabled={docPage >= totalDocPages} onClick={() => setDocPage(p => p + 1)}>Next</Button>
            </div>
          )}

          {previewDoc && (
            <div className="bg-bg-surface border border-border rounded-xl p-5">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-sm font-semibold text-text-primary">{previewDoc.filename as string}</h3>
                <span className="text-xs text-text-muted">{(previewDoc.model as string).replace('invoice_dataset_', '')}</span>
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  {(previewDoc.image_path as string) && (
                    <img src={`/api/image/${encodeURIComponent(previewDoc.image_path as string)}`} alt=""
                      className="w-full rounded-lg border border-border" />
                  )}
                </div>
                <div>
                  <h4 className="text-xs font-semibold text-text-muted mb-2">Ground Truth Fields</h4>
                  {previewFields.length === 0 ? (
                    <div className="text-xs text-text-muted italic">Loading annotations...</div>
                  ) : (
                    <div className="space-y-1 max-h-96 overflow-y-auto">
                      {previewFields.map((ann: Record<string, unknown>, i: number) => (
                        <div key={i} className="flex items-center gap-2 text-xs py-0.5">
                          <span className="px-1.5 py-0.5 rounded bg-bg-elevated text-xs font-mono text-text-muted shrink-0"
                            style={{color: ann.color as string || '#95a5a6'}}>{ann.label as string}</span>
                          <span className="text-text-primary truncate">{ann.text as string}</span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
