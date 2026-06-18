import { useState, useEffect, useCallback } from 'react'
import { Button } from '@/components/ui/button'
import { HugeiconsIcon } from '@hugeicons/react'
import { Loading01Icon, CheckmarkCircleIcon, AlertCircleIcon, Upload01Icon } from '@hugeicons/core-free-icons'

interface BatchDoc {
  id: string
  batch_id: string
  filename: string
  status: string
  confidence?: number | null
  needs_review?: number
  error?: string | null
  elapsed?: number | null
  created_at: string
  completed_at?: string | null
}

interface Batch {
  id: string
  status: string
  total_docs: number
  completed_docs: number
  failed_docs: number
  priority: string
  created_at: string
  started_at?: string | null
  completed_at?: string | null
  docs?: BatchDoc[]
}

interface Stats {
  total_batches: number
  active_batches: number
  total_docs: number
  completed_docs: number
  failed_docs: number
  queued_docs: number
}

export function BatchQueueView() {
  const [stats, setStats] = useState<Stats | null>(null)
  const [batches, setBatches] = useState<Batch[]>([])
  const [uploading, setUploading] = useState(false)
  const [selectedBatch, setSelectedBatch] = useState<Batch | null>(null)

  const fetchData = useCallback(async () => {
    try {
      const res = await fetch('/api/batch/stats')
      if (res.ok) setStats(await res.json())
    } catch { /* ignore */ }
  }, [])

  useEffect(() => {
    fetchData()
    const interval = setInterval(fetchData, 5000)
    return () => clearInterval(interval)
  }, [fetchData])

  async function handleBatchUpload(e: React.ChangeEvent<HTMLInputElement>) {
    const files = e.target.files
    if (!files?.length) return
    setUploading(true)
    try {
      const form = new FormData()
      for (const f of Array.from(files)) form.append('files', f)
      form.append('priority', 'normal')
      const res = await fetch('/api/batch', { method: 'POST', body: form })
      if (res.ok) {
        const data = await res.json()
        setBatches(prev => [...prev, { id: data.batch_id, status: 'queued', total_docs: data.total_docs, completed_docs: 0, failed_docs: 0, priority: 'normal', created_at: new Date().toISOString() }])
      }
    } catch { /* ignore */ }
    setUploading(false)
    if (e.target) e.target.value = ''
  }

  async function viewBatch(batchId: string) {
    try {
      const res = await fetch(`/api/batch/${batchId}`)
      if (res.ok) setSelectedBatch(await res.json())
    } catch { /* ignore */ }
  }

  const statusColor = (s: string) => {
    switch (s) {
      case 'completed': return 'text-accent-green'
      case 'running': return 'text-accent-violet'
      case 'failed': return 'text-accent-coral'
      case 'queued': return 'text-text-muted'
      default: return 'text-text-muted'
    }
  }

  return (
    <div className="h-screen bg-bg-base flex flex-col">
      <header className="h-14 bg-bg-surface/90 backdrop-blur-sm border-b border-border flex items-center justify-between px-6 shrink-0">
        <div className="flex items-center gap-2">
          <div className="text-sm font-semibold text-text-primary">Batch Queue Dashboard</div>
          {stats && (
            <div className="flex items-center gap-3 ml-4 text-xs text-text-muted">
              <span>{stats.active_batches} active</span>
              <span>{stats.queued_docs} queued</span>
              <span>{stats.completed_docs} done</span>
              <span className="text-accent-coral">{stats.failed_docs} failed</span>
            </div>
          )}
        </div>
        <label className="cursor-pointer">
          <Button variant="default" size="sm" loading={uploading} onClick={() => document.getElementById('batch-file-input')?.click()}>
            <HugeiconsIcon icon={Upload01Icon} className="size-3.5 mr-1" />
            Upload Batch
          </Button>
          <input id="batch-file-input" type="file" multiple accept=".pdf,.jpg,.jpeg,.png,.zip" onChange={handleBatchUpload} className="hidden" />
        </label>
      </header>
      <div className="flex-1 overflow-auto p-6">
        <div className="max-w-4xl mx-auto space-y-6">
          {/* Selected batch detail */}
          {selectedBatch && (
            <div className="bg-bg-surface border border-border rounded-xl overflow-hidden">
              <div className="flex items-center justify-between px-5 py-3 border-b border-border">
                <div>
                  <h3 className="text-sm font-semibold text-text-primary">{selectedBatch.id}</h3>
                  <div className="text-xs text-text-muted mt-0.5">{selectedBatch.priority} priority · {selectedBatch.total_docs} docs</div>
                </div>
                <span className={`inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-xs font-medium ${selectedBatch.status === 'completed' ? 'bg-accent-green/15 text-accent-green' : selectedBatch.status === 'running' ? 'bg-accent-violet/15 text-accent-violet' : selectedBatch.status === 'failed' ? 'bg-accent-coral/15 text-accent-coral' : 'bg-bg-elevated text-text-muted'}`}>
                  {selectedBatch.status}
                </span>
              </div>
              {/* Progress bar */}
              {selectedBatch.total_docs > 0 && (
                <div className="px-5 py-3 border-b border-border">
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-xs text-text-muted">Progress</span>
                    <span className="text-xs text-text-muted tabular-nums">{selectedBatch.completed_docs + selectedBatch.failed_docs}/{selectedBatch.total_docs}</span>
                  </div>
                  <div className="h-2 bg-border rounded-full overflow-hidden flex">
                    <div className="h-full bg-accent-green transition-all" style={{ width: `${selectedBatch.total_docs > 0 ? (selectedBatch.completed_docs / selectedBatch.total_docs) * 100 : 0}%` }} />
                    <div className="h-full bg-accent-coral transition-all" style={{ width: `${selectedBatch.total_docs > 0 ? (selectedBatch.failed_docs / selectedBatch.total_docs) * 100 : 0}%` }} />
                  </div>
                </div>
              )}
              {/* Doc list */}
              {selectedBatch.docs && (
                <div className="divide-y divide-border/50">
                  {selectedBatch.docs.map(doc => (
                    <div key={doc.id} className="flex items-center gap-3 px-5 py-2.5 hover:bg-bg-elevated/20 transition-colors">
                      <HugeiconsIcon icon={doc.status === 'completed' ? CheckmarkCircleIcon : doc.status === 'failed' ? AlertCircleIcon : Loading01Icon}
                        className={`size-3.5 shrink-0 ${doc.status === 'completed' ? 'text-accent-green' : doc.status === 'failed' ? 'text-accent-coral' : 'text-accent-violet animate-spin'}`} />
                      <span className="flex-1 text-xs text-text-primary truncate">{doc.filename}</span>
                      {doc.needs_review === 1 && (
                        <span className="text-xs px-2 py-0.5 rounded-full bg-accent-coral/15 text-accent-coral font-medium">Review</span>
                      )}
                      {doc.confidence != null && (
                        <span className={`text-xs font-mono tabular-nums ${doc.confidence >= 0.85 ? 'text-accent-green' : doc.confidence >= 0.7 ? 'text-accent-yellow' : 'text-accent-coral'}`}>
                          {Math.round(doc.confidence * 100)}%
                        </span>
                      )}
                      {doc.elapsed != null && (
                        <span className="text-xs text-text-muted tabular-nums">{doc.elapsed.toFixed(1)}s</span>
                      )}
                      <span className={`text-xs ${statusColor(doc.status)}`}>{doc.status}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
          {/* Batch list */}
          <div className="bg-bg-surface border border-border rounded-xl">
            <div className="px-5 py-3 border-b border-border">
              <h3 className="text-sm font-semibold text-text-primary">Recent Batches</h3>
            </div>
            {batches.length === 0 && !selectedBatch ? (
              <div className="p-8 text-center text-sm text-text-muted">Upload a batch of documents to get started</div>
            ) : (
              <div className="divide-y divide-border/50">
                {batches.map(b => (
                  <button key={b.id} onClick={() => viewBatch(b.id)}
                    className="w-full flex items-center gap-4 px-5 py-3 text-left hover:bg-bg-elevated/20 transition-colors">
                    <span className="text-xs font-mono text-text-muted truncate flex-1">{b.id}</span>
                    <span className="text-xs tabular-nums text-text-muted">{b.completed_docs}/{b.total_docs}</span>
                    <span className={`text-xs ${statusColor(b.status)} capitalize`}>{b.status}</span>
                  </button>
                ))}
                {selectedBatch && (
                  <button onClick={() => viewBatch(selectedBatch.id)}
                    className="w-full flex items-center gap-4 px-5 py-3 text-left bg-accent-violet/10">
                    <span className="text-xs font-mono text-text-primary truncate flex-1">{selectedBatch.id}</span>
                    <span className="text-xs tabular-nums text-text-muted">{selectedBatch.completed_docs}/{selectedBatch.total_docs}</span>
                    <span className={`text-xs ${statusColor(selectedBatch.status)} capitalize`}>{selectedBatch.status}</span>
                  </button>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
