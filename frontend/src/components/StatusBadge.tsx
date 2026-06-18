import { HugeiconsIcon } from "@hugeicons/react"
import { CheckmarkCircleIcon, AlertCircleIcon } from "@hugeicons/core-free-icons"

export function StatusBadge({ status }: { status: string }) {
  const styles: Record<string, string> = {
    completed: 'bg-accent-green/15 text-accent-green border border-accent-green/30',
    running: 'bg-accent-yellow/15 text-accent-yellow border border-accent-yellow/30 animate-pulse',
    failed: 'bg-accent-coral/15 text-accent-coral border border-accent-coral/30',
    pending: 'bg-border/50 text-text-muted border border-border',
  }
  const labels: Record<string, string> = {
    completed: 'Done',
    running: 'Processing',
    failed: 'Failed',
    pending: 'Pending',
  }
  return (
    <span className={`inline-flex items-center gap-1 px-2.5 py-1 rounded-xl text-xs font-semibold ${styles[status] || styles.pending}`}>
      {status === 'running' && <span className="w-1.5 h-1.5 rounded-full bg-accent-yellow animate-pulse" />}
      {status === 'completed' && <HugeiconsIcon icon={CheckmarkCircleIcon} className="size-3" />}
      {status === 'failed' && <HugeiconsIcon icon={AlertCircleIcon} className="size-3" />}
      {labels[status] || status}
    </span>
  )
}
