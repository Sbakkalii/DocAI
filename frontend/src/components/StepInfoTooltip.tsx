import { Button } from "@/components/ui/button"
import { HugeiconsIcon } from "@hugeicons/react"
import { Cancel01Icon } from "@hugeicons/core-free-icons"
import { STEP_LABELS, STEP_INFO } from './constants'

export function StepInfoTooltip({ step, onClose }: { step: string; onClose: () => void }) {
  const info = STEP_INFO[step]
  if (!info) return null
  return (
    <div className="absolute left-0 right-0 z-50 mt-1 mx-2 p-3 bg-bg-elevated border border-accent-violet/30 rounded-xl shadow-xl animate-fade-in"
      onClick={e => e.stopPropagation()}>
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs font-semibold text-accent-violet">{STEP_LABELS[step] || step}</span>
        <Button variant="ghost" size="icon-sm" onClick={onClose} className="text-text-muted hover:text-text-primary">
          <HugeiconsIcon icon={Cancel01Icon} className="size-3" />
        </Button>
      </div>
      <div className="space-y-1.5 text-[13px]">
        <div><span className="text-text-muted font-medium">What it does:</span> <span className="text-text-primary">{info.what}</span></div>
        <div><span className="text-text-muted font-medium">How:</span> <span className="text-text-primary">{info.how}</span></div>
        <div><span className="text-text-muted font-medium">Expected output:</span> <span className="text-text-primary">{info.expected}</span></div>
      </div>
    </div>
  )
}
