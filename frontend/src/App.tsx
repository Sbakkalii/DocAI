import { useState, useEffect, useCallback } from 'react'
import { Button } from '@/components/ui/button'
import { TooltipProvider } from '@/components/ui/tooltip'
import { HugeiconsIcon } from "@hugeicons/react"
import { Upload01Icon, File01Icon, AlertCircleIcon, BarChartIcon, Sun01Icon, MoonIcon } from "@hugeicons/core-free-icons"
import { UploadShell } from './components/UploadShell'
import { PipelineView } from './components/PipelineView'
import { DatasetView } from './components/DatasetView'
import { BatchEvalView } from './components/BatchEvalView'
import { BatchQueueView } from './components/BatchQueueView'
import type { PipelineResult, StepState } from './types'

export default function App() {
  const [activeView, setActiveView] = useState<'pipeline' | 'results' | 'dataset' | 'batch_eval' | 'batch_queue'>('pipeline')
  const [sessionId, setSessionId] = useState<string | null>(null)
  const [result, setResult] = useState<PipelineResult | null>(null)
  const [steps, setSteps] = useState<Record<string, StepState>>({})
  const [selectedStep, setSelectedStep] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [uploadFileName, setUploadFileName] = useState<string>('')
  const [waiting, setWaiting] = useState(false)
  const [, setNextStepName] = useState<string | null>(null)

  const [theme, setTheme] = useState<'dark' | 'light'>('dark')

  useEffect(() => {
    document.documentElement.classList.toggle('light', theme === 'light')
  }, [theme])

  const handleStart = useCallback((sid: string, fileName: string) => {
    setSessionId(sid)
    setUploadFileName(fileName)
    setSteps({})
    setSelectedStep(null)
    setError(null)
    setWaiting(false)
    setResult(null)
    setActiveView('pipeline')
  }, [])

  const handleExplore = useCallback((data: PipelineResult) => {
    setResult(data)
    setSelectedStep('llm_extraction')
  }, [])

  const handleReset = useCallback(() => {
    setSessionId(null)
    setResult(null)
    setSteps({})
    setSelectedStep(null)
    setError(null)
    setWaiting(false)
    setNextStepName(null)
    setUploadFileName('')
    setResult(null)
    setActiveView('pipeline')
  }, [])

  const completedCount = Object.values(steps).filter(s => s.status === 'completed').length

  return (
    <TooltipProvider delayDuration={300}>
    <div className="h-screen bg-bg-base flex flex-col">
      {/* Top bar */}
      <header className="h-14 bg-bg-surface/90 backdrop-blur-sm border-b border-border flex items-center justify-between px-6 shrink-0 shadow-sm">
        <div className="flex items-center gap-3">
          <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-accent-violet to-accent-violet-dark flex items-center justify-center text-white font-bold text-xs shadow-sm">D</div>
          {sessionId && (
            <div className="flex items-center gap-1 bg-bg-surface rounded-lg p-0.5 border border-border">
              <Button variant="ghost" size="sm" onClick={() => setActiveView('pipeline')}
                className={activeView === 'pipeline' ? '!bg-accent-violet/20 !text-accent-violet !shadow-sm' : ''}>
                Pipeline
              </Button>
            </div>
          )}
          {sessionId && (
            <div className="flex items-center gap-2 text-xs text-text-muted">
              <span>{completedCount} steps completed</span>
              {waiting && <span className="text-accent-violet">· waiting</span>}
              {error && <span className="text-accent-coral">· error</span>}
            </div>
          )}
          {!sessionId && activeView === 'dataset' ? (
            <div className="text-xs text-text-muted">Dataset Explorer</div>
          ) : !sessionId ? (
            <div className="text-xs text-text-muted">Agentic Document Intelligence</div>
          ) : null}
        </div>
        <div className="flex items-center gap-2">
          {!sessionId && (
            <div className="flex items-center gap-1 bg-bg-surface rounded-lg p-0.5 border border-border">
              <Button variant="ghost" size="sm" onClick={() => setActiveView('pipeline')}
                className={activeView === 'pipeline' ? '!bg-accent-violet/20 !text-accent-violet !shadow-sm' : ''}>
                <HugeiconsIcon icon={Upload01Icon} className="size-3 inline mr-1" />Upload
              </Button>
              <Button variant="ghost" size="sm" onClick={() => setActiveView('dataset')}
                className={activeView === 'dataset' ? '!bg-accent-violet/20 !text-accent-violet !shadow-sm' : ''}>
                <HugeiconsIcon icon={File01Icon} className="size-3 inline mr-1" />Dataset
              </Button>
              <Button variant="ghost" size="sm" onClick={() => setActiveView('batch_eval')}
                className={activeView === 'batch_eval' ? '!bg-accent-violet/20 !text-accent-violet !shadow-sm' : ''}>
                <HugeiconsIcon icon={BarChartIcon} className="size-3 inline mr-1" />Batch Eval
              </Button>
              <Button variant="ghost" size="sm" onClick={() => setActiveView('batch_queue')}
                className={activeView === 'batch_queue' ? '!bg-accent-violet/20 !text-accent-violet !shadow-sm' : ''}>
                <HugeiconsIcon icon={File01Icon} className="size-3 inline mr-1" />Queue
              </Button>
            </div>
          )}
          {sessionId && (
            <span className="text-xs text-text-muted bg-bg-surface px-2 py-1 rounded-md font-mono tabular-nums border border-border" title={uploadFileName}>
              {uploadFileName}
            </span>
          )}
          {sessionId && (
            <Button variant="ghost" size="sm" onClick={handleReset}>
              <HugeiconsIcon icon={Upload01Icon} className="size-3" />
              New Document
            </Button>
          )}
          <Button variant="ghost" size="icon-sm" onClick={() => setTheme(t => t === 'dark' ? 'light' : 'dark')}
            title={`Switch to ${theme === 'dark' ? 'light' : 'dark'} theme`}>
            {theme === 'dark' ? <HugeiconsIcon icon={Sun01Icon} className="size-4" /> : <HugeiconsIcon icon={MoonIcon} className="size-4" />}
          </Button>
        </div>
      </header>

      {/* Error Toast */}
      {error && (
        <div className="absolute top-16 left-1/2 -translate-x-1/2 z-50 flex items-center gap-2 px-4 py-2.5 bg-accent-coral/20 border border-accent-coral/30 rounded-xl text-sm text-accent-coral shadow-lg animate-scale-in backdrop-blur-sm">
          <HugeiconsIcon icon={AlertCircleIcon} className="size-4 shrink-0 text-accent-coral" />
          {error}
        </div>
      )}

      {/* Views */}
      <div className="flex-1 overflow-hidden">
        {!sessionId && activeView === 'batch_eval' ? (
          <BatchEvalView />
        ) : !sessionId && activeView === 'batch_queue' ? (
          <BatchQueueView />
        ) : !sessionId && activeView === 'dataset' ? (
          <DatasetView />
        ) : !sessionId ? (
          <UploadShell onStart={handleStart} />
        ) : (
          <>
            {/* Always mount PipelineView (hidden = WebSocket stays alive so onDone fires) */}
            <div className={activeView === 'pipeline' ? 'flex-1 overflow-hidden' : 'hidden'}>
              <PipelineView
                sessionId={sessionId}
                onDone={handleExplore}
                steps={steps}
                setSteps={setSteps}
                error={error}
                setError={setError as (e: string | null) => void}
                selectedStep={selectedStep}
                onSelectStep={setSelectedStep}
                waiting={waiting}
                setWaiting={setWaiting}
                setNextStepName={setNextStepName}
                resultReady={!!result}
              />
            </div>
          </>
        )}
      </div>
    </div>
    </TooltipProvider>
  )
}
