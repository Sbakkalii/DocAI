import { Button } from "@/components/ui/button"
import { useState, useRef, useEffect } from 'react'
import { HugeiconsIcon } from "@hugeicons/react"
import { ZoomInAreaIcon, ZoomOutAreaIcon, Layers01Icon, Loading01Icon } from "@hugeicons/core-free-icons"
import type { BBox, AnnotationBox } from '../types'

interface Props {
  imagePath: string
  boxes?: BBox[]
  imageWidth?: number
  imageHeight?: number
  pageNumber?: number
  groundTruthAnnotations?: AnnotationBox[]
  predictedAnnotations?: AnnotationBox[]
  highlightedEvidence?: AnnotationBox[]  // field(s) to glow-highlight
  onSelectField?: (fieldName: string) => void
}

const ANNOTATION_COLORS: Record<string, string> = {
  NUMBER: '#ff4757',
  SUPPLIER: '#2ed573',
  ADDRESS: '#1e90ff',
  INVOICE_DATE: '#ffa502',
  INVOICE_DUE_DATE: '#ff6348',
  PO_NUMBER: '#a29bfe',
  TOTAL: '#00d2d3',
  TOTAL_AMOUNT: '#6c5ce7',
  TOTAL_UNTAXED: '#7bed9f',
  TAX_AMOUNT: '#e056fd',
  'LINE/DESCRIPTION': '#f9ca24',
  'LINE/QUANTITY': '#0abde3',
  'LINE/UOM': '#26de81',
  'LINE/PRICE': '#45aaf2',
  'LINE/SUB_TOTAL': '#fc5c65',
  'LINE/TAX': '#8854d0',
  O: '#95a5a6',
}

const ANNOTATION_GROUPS = [
  { key: 'header', label: 'Header', fields: ['NUMBER', 'SUPPLIER', 'ADDRESS', 'INVOICE_DATE', 'INVOICE_DUE_DATE', 'PO_NUMBER'] },
  { key: 'totals', label: 'Totals', fields: ['TOTAL', 'TOTAL_AMOUNT', 'TOTAL_UNTAXED', 'TAX_AMOUNT'] },
  { key: 'lines', label: 'Line Items', fields: ['LINE/DESCRIPTION', 'LINE/QUANTITY', 'LINE/UOM', 'LINE/PRICE', 'LINE/SUB_TOTAL', 'LINE/TAX'] },
  { key: 'other', label: 'Other', fields: ['O'] },
]

function getDefaultColor(label: string): string {
  return ANNOTATION_COLORS[label] || '#95a5a6'
}

export default function DocumentViewer({
  imagePath, boxes, imageWidth, imageHeight, pageNumber,
  groundTruthAnnotations, predictedAnnotations,
  highlightedEvidence, onSelectField,
}: Props) {
  const [imgNatural, setImgNatural] = useState({ w: 0, h: 0 })
  const [hoveredWord, setHoveredWord] = useState<string | null>(null)
  const [hoveredAnnotation, setHoveredAnnotation] = useState<AnnotationBox | null>(null)
  const [showBoxes, setShowBoxes] = useState(true)
  const [showGT, setShowGT] = useState(true)
  const [showPredicted, setShowPredicted] = useState(true)
  const [zoom, setZoom] = useState<number | null>(null)
  const [panOffset, setPanOffset] = useState({ x: 0, y: 0 })
  const fitRef = useRef<HTMLDivElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const imgRef = useRef<HTMLImageElement>(null)
  const dragRef = useRef<{ startX: number; startY: number; startPanX: number; startPanY: number; dragging: boolean }>({ startX: 0, startY: 0, startPanX: 0, startPanY: 0, dragging: false })
  const isZoomed = zoom !== null && (zoom ?? 1) > 1

  const displayW = imageWidth || imgNatural.w
  const displayH = imageHeight || imgNatural.h

  const apiPath = imagePath
    ? `/api/image/${encodeURIComponent(imagePath)}`
    : null

  const hasAnnotations = (groundTruthAnnotations?.length || 0) > 0 || (predictedAnnotations?.length || 0) > 0

  const effectiveZoom = zoom ?? 1

  function handleMouseDown(e: React.MouseEvent<HTMLDivElement>) {
    if (!isZoomed) return
    e.preventDefault()
    dragRef.current = {
      startX: e.clientX,
      startY: e.clientY,
      startPanX: panOffset.x,
      startPanY: panOffset.y,
      dragging: true,
    }
  }

  function handleMouseMove(e: React.MouseEvent<HTMLDivElement>) {
    if (dragRef.current.dragging) {
      const dx = e.clientX - dragRef.current.startX
      const dy = e.clientY - dragRef.current.startY
      setPanOffset({
        x: dragRef.current.startPanX + dx,
        y: dragRef.current.startPanY + dy,
      })
      return
    }
    if (!displayW || !displayH || !imgRef.current) return
    const rect = imgRef.current.getBoundingClientRect()
    const x = ((e.clientX - rect.left) / rect.width) * displayW
    const y = ((e.clientY - rect.top) / rect.height) * displayH

    if (boxes) {
      const hit = boxes.find(b => {
        const [x0, y0, x1, y1] = b.box
        return x >= x0 && x <= x1 && y >= y0 && y <= y1
      })
      setHoveredWord(hit ? hit.word : null)
    }

    const allAnns = [
      ...(showGT ? groundTruthAnnotations || [] : []).map(a => ({ ...a, _source: 'gt' as const })),
      ...(showPredicted ? predictedAnnotations || [] : []).map(a => ({ ...a, _source: 'pred' as const })),
    ]
    const hitAnn = allAnns.find(a => {
      const [x0, y0, x1, y1] = a.box
      return x >= x0 && x <= x1 && y >= y0 && y <= y1
    })
    setHoveredAnnotation(hitAnn || null)
  }

  function handleMouseUp() {
    dragRef.current.dragging = false
  }

  useEffect(() => {
    function globalMouseUp() { dragRef.current.dragging = false }
    window.addEventListener('mouseup', globalMouseUp)
    return () => window.removeEventListener('mouseup', globalMouseUp)
  }, [])

  useEffect(() => {
    if (!isZoomed) setPanOffset({ x: 0, y: 0 })
  }, [isZoomed])

  return (
    <div className="bg-bg-surface rounded-2xl border border-border overflow-hidden w-full">
      {/* Toolbar */}
      <div className="flex items-center justify-between px-4 py-2 border-b border-border bg-bg-elevated/50 flex-wrap gap-2">
        <div className="flex items-center gap-2.5">
          <span className="text-[13px] font-semibold text-text-muted uppercase tracking-wider">
            {pageNumber ? `Page ${pageNumber}` : 'Document'}
          </span>
          {boxes && (
            <span className="text-[13px] text-text-muted tabular-nums">{boxes.length} words</span>
          )}
          {groundTruthAnnotations && groundTruthAnnotations.length > 0 && (
            <span className="text-[13px] text-accent-green font-medium tabular-nums">{groundTruthAnnotations.length} GT</span>
          )}
          {predictedAnnotations && predictedAnnotations.length > 0 && (
            <span className="text-[13px] text-accent-violet font-medium tabular-nums">{predictedAnnotations.length} pred</span>
          )}
        </div>

        <div className="flex items-center gap-1.5">
          {hasAnnotations && (
            <>
              <Button variant={showGT ? 'default' : 'outline'} size="sm" onClick={() => setShowGT(!showGT)}
                className={`text-xs ${showGT ? 'bg-accent-green text-white border-accent-green shadow-sm' : ''}`}>
                GT
              </Button>
              <Button variant={showPredicted ? 'default' : 'outline'} size="sm" onClick={() => setShowPredicted(!showPredicted)}
                className={`text-xs ${showPredicted ? 'shadow-sm' : ''}`}>
                Pred
              </Button>
            </>
          )}
          {boxes && boxes.length > 0 && (
            <Button variant={showBoxes ? 'default' : 'outline'} size="sm" onClick={() => setShowBoxes(!showBoxes)}
              className={`text-xs ${showBoxes ? 'shadow-sm' : ''}`}>
              <HugeiconsIcon icon={Layers01Icon} className="size-3 inline mr-1 -mt-0.5" />
              {showBoxes ? 'Hide boxes' : 'Boxes'}
            </Button>
          )}
          <div className="w-px h-5 bg-border mx-1" />
          <div className="flex items-center gap-0.5">
            <Button variant="ghost" size="icon-sm" onClick={() => setZoom(z => Math.max(0.25, (z ?? 1) - 0.25))}
              className="text-text-muted hover:text-text-primary">
              <HugeiconsIcon icon={ZoomOutAreaIcon} className="size-3.5" />
            </Button>
            <span className="text-xs text-text-muted w-10 text-center tabular-nums cursor-pointer hover:text-accent-violet font-medium"
              onClick={() => setZoom(null)} title="Fit to container">
              {zoom === null ? 'Fit' : `${Math.round(zoom * 100)}%`}
            </span>
            <Button variant="ghost" size="icon-sm" onClick={() => setZoom(z => Math.min(3, (z ?? 1) + 0.25))}
              className="text-text-muted hover:text-text-primary">
              <HugeiconsIcon icon={ZoomInAreaIcon} className="size-3.5" />
            </Button>
          </div>
        </div>
      </div>

      {/* Annotation type legend */}
      {hasAnnotations && (showGT || showPredicted) && (
        <div className="px-4 py-1.5 border-b border-border bg-bg-elevated/30 flex flex-wrap gap-x-4 gap-y-0.5 text-xs">
          {ANNOTATION_GROUPS.map(group => {
            const groupColor = getDefaultColor(group.fields[0])
            return (
              <div key={group.key} className="flex items-center gap-1.5">
                <span className="w-2 h-2 rounded-sm inline-block" style={{ backgroundColor: groupColor }} />
                <span className="text-text-muted">{group.label}</span>
              </div>
            )
          })}
        </div>
      )}

      {/* Document canvas */}
      <div ref={containerRef}
        className={`relative bg-bg-base flex items-start justify-center p-4 min-h-48 ${isZoomed ? (dragRef.current.dragging ? 'cursor-grabbing' : 'cursor-grab') : ''} select-none`}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={() => { dragRef.current.dragging = false; setHoveredWord(null); setHoveredAnnotation(null) }}>
        <div ref={fitRef} className="relative inline-block" style={{ transform: `translate(${panOffset.x}px, ${panOffset.y}px) scale(${effectiveZoom})`, transformOrigin: 'top center' }}>
          {apiPath ? (
            <img ref={imgRef}
              src={apiPath} alt="Document"
              className="max-w-full h-auto max-h-[50vh] shadow-md rounded-lg"
              onLoad={() => {
                if (imgRef.current) {
                  const nw = imgRef.current.naturalWidth
                  const nh = imgRef.current.naturalHeight
                  setImgNatural({ w: nw, h: nh })
                  if (fitRef.current && nw > 0 && nh > 0) {
                    const parent = fitRef.current.parentElement
                    if (parent) {
                      const pw = parent.clientWidth - 48
                      const ph = parent.clientHeight - 48
                      if (pw > 0 && ph > 0) {
                        const fit = Math.min(pw / nw, ph / nh, 1)
                        if (fit < 1) setZoom(Math.max(0.1, Math.round(fit * 100) / 100))
                      }
                    }
                  }
                }
              }} />
          ) : (
            <div className="flex items-center justify-center bg-bg-surface rounded-lg border border-border text-text-muted text-xs aspect-[4/3] w-full max-w-48 min-h-28">
              <div className="text-center">
                <HugeiconsIcon icon={Loading01Icon} className="size-5 mx-auto mb-1 text-accent-violet animate-spin" />
                <p>Loading document...</p>
              </div>
            </div>
          )}

          {/* SVG overlay for boxes and annotations */}
          {displayW > 0 && displayH > 0 && (
            <svg className="absolute inset-0 pointer-events-none w-full h-full"
              viewBox={`0 0 ${displayW} ${displayH}`}>
              
              {/* OCR word boxes */}
              {showBoxes && boxes && boxes.map((b, i) => {
                const [x0, y0, x1, y1] = b.box
                const isHovered = hoveredWord === b.word
                return (
                  <rect key={`ocr-${i}`}
                    x={x0} y={y0}
                    width={x1 - x0} height={y1 - y0}
                    fill={isHovered ? 'rgba(59, 130, 246, 0.35)' : 'rgba(59, 130, 246, 0.15)'}
                    stroke={isHovered ? '#60a5fa' : 'rgba(59, 130, 246, 0.6)'}
                    strokeWidth={isHovered ? 2 : 1} rx={1} />
                )
              })}

              {/* Ground truth annotation boxes (dashed) */}
              {showGT && groundTruthAnnotations && groundTruthAnnotations.map((a, i) => {
                const [x0, y0, x1, y1] = a.box
                const color = a.color || getDefaultColor(a.label)
                const isHovered = hoveredAnnotation === a
                return (
                  <rect key={`gt-${i}`}
                    x={x0} y={y0}
                    width={x1 - x0} height={y1 - y0}
                    fill={isHovered ? `${color}66` : `${color}44`}
                    stroke={color}
                    strokeWidth={isHovered ? 3 : 1.5}
                    strokeDasharray="4 2" rx={1} />
                )
              })}

              {/* Predicted annotation boxes */}
              {showPredicted && predictedAnnotations && predictedAnnotations.map((a, i) => {
                const [x0, y0, x1, y1] = a.box
                const color = a.color || getDefaultColor(a.label)
                const isHovered = hoveredAnnotation === a
                const isHighlighted = highlightedEvidence?.some(h => h.label === a.label && h.text === a.text)
                return (
                  <rect key={`pred-${i}`}
                    x={x0} y={y0}
                    width={x1 - x0} height={y1 - y0}
                    fill={isHighlighted ? 'rgba(250, 204, 21, 0.5)' : isHovered ? `${color}66` : `${color}33`}
                    stroke={isHighlighted ? '#eab308' : color}
                    strokeWidth={isHighlighted ? 3 : isHovered ? 3 : 1.5}
                    rx={1}
                    className={onSelectField ? 'cursor-pointer' : ''}
                    onClick={() => onSelectField?.(a.label)} />
                )
              })}

              {/* Highlighted evidence glow */}
              {highlightedEvidence && highlightedEvidence.map((a, i) => {
                const [x0, y0, x1, y1] = a.box
                return (
                  <rect key={`hl-${i}`}
                    x={x0 - 4} y={y0 - 4}
                    width={x1 - x0 + 8} height={y1 - y0 + 8}
                    fill="none" stroke="#eab308"
                    strokeWidth={2} rx={4}
                    strokeDasharray="6 3"
                    className="animate-pulse" />
                )
              })}
            </svg>
          )}
        </div>
      </div>

      {/* Hover tooltip - annotation */}
      {hoveredAnnotation && (
        <div className="px-4 py-2 border-t border-border bg-bg-elevated/80 animate-fade-in">
          <div className="flex items-center gap-3 text-xs">
            <span className="w-2.5 h-2.5 rounded-sm inline-block shrink-0"
              style={{ backgroundColor: hoveredAnnotation.color || getDefaultColor(hoveredAnnotation.label) }} />
            <span className="font-semibold text-text-primary">{hoveredAnnotation.label}</span>
            <span className="text-text-muted font-mono" style={{ fontFamily: "'JetBrains Mono', monospace" }}>{hoveredAnnotation.text}</span>
            <span className={`ml-auto px-1.5 py-0.5 rounded text-xs font-medium ${
              hoveredAnnotation.source === 'ground_truth' ? 'bg-accent-green/20 text-accent-green' : 'bg-accent-violet/20 text-accent-violet'
            }`}>
              {hoveredAnnotation.source === 'ground_truth' ? 'GT' : 'Predicted'}
            </span>
          </div>
        </div>
      )}

      {/* Hover tooltip - word */}
      {hoveredWord && !hoveredAnnotation && (
        <div className="px-4 py-2 bg-bg-elevated/80 border-t border-border animate-fade-in">
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded-full bg-accent-violet/10 flex items-center justify-center">
              <span className="text-[8px] text-accent-violet font-bold">i</span>
            </div>
            <span className="text-sm text-text-primary font-mono" style={{ fontFamily: "'JetBrains Mono', monospace" }}>{hoveredWord}</span>
          </div>
        </div>
      )}
    </div>
  )
}
