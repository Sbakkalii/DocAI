import { useRef, useEffect, useState } from 'react'
import type { KnowledgeGraph as KG } from '../types'
import { HugeiconsIcon } from "@hugeicons/react"
import { InformationCircleIcon } from "@hugeicons/core-free-icons"

interface Props {
  graph: KG | null
  height?: number
}

interface Position {
  x: number
  y: number
  vx: number
  vy: number
}

const COLORS: Record<string, string> = {
  extracted_field: '#0099DD',
  ocr_token: '#FF6B35',
  page: '#0D9488',
  default: '#64748b',
}

const EDGE_COLORS: Record<string, string> = {
  extracted_from: '#0099DD',
  default: '#94A3B8',
}

export default function KnowledgeGraphView({ graph, height = 400 }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const animRef = useRef<number>(0)
  const [selectedNode, setSelectedNode] = useState<string | null>(null)
  const [hoveredNode, setHoveredNode] = useState<string | null>(null)
  const posRef = useRef<Map<string, Position>>(new Map())
  const stats = graph?.statistics

  useEffect(() => {
    const g = graph
    if (!g || !g.nodes.length) return
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const rect = canvas.getBoundingClientRect()
    const dpr = window.devicePixelRatio || 1
    canvas.width = rect.width * dpr
    canvas.height = height * dpr
    ctx.scale(dpr, dpr)
    const W = rect.width
    const H = height

    const positions = posRef.current
    const centerX = W / 2
    const centerY = H / 2

    if (positions.size === 0) {
      g.nodes.forEach((n, i) => {
        const angle = (2 * Math.PI * i) / g.nodes.length
        const radius = Math.min(W, H) * 0.32
        positions.set(n.id, {
          x: centerX + radius * Math.cos(angle),
          y: centerY + radius * Math.sin(angle),
          vx: 0,
          vy: 0,
        })
      })
    }

    const adj = new Map<string, Set<string>>()
    g.nodes.forEach(n => adj.set(n.id, new Set()))
    g.edges.forEach(e => {
      adj.get(e.source)?.add(e.target)
      adj.get(e.target)?.add(e.source)
    })

    let running = true
    const REPULSION = 8000
    const ATTRACTION = 0.003
    const DAMPING = 0.9
    const ITERATIONS = 120

    function simulate() {
      for (let iter = 0; iter < ITERATIONS && running; iter++) {
        const entries = Array.from(positions.entries())

        for (let i = 0; i < entries.length; i++) {
          for (let j = i + 1; j < entries.length; j++) {
            const [, posA] = entries[i]
            const [, posB] = entries[j]
            let dx = posB.x - posA.x
            let dy = posB.y - posA.y
            const dist = Math.sqrt(dx * dx + dy * dy) || 1
            const force = REPULSION / (dist * dist)
            const fx = (dx / dist) * force
            const fy = (dy / dist) * force
            posA.vx -= fx
            posA.vy -= fy
            posB.vx += fx
            posB.vy += fy
          }
        }

        g!.edges.forEach(e => {
          const pa = positions.get(e.source)
          const pb = positions.get(e.target)
          if (!pa || !pb) return
          const dx = pb.x - pa.x
          const dy = pb.y - pa.y
          const dist = Math.sqrt(dx * dx + dy * dy) || 1
          const force = ATTRACTION * dist
          pa.vx += (dx / dist) * force
          pa.vy += (dy / dist) * force
          pb.vx -= (dx / dist) * force
          pb.vy -= (dy / dist) * force
        })

        entries.forEach(([, pos]) => {
          pos.vx += (centerX - pos.x) * 0.001
          pos.vy += (centerY - pos.y) * 0.001
        })

        entries.forEach(([, pos]) => {
          pos.vx *= DAMPING
          pos.vy *= DAMPING
          pos.x += pos.vx
          pos.y += pos.vy
          pos.x = Math.max(10, Math.min(W - 10, pos.x))
          pos.y = Math.max(10, Math.min(H - 10, pos.y))
        })
      }

      if (running) draw()
    }

    function draw() {
      if (!ctx || !running) return
      ctx.clearRect(0, 0, W, H)

      const bgGrad = ctx.createLinearGradient(0, 0, W, H)
      bgGrad.addColorStop(0, '#0F0A1E')
      bgGrad.addColorStop(1, '#1A1035')
      ctx.fillStyle = bgGrad
      ctx.fillRect(0, 0, W, H)

      g!.edges.forEach(e => {
        const pa = positions.get(e.source)
        const pb = positions.get(e.target)
        if (!pa || !pb) return
        ctx.beginPath()
        ctx.moveTo(pa.x, pa.y)
        ctx.lineTo(pb.x, pb.y)
        const isActive = selectedNode && (e.source === selectedNode || e.target === selectedNode)
        ctx.strokeStyle = isActive ? '#0099DD' : (EDGE_COLORS[e.type] || EDGE_COLORS.default)
        ctx.lineWidth = isActive ? 2 : 1
        ctx.stroke()
      })

      g!.nodes.forEach(n => {
        const pos = positions.get(n.id)
        if (!pos) return
        const isSelected = selectedNode === n.id
        const isHovered = hoveredNode === n.id
        const isConnected = selectedNode && adj.get(selectedNode)?.has(n.id)

        const radius = n.type === 'extracted_field' ? 6 : 4
        const color = COLORS[n.type] || COLORS.default

        if (isSelected || isHovered) {
          ctx.beginPath()
          ctx.arc(pos.x, pos.y, radius + 4, 0, Math.PI * 2)
          ctx.fillStyle = `${color}33`
          ctx.fill()
        }

        ctx.beginPath()
        ctx.arc(pos.x, pos.y, isSelected || isHovered ? radius + 2 : radius, 0, Math.PI * 2)
        ctx.fillStyle = isSelected ? color : isHovered ? '#0099DD' : (isConnected ? '#0099DD' : color)
        ctx.fill()

        if (isSelected || isHovered) {
          ctx.strokeStyle = '#F8FAFC'
          ctx.lineWidth = 2
          ctx.stroke()
        }

        if (n.type === 'extracted_field' || isHovered || isConnected) {
          ctx.fillStyle = '#94A3B8'
          ctx.font = '10px Geist Variable, system-ui, sans-serif'
          ctx.textAlign = 'center'
          ctx.fillText(n.label, pos.x, pos.y - radius - 6)
        }
      })

      animRef.current = requestAnimationFrame(draw)
    }

    simulate()

    return () => {
      running = false
      cancelAnimationFrame(animRef.current)
    }
  }, [graph, selectedNode, hoveredNode, height])

  function handleCanvasClick(e: React.MouseEvent<HTMLCanvasElement>) {
    if (!graph || !canvasRef.current) return
    const rect = canvasRef.current.getBoundingClientRect()
    const x = e.clientX - rect.left
    const y = e.clientY - rect.top
    const positions = posRef.current

    let found: string | null = null
    graph.nodes.forEach(n => {
      const pos = positions.get(n.id)
      if (!pos) return
      const dx = x - pos.x
      const dy = y - pos.y
      if (Math.sqrt(dx * dx + dy * dy) < 10) {
        found = n.id
      }
    })
    setSelectedNode(found === selectedNode ? null : found)
  }

  function handleCanvasMove(e: React.MouseEvent<HTMLCanvasElement>) {
    if (!graph || !canvasRef.current) return
    const rect = canvasRef.current.getBoundingClientRect()
    const x = e.clientX - rect.left
    const y = e.clientY - rect.top
    const positions = posRef.current

    let found: string | null = null
    for (const n of graph.nodes) {
      const pos = positions.get(n.id)
      if (!pos) continue
      const dx = x - pos.x
      const dy = y - pos.y
      if (Math.sqrt(dx * dx + dy * dy) < 10) {
        found = n.id
        break
      }
    }
    setHoveredNode(found)
  }

  const selectedNodeData = selectedNode && graph
    ? graph.nodes.find(n => n.id === selectedNode)
    : null

  return (
    <div className="bg-bg-surface rounded-2xl border border-border overflow-hidden">
      <div className="flex items-center justify-between px-4 py-2.5 border-b border-border">
        <div className="flex items-center gap-2">
          <div className="w-5 h-5 rounded bg-accent-violet/10 flex items-center justify-center">
            <svg className="w-3 h-3 text-accent-violet" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1" /></svg>
          </div>
          <span className="text-xs font-semibold text-text-primary">Knowledge Graph</span>
        </div>
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-sm bg-accent-violet inline-block" />
            <span className="text-[13px] text-text-muted">Field</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-sm bg-accent-coral inline-block" />
            <span className="text-[13px] text-text-muted">OCR Token</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-sm bg-[#0D9488] inline-block" />
            <span className="text-[13px] text-text-muted">Page</span>
          </div>
          {stats && (
            <span className="text-xs text-text-muted tabular-nums font-medium ml-2">
              {stats.total_nodes} nodes · {stats.total_edges} edges
            </span>
          )}
        </div>
      </div>

      <canvas
        ref={canvasRef}
        className="w-full cursor-pointer"
        style={{ height }}
        onClick={handleCanvasClick}
        onMouseMove={handleCanvasMove}
        onMouseLeave={() => setHoveredNode(null)}
      />

      {selectedNodeData && (
        <div className="px-4 py-2.5 bg-bg-elevated/80 border-t border-border animate-fade-in">
          <div className="flex items-start gap-2">
            <HugeiconsIcon icon={InformationCircleIcon} className="size-4 text-accent-violet mt-0.5 shrink-0" />
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2">
                <span className="text-sm font-semibold text-text-primary">{selectedNodeData.label}</span>
                <span className="inline-flex items-center gap-1 px-2 py-0.5 text-xs font-medium rounded-full bg-bg-elevated text-text-muted">{selectedNodeData.type}</span>
              </div>
              {selectedNodeData.properties && (
                <pre className="mt-1 text-[13px] text-text-muted font-mono whitespace-pre-wrap" style={{ fontFamily: "'JetBrains Mono', monospace" }}>
                  {JSON.stringify(selectedNodeData.properties, null, 1)}
                </pre>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
