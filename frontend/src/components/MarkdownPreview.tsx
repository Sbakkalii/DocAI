import { useMemo } from 'react'

function escapeHtml(s: string): string {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
}

function renderInline(text: string): string {
  return escapeHtml(text)
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
}

export default function MarkdownPreview({ content }: { content: string }) {
  const html = useMemo(() => {
    const lines = content.split('\n')
    const out: string[] = []
    let inTable = false

    for (let i = 0; i < lines.length; i++) {
      const line = lines[i]

      if (/^\|.+\|$/.test(line.trim())) {
        if (!inTable) {
          inTable = true
          out.push('<table class="ocr-markdown-table">')
        }
        const isHeader = i + 1 < lines.length && /^\|[-:| ]+\|$/.test(lines[i + 1].trim())
        if (isHeader) {
          out.push('<thead><tr>')
          const cells = line.split('|').filter(c => c.trim() !== '')
          for (const cell of cells) {
            out.push(`<th>${renderInline(cell.trim())}</th>`)
          }
          out.push('</tr></thead>')
          out.push('<tbody>')
          i++ // skip separator row
          continue
        }
        out.push('<tr>')
        const cells = line.split('|').filter(c => c.trim() !== '')
        for (const cell of cells) {
          out.push(`<td>${renderInline(cell.trim())}</td>`)
        }
        out.push('</tr>')
        continue
      }

      if (inTable) {
        inTable = false
        out.push('</tbody></table>')
      }

      if (/^##\s/.test(line)) {
        out.push(`<h2 class="ocr-markdown-h2">${renderInline(line.replace(/^##\s*/, ''))}</h2>`)
      } else if (/^---+\s*$/.test(line.trim())) {
        out.push('<hr class="ocr-markdown-hr" />')
      } else if (line.trim() === '') {
        // blank line between sections
      } else {
        out.push(`<p class="ocr-markdown-p">${renderInline(line)}</p>`)
      }
    }

    if (inTable) {
      out.push('</tbody></table>')
    }

    return out.join('\n')
  }, [content])

  return (
    <div
      className="ocr-markdown-root text-sm leading-relaxed"
      dangerouslySetInnerHTML={{ __html: html }}
    />
  )
}
