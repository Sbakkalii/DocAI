import { FIELD_RE } from './constants'

export function QAMessage({ content, evidence, onFieldClick }: { content: string; evidence?: Record<string, string>; onFieldClick?: (f: string) => void }) {
  const parts: React.ReactNode[] = []
  let last = 0
  let m: RegExpExecArray | null
  const re = new RegExp(FIELD_RE.source, 'g')
  while ((m = re.exec(content)) !== null) {
    if (m.index > last) parts.push(content.slice(last, m.index))
    const fieldName = m[1]
    const ev = evidence?.[fieldName]
    if (onFieldClick) {
      parts.push(
        <span key={m.index} className="inline-flex flex-col items-start">
          <button onClick={() => onFieldClick(fieldName)}
            className="text-yellow-400 hover:text-yellow-300 hover:underline font-medium cursor-pointer transition-colors">
            {fieldName}
          </button>
          {ev && <span className="text-[10px] text-text-muted italic leading-tight opacity-70" title={`OCR evidence: "${ev}"`}>{ev}</span>}
        </span>
      )
    } else {
      parts.push(
        <span key={m.index} className="inline-flex flex-col items-start">
          <span className="text-yellow-400 font-medium">{fieldName}</span>
          {ev && <span className="text-[10px] text-text-muted italic leading-tight opacity-70" title={`OCR evidence: "${ev}"`}>{ev}</span>}
        </span>
      )
    }
    last = re.lastIndex
  }
  if (last < content.length) parts.push(content.slice(last))
  if (!onFieldClick && !evidence) return <>{content}</>
  return <>{parts}</>
}
