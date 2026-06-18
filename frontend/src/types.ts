export interface Preset {
  id: string
  label: string
  description: string
}

export interface UploadResponse {
  session_id: string
  filename: string
  status: string
}

export interface BBox {
  word: string
  box: [number, number, number, number]
  confidence: number
}

export interface RetrievedExample {
  ocr_text: string
  fields: Record<string, string>
  source?: string
  image_path?: string
}

export interface KnowledgeGraphNode {
  id: string
  type: string
  label: string
  properties?: Record<string, unknown>
}

export interface KnowledgeGraphEdge {
  source: string
  target: string
  type: string
}

export interface KnowledgeGraph {
  nodes: KnowledgeGraphNode[]
  edges: KnowledgeGraphEdge[]
  field_traces?: Record<string, { value: string; validation_status: string }>
  statistics?: { total_nodes: number; total_edges: number; fields_traced: number }
  page_graphs?: Record<number, KnowledgeGraph>
}

export interface StepDataIngestion {
  document_type: string
  total_pages: number
  pages: Array<{ page_number: number; source_file: string; image_path: string }>
}

export interface StepDataOCR {
  pages: Array<{
    page_number: number
    text: string
    markdown: string
    word_count: number
    boxes: BBox[]
    image_width: number
    image_height: number
  }>
}

export interface StepDataVisionOCR {
  pages: Array<{
    page_number: number
    text: string
    markdown: string
    model: string
  }>
}

export interface StepDataRetrieval {
  pages: Array<{
    page_number: number
    examples: RetrievedExample[]
    num_examples: number
  }>
}

export interface StepDataRAG {
  pages: Array<{
    page_number: number
    rules: Record<string, unknown>[]
    templates: Record<string, unknown>[]
  }>
}

export interface StepDataLLMExtraction {
  pages: Array<{
    page_number: number
    fields: Record<string, string>
    prompt: string
  }>
}

export interface StepDataValidation {
  pages: Array<{
    page_number: number
    validation: ValidationResult | null
  }>
}

export interface StepDataKnowledgeGraph {
  pages: Array<{
    page_number: number
    graph: KnowledgeGraph | null
  }>
  global_graph?: KnowledgeGraph | null
}

export interface StepDataEvaluation {
  metrics: Record<string, unknown>
}

export interface ProgressMessage {
  type: 'progress'
  step: string
  status: string
  elapsed: number
  data: Record<string, unknown>
}

export interface CompletedMessage {
  type: 'completed'
  session_id: string
  elapsed: number
  result: PipelineResult
}

export interface ErrorMessage {
  type: 'error'
  session_id: string
  error: string
}

export type WSMessage = ProgressMessage | CompletedMessage | ErrorMessage

export interface StepState {
  status: 'pending' | 'running' | 'completed' | 'failed'
  elapsed: number
  data?: Record<string, unknown>
  stepIndex: number
}

export interface AnnotationBox {
  label: string
  text: string
  box: [number, number, number, number]
  confidence: number
  color: string
  source: 'ground_truth' | 'predicted'
}

export interface PageFragment {
  fragment_type: 'title' | 'text' | 'table' | 'field' | 'header' | 'footer' | 'separator' | 'other'
  content: Record<string, unknown>
  reading_order: number
}

export interface PageResult {
  page_number: number
  page_type: string | null
  page_type_confidence: number
  ocr_word_count: number
  ocr_markdown: string
  extracted_fields: Record<string, string>
  line_items: LineItem[]
  validation: ValidationResult | null
  knowledge_graph: KnowledgeGraph | null
  ocr_text: string
  vlm_text?: string
  vlm_markdown?: string
  ocr_boxes: BBox[]
  image_width: number
  image_height: number
  image_path: string
  retrieved_examples: RetrievedExample[]
  rag_rules: Record<string, unknown>[]
  rag_templates: Record<string, unknown>[]
  last_prompt: string
  ground_truth_annotations: AnnotationBox[]
  predicted_annotations: AnnotationBox[]
  extraction_evidence: Record<string, string>
  overall_confidence?: number | null
  needs_review?: boolean
  field_confidence?: Record<string, { confidence: number; level: string; needs_review: boolean; signals: { ocr_confidence: number; evidence_match: number; format_valid: number } }>
  vendor_match?: Record<string, unknown> | null
  vendor_anomalies?: Array<Record<string, unknown>>
  anomalies?: Array<Record<string, unknown>>
  page_fragments: PageFragment[]
  linked_entities: Record<string, unknown>[]
  session_id: string
  original_filename?: string
}

export interface LineItem {
  description?: string
  quantity?: string
  unit_price?: string
  total?: string
  vat_rate?: string
  sub_total?: string
  page: number
}

export interface ValidationIssue {
  rule: string
  severity: string
  message: string
  fields: string[]
}

export interface ValidationResult {
  is_valid: boolean
  issues: ValidationIssue[]
  error_count: number
  warning_count: number
}

export interface PipelineResult {
  session_id: string
  input_path: string
  document_type: string | null
  classified_type: string
  classified_confidence: number
  pages: PageResult[]
  num_pages: number
  timing: Record<string, number>
  total_time: number
  evaluation: Record<string, unknown> | null
  exports?: Record<string, unknown>
  multi_task?: Record<string, unknown>
  errors: string[]
}

export interface PipelineData {
  result: PipelineResult | null
  steps: Record<string, StepState>
  stepOrder: string[]
  currentStep: string | null
  status: string
}
