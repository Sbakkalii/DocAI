import type { BBox, AnnotationBox } from '../types'

export const STEP_LABELS: Record<string, string> = {
  ingestion: 'Document Ingestion',
  ocr: 'OCR Text Extraction',
  vision_ocr: 'Vision OCR (VLM)',
  hybrid_ocr: 'Hybrid OCR',
  document_graph: 'Document Graph',
  end_to_end_vlm: 'End-to-End VLM',
  document_classifier: 'Document Type Classifier',
  embedding: 'Layout Embedding',
  retrieval: 'Example Retrieval',
  rag: 'RAG Context Building',
  llm_extraction: 'LLM Field Extraction',
  validation: 'Field Validation',
  confidence_scoring: 'Confidence Scoring',
  export: 'ERP Export',
  vendor_lookup: 'Vendor Lookup',
  anomaly: 'Anomaly Detection',
  multi_task: 'Multi-Task NLP',
  review: 'Human Review',
  cross_page: 'Cross-Page Resolution',
  knowledge_graph: 'Knowledge Graph',
  evaluation: 'Evaluation',
}

export const STEP_INFO: Record<string, { what: string; how: string; expected: string }> = {
  ingestion: {
    what: 'Loads the document, splits into pages, validates format, extracts metadata.',
    how: 'Reads image/PDF files, creates PageResult objects with image paths and metadata. Detects page count and document type.',
    expected: 'A list of pages ready for downstream processing.',
  },
  ocr: {
    what: 'Extracts text and bounding boxes from document images using PaddleOCR or RapidOCR.',
    how: 'Runs PaddleOCR (CPU) which detects words and their spatial positions (x0,y0,x1,y1 boxes) with confidence scores. Optional LLM post-correction fixes spacing artifacts.',
    expected: 'OCRResult with words, bounding boxes, confidences — used by to_markdown() for spatial layout reconstruction.',
  },
  vision_ocr: {
    what: 'VLM-based text extraction using gemma3:4b as an alternative OCR engine.',
    how: 'Sends the image to a vision-language model (gemma3:4b via Ollama) which reads text directly from the visual input. Outputs raw text with line breaks preserved. Post-correction fixes spacing.',
    expected: 'Plain text with approximate layout. No bounding boxes. Less spatial accuracy than PaddleOCR but better at reading damaged or complex layouts.',
  },
  hybrid_ocr: {
    what: 'Combines PaddleOCR spatial accuracy with VLM text quality.',
    how: 'Runs PaddleOCR (for bounding box layout) AND VLM OCR (for text quality), then overlays VLM-corrected words onto PaddleOCR\'s bounding boxes. Uses PaddleOCR\'s to_markdown() for perfect pipe-table reconstruction.',
    expected: 'Best of both worlds: clean text from VLM + exact table structure from PaddleOCR boxes. Ideal for documents with tables.',
  },
  document_graph: {
    what: 'Builds a spatial graph from OCR bounding boxes for advanced layout analysis.',
    how: 'Each word becomes a graph node. Edges encode spatial relationships (right_of, below, same_column). Table regions detected by aligned node groups. Key-value pairs found by proximity of label-like text (e.g., "Total:" → "3463,20").',
    expected: 'A document graph with nodes, edges, detected tables, and KV pairs. Markdown reconstructed by traversing the graph top-to-bottom, left-to-right.',
  },
  end_to_end_vlm: {
    what: 'Direct image-to-field VLM extraction. OCR runs separately for confidence scoring and RAG context.',
    how: 'Sends the raw invoice image to gemma3:4b with a prompt asking for JSON output of all invoice fields. VLM output text feeds validation and document classification.',
    expected: 'Extracted fields directly from the VLM. OCR runs alongside for confidence evidence and post-results RAG analysis.',
  },
  document_classifier: {
    what: 'Classifies each page type and the overall document type, then auto-selects relevant extraction fields.',
    how: 'Per-page keyword scoring (invoice/contract/report/correspondence/form/other) sets page_type. Document-level majority vote + extended keyword search detects invoice, contract, purchase_order, delivery_note, bank_statement, or id_card. Routes target_fields and RAG templates based on detected type.',
    expected: 'Per-page types + document type string + confidence. Downstream extraction uses type-specific field set.',
  },
  embedding: {
    what: 'Computes text embeddings using E5-small multilingual model.',
    how: 'Runs sentence-transformers with intfloat/multilingual-e5-small (384-d). Stores embeddings in TurboVec 4-bit index for fast approximate search.',
    expected: '384-dimensional embedding vectors per page, indexed for retrieval.',
  },
  retrieval: {
    what: 'Retrieves similar invoice examples using hybrid dense+sparse search.',
    how: 'Dense: cosine similarity via TurboVec (E5 embeddings). Sparse: BM25Okapi with French/English stop words. RRF fusion combines both rankings into a unified result list.',
    expected: 'Top-K similar invoices with their OCR text and extracted fields for few-shot prompting.',
  },
  rag: {
    what: 'Retrieves field rules and template hints from the knowledge base.',
    how: 'Keyword matching against FieldRulesStore (11 field rules + 2 invoice templates). Returns field descriptions, format patterns, layout hints, and template metadata with bilingual (fr/en) support.',
    expected: 'FieldRule and TemplateHint objects injected into the LLM extraction prompt.',
  },
  llm_extraction: {
    what: 'Extracts structured invoice fields using an LLM with few-shot + RAG prompting.',
    how: 'Builds a prompt with: detected language context, French field synonyms, target field list, few-shot examples from retrieval, RAG rules with locale-aware descriptions, and OCR/VLM text. LLM (llama3.2:3b) outputs JSON with _evidence key for traceability.',
    expected: 'JSON dict of 11 target fields (NUMBER, SUPPLIER, ADDRESS, INVOICE_DATE, TOTAL, TOTAL_AMOUNT, LINE/*). Evidence citations for each field value.',
  },
  validation: {
    what: 'Validates extracted fields against business rules.',
    how: '6 configurable checks: required fields (NUMBER, SUPPLIER, TOTAL, DATE), arithmetic (TOTAL ≈ sum of line subtotals), format (date patterns), ranges (no negative totals), currency consistency, OCR evidence (token overlap with source text).',
    expected: 'ValidationResult with is_valid flag, list of issues (error/warning severity), error_count, warning_count.',
  },
  confidence_scoring: {
    what: 'Computes calibrated per-field confidence using a weighted 3-signal score — replaces unreliable LLM self-reported confidence.',
    how: 'Signal 1 (0.4): Average OCR word confidence over evidence text span. Signal 2 (0.4): Fuzzy string match between evidence text and extracted value. Signal 3 (0.2): Format validation pass/fail (dates, amounts, IBAN, identifiers). Fields with overall < 0.70 are flagged for human review.',
    expected: 'Per-field confidence scores with sub-signal breakdown, overall document confidence, needs_review flag.',
  },
  export: {
    what: 'Exports extracted fields to ERP-compatible formats: UBL 2.1 XML (European e-invoice standard), EDI 810 (US invoice), and configurable CSV.',
    how: 'Gathers all extracted fields across pages, reconstructs line items from LINE/* fields, maps to UBL/EDI XML schemas or configurable CSV columns via data/field_map.yaml.',
    expected: 'Exported files on disk (export_ubl_xml.xml, export_edi810.txt, export_csv.csv) with format selector.',
  },
  vendor_lookup: {
    what: 'Looks up extracted supplier in a local vendor registry, pre-fills known fields (address, IBAN), and flags mismatches as potential fraud signals.',
    how: 'Fuzzy string match against SQLite vendors.db registry (threshold 0.80). Pre-fills validated fields with 100% confidence when sourced from registry. Detects IBAN/VAT mismatches and unknown suppliers.',
    expected: 'Vendor match with confidence score, pre-filled fields, list of anomalies (mismatches or unknown supplier flags).',
  },
  anomaly: {
    what: 'Statistical anomaly & fraud detection with 4 check types: duplicate invoice, amount outlier, VAT rate validation, date sanity.',
    how: 'Duplicate: same vendor+amount+date within 30 days. Amount: total > 3σ from vendor historical average (needs 5+ invoices). VAT: implied rate not in allowed set. Date: invoice date in the future. Results persisted in SQLite for historical tracking.',
    expected: 'List of detected anomalies with severity (error/warning) and explanations.',
  },
  multi_task: {
    what: 'Configurable multi-task NLP: NER, summarization, contract KIE, and clause risk scoring. Select which tasks to run in the step config popover.',
    how: 'Calls the selected VLM or LLM model with task-specific prompts on gathered document text. Each task is independent and produces structured JSON.',
    expected: 'Per-task structured JSON: NER entities, summary bullets, contract clause dicts, risk scores. Available tasks can be toggled before running.',
  },
  review: {
    what: 'Allows a human reviewer to inspect, correct, and approve extracted fields before final output.',
    how: 'Displays all extracted fields with inline editing. Each correction is posted to /api/correct/{session_id} which persists to job._corrections, updates in-memory fields, and writes corrections.json to disk.',
    expected: 'Corrected field values stored on the backend, reflected in subsequent pipeline re-runs and the QA system.',
  },
  cross_page: {
    what: 'Resolves entities and references across multiple pages.',
    how: '3 sub-checks: table_merge (collects line items across pages), entity_link (links supplier/address across pages), reference_resolve (finds common fields between page pairs).',
    expected: 'Cross-page results with linked entities, common fields, merged line item counts.',
  },
  knowledge_graph: {
    what: 'Builds a traceability graph mapping extracted fields to validation status.',
    how: 'Creates graph nodes for each extracted field with properties (value, validation_status). Supports page-level or document-level scope.',
    expected: 'Graph with nodes (extracted fields), field_traces (field → value + validation status), and statistics.',
  },
  table_extraction: {
    what: 'Extracts structured table data (line items) from OCR results.',
    how: 'Parses OCR markdown tables and spatial layout to identify line item rows with description, quantity, unit price, and subtotal columns.',
    expected: 'Structured line_items array with per-row column values.',
  },
  evaluation: {
    what: 'Computes accuracy and faithfulness metrics against ground truth TSV.',
    how: '3 metrics: accuracy (per-field exact match + Token F1 vs GT annotations), faithfulness (token overlap between extracted values and OCR/VLM source text), confidence (per-field confidence from validation). Auto-discovers GT via find_annotation_file().',
    expected: 'Accuracy score, faithfulness score, confidence score with per-field breakdown. Ground truth TSV required.',
  },
}

export const MULTI_TASK_TASK_OPTIONS: Array<{id: string; label: string; description: string}> = [
  { id: 'ner', label: 'Named Entity Recognition', description: 'Extract persons, organizations, dates, amounts, locations, and identifiers.' },
  { id: 'summarization', label: 'Summarization', description: '3 bullet-point key facts + 1-paragraph executive summary.' },
  { id: 'contract_kie', label: 'Contract KIE', description: 'Extract payment terms, termination, liability caps, jurisdiction, confidentiality.' },
  { id: 'clause_risk', label: 'Clause Risk Scoring', description: 'Score each clause as Standard / Non-standard / High-risk with explanation.' },
]

export const STEP_ORDER = Object.keys(STEP_LABELS)

export const STEP_GROUPS: { label: string; steps: string[] }[] = [
  { label: 'Preprocessing', steps: ['ingestion'] },
  { label: 'Extraction & Validation', steps: ['end_to_end_vlm', 'document_classifier', 'llm_extraction', 'validation', 'confidence_scoring'] },
  { label: 'Results', steps: ['export', 'vendor_lookup', 'anomaly'] },
  { label: 'Context Engineering (RAG)', steps: ['ocr', 'embedding', 'retrieval', 'rag'] },
  { label: 'Evaluation', steps: ['evaluation', 'multi_task'] },
  { label: 'Advanced (Hybrid)', steps: ['vision_ocr', 'hybrid_ocr', 'document_graph', 'table_extraction', 'cross_page', 'knowledge_graph'] },
]

export function getPreprocSteps(mode: string): string[] {
  if (mode === 'end_to_end') return ['ingestion']
  if (mode === 'graph') return ['ingestion', 'document_graph', 'table_extraction']
  return ['ingestion', 'hybrid_ocr', 'table_extraction']
}

export const MODE_STEP_ALLOWLIST: Record<string, Set<string>> = {
  hybrid: new Set(['ingestion', 'hybrid_ocr', 'document_classifier', 'embedding', 'retrieval', 'rag', 'llm_extraction', 'validation', 'confidence_scoring', 'export', 'vendor_lookup', 'anomaly', 'multi_task', 'evaluation', 'table_extraction']),
  graph: new Set(['ingestion', 'document_graph', 'document_classifier', 'embedding', 'retrieval', 'rag', 'llm_extraction', 'validation', 'confidence_scoring', 'export', 'vendor_lookup', 'anomaly', 'multi_task', 'evaluation', 'table_extraction']),
  end_to_end: new Set(['ingestion', 'document_classifier', 'end_to_end_vlm', 'validation', 'confidence_scoring', 'export', 'vendor_lookup', 'anomaly', 'multi_task', 'evaluation']),
}

export function getEnabledSteps(mode: string): string[] {
  const opt = MODE_OPTIONS.find(m => m.value === mode)
  if (opt?.disabled) return []
  return [...(MODE_STEP_ALLOWLIST[mode] || MODE_STEP_ALLOWLIST.hybrid)]
}

export function fmtTime(seconds: number): string {
  if (seconds < 1) return `${(seconds * 1000).toFixed(0)}ms`
  if (seconds < 60) return `${seconds.toFixed(2)}s`
  const m = Math.floor(seconds / 60)
  const s = seconds % 60
  return s > 0 ? `${m}m ${s.toFixed(1)}s` : `${m}m`
}

export const DEFAULT_FIELDS = [
  'NUMBER', 'SUPPLIER', 'ADDRESS', 'INVOICE_DATE',
  'TOTAL', 'TOTAL_AMOUNT',
  'LINE/DESCRIPTION', 'LINE/QUANTITY', 'LINE/UOM',
  'LINE/UNIT_PRICE', 'LINE/SUB_TOTAL',
]

export const ALL_FIELDS = [
  'NUMBER', 'SUPPLIER', 'ADDRESS', 'INVOICE_DATE',
  'TOTAL', 'TOTAL_AMOUNT',
  'LINE/DESCRIPTION', 'LINE/QUANTITY', 'LINE/UOM',
  'LINE/UNIT_PRICE', 'LINE/SUB_TOTAL',
  'CONTRACT_DATE', 'PARTIES', 'EFFECTIVE_DATE',
  'TERMINATION_CLAUSE', 'SIGNATORY', 'CONTRACT_VALUE',
  'SCOPE_OF_WORK', 'GOVERNING_LAW',
  'PO_NUMBER', 'ORDER_DATE', 'DELIVERY_DATE', 'SHIPPING_ADDRESS',
  'LINE/TOTAL',
  'DN_NUMBER', 'RECEIVER_NAME', 'SIGNATURE',
  'ACCOUNT_NUMBER', 'STATEMENT_DATE', 'OPENING_BALANCE',
  'CLOSING_BALANCE', 'BANK_NAME', 'IBAN',
  'DOCUMENT_ID', 'FULL_NAME', 'DATE_OF_BIRTH', 'NATIONALITY',
  'EXPIRY_DATE', 'DOCUMENT_NUMBER', 'GENDER', 'PLACE_OF_BIRTH',
]

export const FIELD_CATEGORIES = [
  {
    label: 'Invoice',
    fields: ['NUMBER', 'SUPPLIER', 'ADDRESS', 'INVOICE_DATE', 'TOTAL', 'TOTAL_AMOUNT',
             'LINE/DESCRIPTION', 'LINE/QUANTITY', 'LINE/UOM', 'LINE/UNIT_PRICE', 'LINE/SUB_TOTAL'],
  },
  {
    label: 'Contract',
    fields: ['CONTRACT_DATE', 'PARTIES', 'EFFECTIVE_DATE', 'TERMINATION_CLAUSE', 'SIGNATORY',
             'CONTRACT_VALUE', 'SCOPE_OF_WORK', 'GOVERNING_LAW'],
  },
  {
    label: 'Purchase Order',
    fields: ['PO_NUMBER', 'ORDER_DATE', 'DELIVERY_DATE', 'SHIPPING_ADDRESS', 'LINE/TOTAL'],
  },
  {
    label: 'Delivery Note',
    fields: ['DN_NUMBER', 'RECEIVER_NAME', 'SIGNATURE'],
  },
  {
    label: 'Bank Statement',
    fields: ['ACCOUNT_NUMBER', 'STATEMENT_DATE', 'OPENING_BALANCE', 'CLOSING_BALANCE', 'BANK_NAME', 'IBAN'],
  },
  {
    label: 'ID Card',
    fields: ['DOCUMENT_ID', 'FULL_NAME', 'DATE_OF_BIRTH', 'NATIONALITY', 'EXPIRY_DATE',
             'DOCUMENT_NUMBER', 'GENDER', 'PLACE_OF_BIRTH'],
  },
]

export const AVAILABLE_MODELS = [
  'qwen2.5:3b',
  'qwen2.5:7b',
  'qwen2.5:14b',
  'llama3.2:3b',
  'llama3.1:8b',
  'mistral:7b',
  'deepseek-r1:8b',
  'phi4:14b',
]

export const AVAILABLE_VLM_MODELS = [
  'gemma3:4b',
  'qwen2.5vl:3b',
  'deepseek-ocr:latest',  
]

export const DOCUMENT_TYPE_MODELS: Record<string, { recommended: string; reason: string }> = {
  invoice: { recommended: 'qwen2.5:7b', reason: 'Best for structured tabular data & numeric precision' },
  contract: { recommended: 'deepseek-r1:8b', reason: 'Reasoning needed for legal clauses & long text' },
  purchase_order: { recommended: 'qwen2.5:7b', reason: 'Same structure as invoices with product line items' },
  delivery_note: { recommended: 'qwen2.5:3b', reason: 'Simple documents with few fields, fast extraction' },
  bank_statement: { recommended: 'qwen2.5:7b', reason: 'Numeric precision for balances & transaction data' },
  id_card: { recommended: 'qwen2.5:3b', reason: 'Simple key-value pairs, fast extraction sufficient' },
}

export const MODE_OPTIONS = [
  { value: 'end_to_end', label: 'End-to-End VLM', desc: 'Image → fields directly via VLM', disabled: false },
  { value: 'hybrid', label: 'Hybrid OCR + LLM', desc: 'OCR + retrieval + RAG + LLM extraction', disabled: true },
  { value: 'graph', label: 'Graph-Based', desc: 'Spatial document graph + LLM extraction', disabled: true },
]

export const MODE_INFO: Record<string, { what: string; how: string; expected: string }> = {
  end_to_end: {
    what: 'Image → fields directly via a vision-language model.',
    how: 'Sends the raw invoice image to a VLM (gemma3:4b, qwen2.5vl:3b, deepseek-ocr:3b) with a prompt asking for JSON output of all invoice fields. Bypasses OCR, embedding, retrieval, and RAG entirely.',
    expected: 'Extracted fields directly from the VLM with field name normalization and target field ordering applied.',
  },
  hybrid: {
    what: 'Combines PaddleOCR spatial accuracy with VLM text quality, then extracts fields via LLM.',
    how: 'Runs PaddleOCR (bounding boxes) AND VLM OCR (text quality), merges them, then runs embedding → retrieval → RAG → LLM extraction (qwen2.5:7b). Uses few-shot examples from the vector DB.',
    expected: 'Best accuracy for complex invoices: clean text from VLM, exact table structure from PaddleOCR, and LLM field extraction with RAG context.',
  },
  graph: {
    what: 'Builds a spatial document graph from OCR boxes, then extracts fields via LLM.',
    how: 'Each word from PaddleOCR becomes a graph node with spatial relationships (right_of, below). Detects tables and key-value pairs. Converts to markdown, then runs embedding → retrieval → RAG → LLM extraction.',
    expected: 'Graph-aware text structure that preserves document layout. Useful for non-tabular documents where spatial relationships matter.',
  },
}

export const MODE_OCR_MODEL: Record<string, string> = {
  hybrid: 'PaddleOCR + gemma3:4b VLM',
  graph: 'PaddleOCR (spatial graph)',
  end_to_end: 'gemma3:4b VLM',
}

export const DEFAULT_QA_PROMPT = `You are a precise document QA assistant. Your job is to answer questions using ONLY the extracted fields and OCR text provided below.

## Grounding Rules (CRITICAL)
1. ALWAYS append the FIELD NAME in ALL CAPS in parentheses after every value you cite.
   ✓ Correct: "The total is 24,120.00 (TOTAL)"
   ✓ Correct: "The supplier is ACME Corp (SUPPLIER)"
   ✗ Wrong: "The total is 24,120.00"
2. If evidence (OCR text span) is available, briefly reference it.
   Example: "The total is 24,120.00 (TOTAL) found as '24120.00' in the document."
3. If validation issues exist for a field you cite, mention them.
   Example: "The total is 24,120.00 (TOTAL) — note: arithmetic check shows discrepancy."
4. If you're unsure or data is missing, say so. Never fabricate field names.
5. For "are you sure?" or confidence questions, reference:
   - The validation status: "TOTAL passed arithmetic check (TOTAL)"
   - The evidence text: "found as '24120.00' in the OCR text"
6. Answer in the user's language. Be concise but conversational when appropriate.`

export const METRIC_INFO: Record<string, { label: string; desc: string; category: string }> = {
  accuracy: {
    label: 'Accuracy',
    desc: 'Exact match rate against ground-truth annotations. After OCR normalization and numeric equivalence, each field is scored as exact (1) or not (0). Green ≥80%, Yellow ≥50%, Red <50%.',
    category: 'exact_match',
  },
  faithfulness: {
    label: 'Faithfulness',
    desc: 'Rate of extracted values that appear in the source OCR/VLM text. Measures whether the LLM invented values (hallucination) or faithfully extracted them. Higher is better.',
    category: 'source_grounding',
  },
  token_f1: {
    label: 'Token F1',
    desc: 'Partial credit score: token-level overlap between prediction and ground truth. Accounts for near-misses (e.g. "ACME Corp." vs "ACME Corporation"). Computed via token-set intersection or Levenshtein for single tokens.',
    category: 'partial_credit',
  },
  numeric_delta: {
    label: 'Numeric Δ',
    desc: 'Tolerance-scaled closeness of numeric values (totals, amounts, quantities). Score = 1 - |pred - gt| / |gt|, capped at 0. 1.0 = exact, 0.0 = ≥100% error.',
    category: 'numeric',
  },
  format_compliance: {
    label: 'Format',
    desc: 'Rate of fields meeting expected format rules: IBAN check digit validation, parseable dates, identifier patterns (alphanumeric), account number patterns. Independent of ground truth.',
    category: 'format',
  },
  detection_rate: {
    label: 'Detection',
    desc: 'Rate of non-empty field extractions for binary presence fields (e.g. SIGNATURE). Measures whether expected fields were detected at all, regardless of value correctness.',
    category: 'detection',
  },
}

export const CATEGORY_METRICS: Record<string, string[]> = {
  invoice: ['accuracy', 'faithfulness', 'token_f1', 'numeric_delta', 'format_compliance'],
  contract: ['accuracy', 'faithfulness', 'token_f1', 'numeric_delta', 'format_compliance'],
  purchase_order: ['accuracy', 'faithfulness', 'token_f1', 'numeric_delta', 'format_compliance'],
  delivery_note: ['accuracy', 'faithfulness', 'token_f1', 'detection_rate', 'format_compliance'],
  bank_statement: ['accuracy', 'faithfulness', 'token_f1', 'numeric_delta', 'format_compliance'],
  id_card: ['accuracy', 'faithfulness', 'token_f1', 'format_compliance'],
}

export const STEP_PREREQS: Record<string, string[]> = {
  ingestion: [],
  vision_ocr: ['ingestion'],
  ocr: ['ingestion'],
  hybrid_ocr: ['ingestion'],
  document_graph: ['ingestion'],
  end_to_end_vlm: ['ingestion'],
  document_classifier: ['ingestion'],
  table_extraction: ['ocr', 'hybrid_ocr'],
  embedding: ['document_classifier'],
  retrieval: ['embedding'],
  rag: ['retrieval'],
  llm_extraction: ['rag', 'document_classifier'],
  validation: ['llm_extraction'],
  evaluation: ['llm_extraction'],
  cross_page: ['llm_extraction'],
  knowledge_graph: ['validation', 'cross_page'],
}

export function getDownstream(stepName: string): string[] {
  const visited = new Set<string>()
  const collect = (name: string) => {
    for (const [s, prereqs] of Object.entries(STEP_PREREQS)) {
      if (prereqs.includes(name) && !visited.has(s)) {
        visited.add(s)
        collect(s)
      }
    }
  }
  collect(stepName)
  return [...visited]
}

export const FIELD_RE = /\b([A-Z][A-Z_]+)\b/g

export function findFieldInOcr(fieldName: string, extractedFields: Record<string, string> | undefined, ocrBoxes: BBox[] | undefined): AnnotationBox[] {
  if (!extractedFields || !ocrBoxes?.length) return []
  const value = extractedFields[fieldName]
  if (!value) return []
  const valStr = String(value).toLowerCase().trim()
  if (!valStr) return []

  const joinedText = ocrBoxes.map(b => b.word).join(' ').toLowerCase()

  const idx = joinedText.indexOf(valStr)
  if (idx >= 0) {
    let charPos = 0
    const matched: BBox[] = []
    for (const box of ocrBoxes) {
      const wordLen = box.word.length
      const start = charPos
      const end = charPos + wordLen
      if (start >= idx && end <= idx + valStr.length) {
        matched.push(box)
      }
      charPos = end + 1
      if (charPos > idx + valStr.length) break
    }
    if (matched.length > 0) {
      return [{
        label: fieldName,
        text: value,
        box: [
          Math.min(...matched.map(b => b.box[0])),
          Math.min(...matched.map(b => b.box[1])),
          Math.max(...matched.map(b => b.box[2])),
          Math.max(...matched.map(b => b.box[3])),
        ],
        confidence: 0.5,
        color: '#eab308',
        source: 'predicted' as const,
      }]
    }
  }

  const valDigits = valStr.replace(/[^\d]/g, '')
  if (valDigits) {
    const fieldDigits = joinedText.replace(/[^\d]/g, '')
    const digitIdx = fieldDigits.indexOf(valDigits)
    if (digitIdx >= 0) {
      let charPos = 0
      let matched: BBox[] = []
      for (const box of ocrBoxes) {
        const stripped = box.word.toLowerCase().replace(/[^\d]/g, '')
        if (stripped) {
          const wordStart = fieldDigits.indexOf(stripped, charPos)
          const wordEnd = wordStart + stripped.length
          if (wordStart >= digitIdx && wordEnd <= digitIdx + valDigits.length) {
            matched.push(box)
          }
          charPos = wordStart + stripped.length
        } else {
          charPos += box.word.length + 1
        }
        if (charPos > digitIdx + valDigits.length) break
      }
      if (matched.length > 0) {
        return [{
          label: fieldName,
          text: value,
          box: [
            Math.min(...matched.map(b => b.box[0])),
            Math.min(...matched.map(b => b.box[1])),
            Math.max(...matched.map(b => b.box[2])),
            Math.max(...matched.map(b => b.box[3])),
          ],
          confidence: 0.5,
          color: '#eab308',
          source: 'predicted' as const,
        }]
      }
    }
  }

  return []
}
