import re
import time
import unicodedata
from typing import Any

from pipeline.annotation_utils import find_annotation_file, load_ground_truth
from pipeline.base import BaseStep, PipelineContext
from pipeline.config import PipelineConfig


class EvaluationStep(BaseStep):
    name = "evaluation"
    description = "Compute evaluation metrics against ground truth"

    # Map GT annotation labels to LLM extraction field names when they differ
    GT_TO_LLM = {
        "LINE/PRICE": "LINE/UNIT_PRICE",
        "TOTAL_UNTAXED": "TOTAL",
    }

    CURRENCY_SYMBOLS = '€$£¥₽₩₨₱₿'

    NUMERIC_DELTA_FIELDS = {
        "TOTAL", "TOTAL_AMOUNT", "CONTRACT_VALUE", "LINE/TOTAL",
        "LINE/UNIT_PRICE", "LINE/SUB_TOTAL", "OPENING_BALANCE",
        "CLOSING_BALANCE", "LINE/QUANTITY",
    }

    FORMAT_RULES = {
        "IBAN": "iban",
        "ACCOUNT_NUMBER": "account_number",
        "DOCUMENT_ID": "identifier",
        "DOCUMENT_NUMBER": "identifier",
        "NUMBER": "identifier",
        "PO_NUMBER": "identifier",
        "DN_NUMBER": "identifier",
        "INVOICE_DATE": "date",
        "CONTRACT_DATE": "date",
        "EFFECTIVE_DATE": "date",
        "ORDER_DATE": "date",
        "DELIVERY_DATE": "date",
        "STATEMENT_DATE": "date",
        "DATE_OF_BIRTH": "date",
        "EXPIRY_DATE": "date",
    }

    DETECTION_FIELDS = {"SIGNATURE"}

    DATE_FORMATS = [
        "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%Y/%m/%d",
        "%d-%m-%Y", "%m-%d-%Y", "%d.%m.%Y", "%Y.%m.%d",
        "%Y%m%d", "%d %B %Y", "%B %d, %Y", "%d-%b-%Y",
        "%d/%m/%y", "%m/%d/%y",
    ]

    def _gt_field_name(self, field_name: str) -> str:
        """Map GT label to extracted field name (return original if no alias)."""
        return self.GT_TO_LLM.get(field_name, field_name)

    def _find_extracted(self, field_name: str, extracted: dict) -> Any:
        """Look up a GT field name in extracted fields, trying aliases."""
        mapped = self._gt_field_name(field_name)
        if mapped != field_name:
            return extracted.get(mapped)
        val = extracted.get(field_name)
        if val is not None and val != "":
            return val
        return None

    def _clean_list_value(self, val: list, field_name: str) -> str:
        """Extract clean text from a list value for comparison."""
        # Map GT field name to extraction field name to get correct dict key
        mapped = self._gt_field_name(field_name)
        key_hint = mapped.split("/")[-1].lower()
        parts = []
        for item in val:
            if isinstance(item, dict):
                item_val = item.get(key_hint) or item.get("value") or next(iter(item.values()), "")
                parts.append(str(item_val))
            else:
                parts.append(str(item))
        return " ".join(parts)

    def __init__(self, config: PipelineConfig):
        super().__init__(config)
        self.metrics = config.evaluation.metrics
        self.ground_truth_path = config.evaluation.ground_truth_path

    def _get_target_fields(self, ctx: PipelineContext) -> set:
        """Get target fields from pipeline context overrides or config, in that order."""
        override_fields = ctx.metadata.get("target_fields")
        if override_fields:
            return set(override_fields)
        fields = self.config.llm_extraction.target_fields if self.config.llm_extraction.enabled else self.config.end_to_end_vlm.target_fields
        return set(fields)

    async def execute(self, ctx: PipelineContext) -> PipelineContext:
        self.target_fields = self._get_target_fields(ctx)
        results: dict[str, Any] = {}
        metric_timing: dict[str, float] = {}
        if "accuracy" in self.metrics:
            t0 = time.time()
            results["accuracy"] = await self._compute_accuracy(ctx)
            metric_timing["accuracy"] = round(time.time() - t0, 3)
        if "faithfulness" in self.metrics:
            t0 = time.time()
            results["faithfulness"] = await self._compute_faithfulness(ctx)
            metric_timing["faithfulness"] = round(time.time() - t0, 3)
        if "confidence" in self.metrics:
            t0 = time.time()
            results["confidence"] = await self._compute_confidence(ctx)
            metric_timing["confidence"] = round(time.time() - t0, 3)
        if "numeric_delta" in self.metrics:
            t0 = time.time()
            results["numeric_delta"] = await self._compute_numeric_delta(ctx)
            metric_timing["numeric_delta"] = round(time.time() - t0, 3)
        if "format_compliance" in self.metrics:
            t0 = time.time()
            results["format_compliance"] = await self._compute_format_compliance(ctx)
            metric_timing["format_compliance"] = round(time.time() - t0, 3)
        if "detection_rate" in self.metrics:
            t0 = time.time()
            results["detection_rate"] = await self._compute_detection_rate(ctx)
            metric_timing["detection_rate"] = round(time.time() - t0, 3)

        t0 = time.time()
        results["enrichment"] = self._compute_enrichment(ctx)
        metric_timing["enrichment"] = round(time.time() - t0, 3)

        results["metric_timing"] = metric_timing
        ctx.evaluation_results = results
        total_eval = sum(metric_timing.values())
        self.logger.info(
            f"Evaluation results: accuracy={results.get('accuracy', {}).get('score')}, "
            f"faithfulness={results.get('faithfulness', {}).get('score')}, "
            f"confidence={results.get('confidence', {}).get('average')}, "
            f"numeric_delta={results.get('numeric_delta', {}).get('score')}, "
            f"format_compliance={results.get('format_compliance', {}).get('score')}, "
            f"detection_rate={results.get('detection_rate', {}).get('score')}, "
            f"timing={metric_timing} ({total_eval:.2f}s total)"
        )
        has_fields = bool(ctx.pages[0].extracted_fields) if ctx.pages else False
        fields_count = len(ctx.pages[0].extracted_fields) if ctx.pages and ctx.pages[0].extracted_fields else 0
        self.logger.info(f"Result keys: {list(results.keys())}, pages: {len(ctx.pages)}, "
                         f"extracted_fields on page 0: {has_fields} ({fields_count} fields)")
        return ctx

    def _compute_enrichment(self, ctx: PipelineContext) -> dict[str, Any]:
        """Log enrichment data from vendor lookup, anomaly detection, and agentic retries."""
        enrichment: dict[str, Any] = {
            "document_type": ctx.metadata.get("document_type", "unknown"),
            "document_type_confidence": ctx.metadata.get("document_type_confidence", 0.0),
            "agentic_retries": ctx.metadata.get("agentic_retries", 0),
        }

        vendor_matches = []
        anomaly_flags = []
        for page in ctx.pages:
            vm = page.metadata.get("vendor_match")
            if vm:
                vendor_matches.append({
                    "page": page.page_number,
                    "vendor_name": vm.get("name"),
                    "match_score": vm.get("_score"),
                    "vendor_id": page.metadata.get("vendor_id"),
                })

            for a in page.metadata.get("anomalies", []):
                anomaly_flags.append({"page": page.page_number, **a})
            for a in page.metadata.get("vendor_anomalies", []):
                anomaly_flags.append({"page": page.page_number, **a})

        enrichment["vendor_matches"] = vendor_matches
        enrichment["anomaly_flags"] = anomaly_flags
        enrichment["overall_confidence"] = ctx.metadata.get("overall_confidence")

        return enrichment

    @staticmethod
    def _ocr_norm(v: str) -> str:
        """Normalize text to handle OCR confusions and minor differences.

        Strips accents, currency symbols, and lowercases.
        """
        v = unicodedata.normalize('NFKD', v).encode('ascii', 'ignore').decode('ascii')
        v = v.strip().lower()
        for ch in EvaluationStep.CURRENCY_SYMBOLS:
            v = v.replace(ch, '')
        return v

    @staticmethod
    def _norm_token(tok: str) -> str:
        """Normalize a single token for numerical comparison.

        Replaces comma decimal with dot, strips trailing zeros after decimal point.
        """
        tok = tok.replace(',', '.')
        if '.' in tok:
            tok = tok.rstrip('0').rstrip('.')
        return tok

    def _norm_number(self, v: str) -> str:
        """Normalize number formatting: remove currency, thousands separators,
        normalize comma decimal to dot, strip trailing zeros."""
        for ch in self.CURRENCY_SYMBOLS:
            v = v.replace(ch, '')
        v = v.replace(' ', '').replace(',', '.')
        if '.' in v:
            v = v.rstrip('0').rstrip('.')
        return v

    def _parse_number(self, v: str) -> float | None:
        """Parse a number from a string handling European and US formats."""
        for ch in self.CURRENCY_SYMBOLS:
            v = v.replace(ch, '')
        v = v.strip()
        v = re.sub(r'\s+', '', v)
        if ',' in v and '.' in v:
            if v.rfind(',') > v.rfind('.'):
                v = v.replace('.', '').replace(',', '.')
            else:
                v = v.replace(',', '')
        elif ',' in v:
            v = v.replace(',', '.')
        try:
            return float(v)
        except ValueError:
            return None

    def _norm(self, v: Any) -> str:
        v = str(v).strip().lower().replace(',', '.')
        for ch in self.CURRENCY_SYMBOLS:
            v = v.replace(ch, '')
        return v

    @staticmethod
    def _levenshtein(a: str, b: str) -> int:
        """Levenshtein edit distance."""
        la, lb = len(a), len(b)
        if la < lb:
            a, b = b, a
            la, lb = lb, la
        prev = list(range(lb + 1))
        for i, ca in enumerate(a):
            curr = [i + 1]
            for j, cb in enumerate(b):
                cost = 0 if ca == cb else 1
                curr.append(min(curr[j] + 1, prev[j + 1] + 1, prev[j] + cost))
            prev = curr
        return prev[lb]

    def _token_f1(self, a: str, b: str) -> float:
        a_norm = self._ocr_norm(a)
        b_norm = self._ocr_norm(b)
        at = [self._norm_token(t) for t in a_norm.split()]
        bt = [self._norm_token(t) for t in b_norm.split()]
        # Remove empty tokens (from stripped currency symbols)
        at = [t for t in at if t]
        bt = [t for t in bt if t]
        if not at or not bt:
            return 1.0 if a_norm == b_norm else 0.0
        # If both are single tokens, use character-level similarity
        if len(at) == 1 and len(bt) == 1:
            max_len = max(len(at[0]), len(bt[0]))
            if max_len == 0:
                return 1.0
            dist = self._levenshtein(at[0], bt[0])
            sim = 1.0 - dist / max_len
            return round(sim, 3)
        inter = set(at) & set(bt)
        prec = len(inter) / len(at)
        rec = len(inter) / len(bt)
        return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

    async def _compute_accuracy(self, ctx: PipelineContext) -> dict:
        per_field: dict[str, list] = {}
        exact_total = 0
        exact_hits = 0
        partial_total = 0
        partial_score_sum = 0.0

        for page in ctx.pages:
            img_path = page.metadata.get("image_path", "")
            if not img_path:
                continue
            tsv_file = find_annotation_file(img_path, original_filename=page.metadata.get("original_filename"))
            if not tsv_file:
                continue
            gt = load_ground_truth(tsv_file)
            gt_fields: dict[str, list[str]] = {}
            for w, _b, label in zip(gt.words, gt.boxes, gt.labels, strict=False):
                if label == "O":
                    continue
                gt_fields.setdefault(label, []).append(w)
            extracted = page.extracted_fields or {}

            seen_gt = set()
            for field_name, gt_values in gt_fields.items():
                mapped = self._gt_field_name(field_name)
                if mapped not in self.target_fields:
                    continue
                gt_text = " ".join(gt_values)
                ex_val = self._find_extracted(field_name, extracted)
                if ex_val is None or ex_val == "null" or ex_val == "":
                    per_field.setdefault(field_name, []).append({
                        "gt": gt_text, "pred": None, "exact": 0, "token_f1": 0.0,
                    })
                    exact_total += 1
                    continue
                exact_total += 1
                partial_total += 1
                if isinstance(ex_val, list):
                    ex_val_clean = self._clean_list_value(ex_val, field_name)
                else:
                    ex_val_clean = str(ex_val)
                pred_text = self._norm(ex_val_clean)
                gt_lower = self._norm(gt_text)
                gt_num = self._parse_number(gt_text)
                pred_num = self._parse_number(ex_val_clean)
                exact = 1 if (
                    self._ocr_norm(ex_val_clean) == self._ocr_norm(gt_text) or
                    self._norm_number(ex_val_clean) == self._norm_number(gt_text) or
                    (gt_num is not None and pred_num is not None and gt_num == pred_num)
                ) else 0
                exact_hits += exact
                tf1 = 1.0 if exact else self._token_f1(pred_text, gt_lower)
                partial_score_sum += tf1
                per_field.setdefault(field_name, []).append({
                    "gt": gt_text, "pred": str(ex_val), "exact": exact, "token_f1": round(tf1, 3),
                })
                seen_gt.add(mapped)

            for field_name in extracted:
                if field_name not in self.target_fields:
                    continue
                if field_name in seen_gt:
                    continue
                ex_val = extracted[field_name]
                if ex_val and ex_val != "null":
                    per_field.setdefault(field_name, []).append({
                        "gt": None, "pred": str(ex_val), "exact": 0, "token_f1": 0.0,
                    })
                    exact_total += 1

        field_summary = {}
        for fname, entries in per_field.items():
            exacts = sum(e["exact"] for e in entries)
            tf1s = [e["token_f1"] for e in entries]
            field_summary[fname] = {
                "count": len(entries),
                "exact_match": round(exacts / len(entries), 3) if entries else 0,
                "avg_token_f1": round(sum(tf1s) / len(tf1s), 3) if tf1s else 0,
                "entries": entries[:5],
            }

        return {
            "score": round(exact_hits / exact_total, 3) if exact_total else None,
            "exact_match": exact_hits,
            "total_fields": exact_total,
            "partial_token_f1": round(partial_score_sum / partial_total, 3) if partial_total else None,
            "per_field": field_summary,
        }

    async def _compute_faithfulness(self, ctx: PipelineContext) -> dict:
        faithful = 0
        total = 0
        per_field = {}
        for page in ctx.pages:
            ocr_text = (
                page.metadata.get("hybrid_text", "") or
                page.metadata.get("doc_graph_text", "") or
                page.metadata.get("vlm_text", "") or
                page.metadata.get("e2e_vlm_raw", "") or
                page.metadata.get("ocr_text_post_corrected", "") or
                (page.ocr_result.to_text() if page.ocr_result else "")
            )
            if ocr_text and page.extracted_fields:
                for field_name, value in page.extracted_fields.items():
                    if field_name not in self.target_fields:
                        continue
                    if not value or value == "null":
                        continue
                    total += 1
                    if isinstance(value, list):
                        val_str = self._clean_list_value(value, field_name)
                    else:
                        val_str = str(value)
                    val_norm = self._ocr_norm(val_str)
                    val_tokens = {self._norm_token(t) for t in val_norm.split()}
                    ocr_norm = self._ocr_norm(ocr_text)
                    ocr_tokens_norm = {self._norm_token(t) for t in ocr_norm.split()}
                    val_tokens.discard('')
                    ocr_tokens_norm.discard('')
                    overlap = val_tokens & ocr_tokens_norm
                    f = 1 if overlap else int(val_norm in ocr_norm)
                    faithful += f
                    per_field.setdefault(field_name, {"faithful": 0, "total": 0})
                    per_field[field_name]["total"] += 1
                    if f:
                        per_field[field_name]["faithful"] += 1
        return {
            "score": round(faithful / max(total, 1), 3),
            "faithful": faithful,
            "total": total,
            "per_field": {k: round(v["faithful"] / v["total"], 3) for k, v in per_field.items()},
        }

    async def _compute_confidence(self, ctx: PipelineContext) -> dict:
        confidences = []
        for page in ctx.pages:
            if page.validation_result:
                confidences.append(1.0 if page.validation_result.get("is_valid") else 0.5)
        return {
            "average": round(sum(confidences) / max(len(confidences), 1), 3),
            "count": len(confidences),
        }

    async def _compute_numeric_delta(self, ctx: PipelineContext) -> dict:
        scores = []
        total_fields = 0
        per_field: dict[str, list] = {}

        for page in ctx.pages:
            img_path = page.metadata.get("image_path", "")
            if not img_path:
                continue
            tsv_file = find_annotation_file(img_path, original_filename=page.metadata.get("original_filename"))
            if not tsv_file:
                continue
            gt = load_ground_truth(tsv_file)
            gt_fields: dict[str, list[str]] = {}
            for w, _b, label in zip(gt.words, gt.boxes, gt.labels, strict=False):
                if label == "O":
                    continue
                gt_fields.setdefault(label, []).append(w)

            extracted = page.extracted_fields or {}

            for field_name, gt_values in gt_fields.items():
                mapped = self._gt_field_name(field_name)
                if mapped not in self.target_fields:
                    continue
                if mapped not in self.NUMERIC_DELTA_FIELDS:
                    continue
                gt_text = " ".join(gt_values)
                gt_num = self._parse_number(gt_text)
                if gt_num is None:
                    continue

                ex_val = self._find_extracted(field_name, extracted)
                if ex_val is None or ex_val == "null" or ex_val == "":
                    per_field.setdefault(field_name, []).append({
                        "gt": gt_text, "pred": None, "delta": None,
                    })
                    total_fields += 1
                    continue

                if isinstance(ex_val, list):
                    ex_val_clean = self._clean_list_value(ex_val, field_name)
                else:
                    ex_val_clean = str(ex_val)

                pred_num = self._parse_number(ex_val_clean)
                if pred_num is None:
                    per_field.setdefault(field_name, []).append({
                        "gt": gt_text, "pred": ex_val_clean, "delta": None,
                    })
                    total_fields += 1
                    continue

                total_fields += 1
                if abs(gt_num) < 0.001:
                    delta_score = 1.0 if abs(pred_num - gt_num) < 0.001 else 0.0
                else:
                    delta_score = max(0.0, 1.0 - abs(pred_num - gt_num) / abs(gt_num))
                delta_score = round(delta_score, 3)
                scores.append(delta_score)
                per_field.setdefault(field_name, []).append({
                    "gt": gt_text, "pred": ex_val_clean, "delta": delta_score,
                })

        field_summary = {}
        for fname, entries in per_field.items():
            deltas = [e["delta"] for e in entries if e["delta"] is not None]
            field_summary[fname] = {
                "count": len(entries),
                "avg_delta": round(sum(deltas) / len(deltas), 3) if deltas else None,
                "entries": entries[:5],
            }

        return {
            "score": round(sum(scores) / len(scores), 3) if scores else None,
            "count": len(scores),
            "total_fields": total_fields,
            "per_field": field_summary,
        }

    @staticmethod
    def _validate_iban(value: str) -> bool:
        iban = value.strip().upper().replace(' ', '')
        if not re.match(r'^[A-Z]{2}\d{2}[A-Z0-9]{1,30}$', iban):
            return False
        iban_rearranged = iban[4:] + iban[:4]
        iban_numeric = ''.join(
            str(ord(c) - 55) if c.isalpha() else c
            for c in iban_rearranged
        )
        try:
            return int(iban_numeric) % 97 == 1
        except (ValueError, IndexError):
            return False

    @staticmethod
    def _validate_date(value: str) -> bool:
        from datetime import datetime
        val = value.strip().replace('"', '').replace("'", "")
        for fmt in EvaluationStep.DATE_FORMATS:
            try:
                datetime.strptime(val, fmt)
                return True
            except ValueError:
                continue
        return False

    @staticmethod
    def _validate_identifier(value: str) -> bool:
        val = value.strip()
        if not val:
            return False
        if re.match(r'^[A-Za-z0-9][A-Za-z0-9\s\.\-\_\/]{1,48}[A-Za-z0-9]$', val):
            return True
        return bool(re.match(r'^[A-Za-z0-9]{3,50}$', val))

    @staticmethod
    def _validate_account_number(value: str) -> bool:
        val = value.strip()
        return bool(re.match(r'^[A-Za-z0-9\s\-]{4,34}$', val))

    def _check_format(self, field_name: str, value: str) -> bool:
        rule = self.FORMAT_RULES.get(field_name)
        if rule is None:
            return True
        if rule == "iban":
            return self._validate_iban(value)
        elif rule == "date":
            return self._validate_date(value)
        elif rule == "identifier":
            return self._validate_identifier(value)
        elif rule == "account_number":
            return self._validate_account_number(value)
        return True

    async def _compute_format_compliance(self, ctx: PipelineContext) -> dict:
        total = 0
        passed = 0
        per_field: dict[str, list] = {}

        for page in ctx.pages:
            extracted = page.extracted_fields or {}
            for field_name, value in extracted.items():
                if field_name not in self.target_fields:
                    continue
                if field_name not in self.FORMAT_RULES:
                    continue
                if not value or value == "null":
                    per_field.setdefault(field_name, []).append({
                        "value": None, "valid": False, "reason": "empty",
                    })
                    total += 1
                    continue

                total += 1
                val_str = str(value)
                valid = self._check_format(field_name, val_str)
                if valid:
                    passed += 1
                per_field.setdefault(field_name, []).append({
                    "value": val_str[:50],
                    "valid": valid,
                })

        field_summary = {}
        for fname, entries in per_field.items():
            valids = sum(1 for e in entries if e["valid"])
            field_summary[fname] = {
                "count": len(entries),
                "pass_rate": round(valids / len(entries), 3) if entries else 0,
                "entries": entries[:5],
            }

        return {
            "score": round(passed / max(total, 1), 3),
            "passed": passed,
            "total": total,
            "per_field": field_summary,
        }

    async def _compute_detection_rate(self, ctx: PipelineContext) -> dict:
        total = 0
        detected = 0
        per_field: dict[str, list] = {}

        for page in ctx.pages:
            extracted = page.extracted_fields or {}
            for field_name, value in extracted.items():
                if field_name not in self.target_fields:
                    continue
                if field_name not in self.DETECTION_FIELDS:
                    continue
                total += 1
                is_detected = bool(value) and value != "null" and value != ""
                if is_detected:
                    detected += 1
                per_field.setdefault(field_name, []).append({
                    "detected": is_detected,
                    "value": str(value)[:50] if value else None,
                })

        field_summary = {}
        for fname, entries in per_field.items():
            detected_count = sum(1 for e in entries if e["detected"])
            field_summary[fname] = {
                "count": len(entries),
                "detection_rate": round(detected_count / len(entries), 3) if entries else 0,
                "entries": entries[:5],
            }

        return {
            "score": round(detected / max(total, 1), 3),
            "detected": detected,
            "total": total,
            "per_field": field_summary,
        }
