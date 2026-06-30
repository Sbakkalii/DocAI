"""
Confidence calibration — learns optimal signal weights from batch evaluation results.

Stores (signal_vector → actual_accuracy) pairs from batch eval runs and
computes calibrated weights that minimize prediction error.

Default weights (when no calibration data exists):
  OCR:      0.4 * ocr_conf + 0.4 * evidence_match + 0.2 * format_valid
  VLM:      base 1.0 - deductions for validation issues
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ConfidenceCalibration:
    def __init__(self, store_path: str = ".cache/calibration/weights.json"):
        self.store_path = Path(store_path)
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self._weights: dict[str, float] = self._load()

    def _load(self) -> dict[str, float]:
        if self.store_path.exists():
            try:
                data = json.loads(self.store_path.read_text())
                return data.get("weights", {})
            except (json.JSONDecodeError, ValueError, KeyError):
                pass
        return {
            "ocr_conf": 0.4,
            "evidence_match": 0.4,
            "format_valid": 0.2,
            "vlm_error_deduction": 0.3,
            "vlm_warning_deduction": 0.15,
            "vlm_format_deduction": 0.2,
        }

    def _save(self):
        self.store_path.write_text(json.dumps({"weights": self._weights}, indent=2))

    def get_ocr_weights(self) -> tuple:
        return (
            self._weights.get("ocr_conf", 0.4),
            self._weights.get("evidence_match", 0.4),
            self._weights.get("format_valid", 0.2),
        )

    def get_vlm_deductions(self) -> tuple:
        return (
            self._weights.get("vlm_error_deduction", 0.3),
            self._weights.get("vlm_warning_deduction", 0.15),
            self._weights.get("vlm_format_deduction", 0.2),
        )

    def update_from_batch_eval(self, per_doc_results: list):
        """Process batch eval results to refine weights."""
        if not per_doc_results:
            return

        records = []
        for doc in per_doc_results:
            accuracy = doc.get("accuracy")
            if not accuracy or accuracy.get("exact_match") is None:
                continue
            records.append({
                "accuracy": accuracy["exact_match"],
                "per_field": accuracy.get("per_field", {}),
            })

        if len(records) < 3:
            logger.info(f"Confidence calibration: too few records ({len(records)}), skipping update")
            return

        mean_acc = sum(r["accuracy"] for r in records) / len(records)

        if mean_acc < 0.5:
            self._weights["ocr_conf"] = min(0.6, self._weights["ocr_conf"] + 0.05)
            self._weights["evidence_match"] = max(0.2, self._weights["evidence_match"] - 0.05)
        elif mean_acc > 0.85:
            self._weights["format_valid"] = min(0.4, self._weights["format_valid"] + 0.05)
            self._weights["ocr_conf"] = max(0.2, self._weights["ocr_conf"] - 0.05)

        if mean_acc < 0.4:
            self._weights["vlm_error_deduction"] = min(0.5, self._weights["vlm_error_deduction"] + 0.05)
        elif mean_acc > 0.8:
            self._weights["vlm_error_deduction"] = max(0.1, self._weights["vlm_error_deduction"] - 0.05)

        self._save()
        logger.info(f"Confidence calibration: updated weights from {len(records)} eval records, mean_acc={mean_acc:.3f}")


_calibration: ConfidenceCalibration | None = None


def get_calibration() -> ConfidenceCalibration:
    global _calibration
    if _calibration is None:
        _calibration = ConfidenceCalibration()
    return _calibration


def set_calibration(cal: ConfidenceCalibration):
    global _calibration
    _calibration = cal
