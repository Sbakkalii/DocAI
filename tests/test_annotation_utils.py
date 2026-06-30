"""Tests for annotation utilities."""

from pipeline.annotation_utils import (
    ANNOTATION_COLORS,
    annotations_to_boxes,
    build_page_fragments,
    find_annotation_file,
    load_ground_truth,
    match_predicted_fields,
)


class TestFindAnnotationFile:
    def test_no_annotation_returns_none(self, tmp_path):
        result = find_annotation_file(str(tmp_path / "nonexistent.jpg"))
        assert result is None


class TestLoadGroundTruth:
    def test_parses_tsv_correctly(self, tmp_path):
        tsv = tmp_path / "test.tsv"
        tsv.write_text(
            "text,left,top,width,height,label\n"
            "INVOICE,10,20,100,30,NUMBER\n"
            "TOTAL,50,60,80,20,TOTAL\n"
        )
        gt = load_ground_truth(tsv, image_width=800, image_height=600)
        assert len(gt.words) == 2
        assert gt.words[0] == "INVOICE"
        assert gt.labels[1] == "TOTAL"
        assert gt.boxes[0] == [10, 20, 110, 50]
        assert gt.image_width == 800

    def test_empty_tsv(self, tmp_path):
        tsv = tmp_path / "empty.tsv"
        tsv.write_text("text\tleft\ttop\twidth\theight\tlabel\n")
        gt = load_ground_truth(tsv)
        assert len(gt.words) == 0


class TestMatchPredictedFields:
    def test_exact_value_match(self):
        fields = {"TOTAL": "100.00"}
        words = ["Total", "is", "100.00", "euros"]
        boxes = [[0, 0, 10, 10], [10, 0, 20, 10], [20, 0, 50, 10], [50, 0, 80, 10]]
        result = match_predicted_fields(fields, words, boxes, 800, 600)
        assert len(result) == 1
        assert result[0]["label"] == "TOTAL"

    def test_evidence_text_used(self):
        fields = {"TOTAL": "100.00"}
        evidence = {"TOTAL": "100.00"}
        words = ["Amount:", "100.00", "USD"]
        boxes = [[0, 0, 20, 10], [20, 0, 50, 10], [50, 0, 70, 10]]
        result = match_predicted_fields(fields, words, boxes, 800, 600, evidence=evidence)
        assert len(result) == 1

    def test_no_match_returns_empty(self):
        fields = {"TOTAL": "999.99"}
        words = ["Nothing", "here"]
        boxes = [[0, 0, 10, 10], [10, 0, 20, 10]]
        result = match_predicted_fields(fields, words, boxes, 800, 600)
        assert len(result) == 0

    def test_skip_evidence_field(self):
        fields = {"_evidence": "some text", "TOTAL": "50"}
        words = ["Total", "50"]
        boxes = [[0, 0, 10, 10], [10, 0, 20, 10]]
        result = match_predicted_fields(fields, words, boxes, 800, 600)
        assert len(result) == 1
        assert result[0]["label"] == "TOTAL"


class TestAnnotationsToBoxes:
    def test_predicted_gets_field_color(self):
        anns = [{"label": "NUMBER", "text": "001", "box": [0, 0, 10, 10], "confidence": 1.0, "source": "predicted"}]
        result = annotations_to_boxes(anns)
        assert result[0]["color"] == ANNOTATION_COLORS["NUMBER"]
        assert result[0]["source"] == "predicted"

    def test_ground_truth_gets_blue(self):
        anns = [{"label": "NUMBER", "text": "001", "box": [0, 0, 10, 10], "confidence": 1.0, "source": "ground_truth"}]
        result = annotations_to_boxes(anns)
        assert result[0]["color"] == "#3b82f6"
        assert result[0]["source"] == "ground_truth"

    def test_unknown_field_gets_fallback_color(self):
        anns = [{"label": "UNKNOWN_FIELD", "text": "x", "box": [0, 0, 10, 10], "confidence": 1.0, "source": "predicted"}]
        result = annotations_to_boxes(anns)
        assert result[0]["color"] == "#95a5a6"


def test_build_page_fragments_no_ocr():
    """Page with no OCR result produces only field fragments."""
    from pipeline.base import PageResult
    page = PageResult(page_number=1, extracted_fields={"TOTAL": "100"})
    frags = build_page_fragments(page)
    field_frags = [f for f in frags if f["fragment_type"] == "field"]
    assert len(field_frags) == 1
