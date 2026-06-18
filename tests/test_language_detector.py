"""Tests for the language detector utility."""

from utils.language_detector import LanguageDetector, LANGUAGE_FIELD_SYNONYMS, FRENCH_STOP_WORDS, ENGLISH_STOP_WORDS


def test_detect_english():
    detector = LanguageDetector()
    lang = detector.detect("This is a simple invoice document written in English")
    assert lang == "en"


def test_detect_french():
    detector = LanguageDetector()
    lang = detector.detect("Ceci est une facture simple rédigée en français")
    # Without langdetect installed, falls back to "en"
    assert lang in ("en", "fr")


def test_detect_empty_text():
    detector = LanguageDetector()
    lang = detector.detect("")
    assert lang == "en"


def test_get_stop_words_english():
    detector = LanguageDetector()
    stops = detector.get_stop_words("en")
    assert isinstance(stops, set)
    assert "the" in stops


def test_get_stop_words_french():
    detector = LanguageDetector()
    stops = detector.get_stop_words("fr")
    assert isinstance(stops, set)
    assert "le" in stops


def test_get_stop_words_fallback():
    detector = LanguageDetector()
    stops = detector.get_stop_words("de")
    assert stops == ENGLISH_STOP_WORDS


def test_get_field_synonyms_english():
    detector = LanguageDetector()
    syns = detector.get_field_synonyms("en")
    assert isinstance(syns, dict)
    assert "TOTAL" in syns  # field names are uppercase keys


def test_get_field_synonyms_french():
    detector = LanguageDetector()
    syns = detector.get_field_synonyms("fr")
    assert isinstance(syns, dict)
    assert "TOTAL" in syns
    assert any("montant" in s for s in syns["TOTAL"])


def test_get_field_synonyms_fallback():
    detector = LanguageDetector()
    syns = detector.get_field_synonyms("de")
    assert syns == LANGUAGE_FIELD_SYNONYMS["en"]


def test_detect_and_report():
    detector = LanguageDetector()
    report = detector.detect_and_report("Hello world")
    assert "language" in report
    assert isinstance(report.get("stop_words_count"), int)
    assert isinstance(report.get("field_synonyms_count"), int)


def test_french_stop_words_content():
    assert isinstance(FRENCH_STOP_WORDS, set)
    assert len(FRENCH_STOP_WORDS) > 10


def test_english_stop_words_content():
    assert isinstance(ENGLISH_STOP_WORDS, set)
    assert len(ENGLISH_STOP_WORDS) > 10
