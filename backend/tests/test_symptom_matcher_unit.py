from backend.services.symptom_matcher import SymptomMatcher


def _make_matcher() -> SymptomMatcher:
    # Same minimal feature set as FakeXGB in conftest.
    feats = ["fever", "cough", "sore_throat"]
    return SymptomMatcher(features=feats, device="cpu")


def test_pl_to_en_mapping_basic():
    """Polish phrases map to English feature keys; relies on the dictionary at SYMPTOMS_DICT_PATH."""
    m = _make_matcher()
    text = "Od wczoraj mam gorączkę, kaszel i ból gardła."
    out = m.extract(text)
    out_set = set(out)

    assert "fever" in out_set
    assert "cough" in out_set
    assert "sore_throat" in out_set


def test_duplicates_are_removed():
    """A symptom repeated in the text appears only once in the output."""
    m = _make_matcher()
    text = "Mam kaszel, kaszel i jeszcze raz kaszel."
    out = m.extract(text)

    assert "cough" in out
    assert out.count("cough") == 1
    assert len(out) == len(set(out))


def test_small_typos_tolerated():
    """Fuzzy matching tolerates small typos and missing Polish diacritics ("goraczke" -> fever)."""
    m = _make_matcher()
    text = "Mam goraczke i kaszle od rana."
    out = m.extract(text)
    out_set = set(out)
    assert "fever" in out_set

def test_no_empty_labels_in_output():
    """Vague or empty input never yields empty or whitespace-only labels."""
    m = _make_matcher()

    for text in [
        "",
        "???",
        "jakieś objawy, ale bez konkretów",
        "objawy objawy objawy",
    ]:
        out = m.extract(text)
        assert all(isinstance(s, str) for s in out)
        assert all(s.strip() for s in out), f"Empty label in output for text={text!r}"
