"""Expanding Agent B's recommendation replaces previews rather than duplicating them."""

from types import SimpleNamespace

from test_cleaning_review_cards import StubUI, _app, _catalog


class CaptureUI(StubUI):
    def __init__(self):
        super().__init__()
        self.expanded = False
        self.written = []
        self.captions = []

    def checkbox(self, label, **kwargs):
        return self.expanded if label == "Show full recommendation" else False

    def write(self, *args, **kwargs):
        self.written.append(args)

    def caption(self, text, **kwargs):
        self.captions.append(str(text))


def test_recommendation_expansion_replaces_preview_not_duplicates(monkeypatch, tmp_path):
    source = tmp_path / "one.csv"
    source.write_text("Taxon,Count\na,0.2\n", encoding="utf-8")
    st = CaptureUI()
    app = _app(monkeypatch, st)
    finding = _catalog()[0]
    fid = finding["id"]
    action = ("Replace exactly the invalid measurements at the documented cell "
              "positions with the agreed missing-value marker while preserving "
              "every other original observation. " * 3).strip()
    rationale = ("Counts must satisfy the validated input specification, and the "
                 "exact invalid positions are recorded in the investigation audit. " * 3).strip()

    class SavedAgentB:
        def snapshot(self, key):
            return {"entries": {fid: {"status": "ready", "suggestion": action,
                                      "decision_needed": rationale}}}

    monkeypatch.setattr("cleaning_jobs.recommendation_manager", lambda: SavedAgentB())
    monkeypatch.setattr("healthcare_data_quality.finding_cleaning.build_cleaning_findings",
                        lambda *args, **kwargs: [finding])
    monkeypatch.setattr("healthcare_data_quality.finding_cleaning.allowed_positions",
                        lambda *args, **kwargs: (set(), set()))

    def render():
        st.written.clear()
        st.captions.clear()
        app.render_cleaning_workspace(
            [source], [SimpleNamespace(name="one.csv")], {}, "test-no-repeat", []
        )

    render()
    assert action not in [item[0] for item in st.written if len(item) == 1]
    assert sum("Replace exactly the invalid measurements" in str(item)
               for item in st.written) == 1
    assert sum("Reason / evidence limit:" in line for line in st.captions) == 1

    st.expanded = True
    render()
    assert [item for item in st.written if item == (action,)] == [(action,)]
    assert [item for item in st.written if len(item) == 2 and item[1] == rationale] == [
        ("**Reason / evidence limit:**", rationale)
    ]
    assert not any("Reason / evidence limit:" in line for line in st.captions)
    assert sum("Replace exactly the invalid measurements" in str(item)
               for item in st.written) == 1
