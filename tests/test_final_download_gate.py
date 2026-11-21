from types import SimpleNamespace

from test_cleaning_review_cards import StubUI, _app, _catalog


class CapturingUI(StubUI):
    def __init__(self):
        super().__init__()
        self.downloads = []
    def download_button(self, label, *, data=None, file_name=None, mime=None, **kwargs):
        self.downloads.append((label, data, file_name, mime, kwargs.get("disabled", False)))
    def dataframe(self, *args, **kwargs):
        pass


def _render(monkeypatch, tmp_path):
    st = CapturingUI()
    app = _app(monkeypatch, st)
    file = tmp_path / 'sample.csv'
    file.write_bytes(b'Taxon,Count\na,0.2\n')
    monkeypatch.setattr('healthcare_data_quality.cleaning.finding_cleaning.build_cleaning_findings',
                        lambda *a, **kw: _catalog())
    monkeypatch.setattr('healthcare_data_quality.cleaning.finding_cleaning.allowed_positions',
                        lambda *a, **kw: (set(), set()))
    def render():
        app.render_cleaning_workspace([file], [SimpleNamespace(name=file.name)], {}, 'key', [])
    render()
    return st, file, render


def test_final_download_not_present_until_all_findings_and_checks_reviewed(monkeypatch, tmp_path):
    st, file, render = _render(monkeypatch, tmp_path)
    # A no-change review is sufficient to verify the export gate; forging an
    # approved edit without an approval ledger would correctly be rejected.
    st.session_state.cleaning_review_statuses[_catalog()[0]['id']] = 'Reviewed — no change'
    render()
    assert len(st.downloads) >= 1 and all(item[4] and item[1] == b"" for item in st.downloads)
    st.session_state.cleaning_review_statuses[_catalog()[1]['id']] = 'Reviewed — no change'
    render()
    assert len(st.downloads) >= 1 and all(item[4] and item[1] == b"" for item in st.downloads)  # the no-issue Gemma check still needs review
    st.session_state.cleaning_review_statuses[_catalog()[2]['id']] = 'Reviewed — no change'
    render()
    assert st.downloads[-1][4] is False
    label, data, name, mime, _ = st.downloads[-1]
    assert label == 'Download cleaned CSV'
    assert data == b'Taxon,Count\na,0.2\n'
    assert name == 'dataset_1_cleaned_sample.csv'
    assert file.read_bytes() == b'Taxon,Count\na,0.2\n'
