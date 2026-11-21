"""One scientific approval can cover a uniform, fully evidenced finding."""
import csv
from io import StringIO

from healthcare_data_quality.cleaning.finding_cleaning import expand_uniform_evidence_plan
from healthcare_data_quality.r_runtime.scoped_r_execution import _r_insertion_function, _generate_r, _as_csv
from healthcare_data_quality.cleaning.in_memory_cleaning import assess_verified_in_memory, apply_verified_in_memory


def _inputs(tmp_path, n=618, *, one_different=False):
    source = tmp_path / "source.csv"
    with source.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["taxon", "Count"])
        for i in range(n):
            writer.writerow([str(i), "unmatched" if (one_different and i == n-1) else ""])
    finding = dict(id="dataset_1:qc:2", dataset_id="dataset_1", check_key="find_invalid_counts",
                   actionable=True, evidence=[{"CSV row": i+2, "Column": 2} for i in range(n)])
    seed = dict(dataset_id="dataset_1", operation="replace_cells",
                changes=[dict(row=i+2, column=2, before="", after="NA") for i in range(32)])
    return source, finding, seed


def test_618_uniform_evidenced_edits_one_plan_and_one_memory_approval(tmp_path):
    source, finding, seed = _inputs(tmp_path)
    plan, added = expand_uniform_evidence_plan(seed, finding, source, full_finding_accepted=True)
    assert added == 586 and len(plan["changes"]) == 618
    r_code = _generate_r([{ "kind":"cell", "csv_row":str(c["row"]),
                            "csv_column":str(c["column"]), "before":c["before"],
                            "after":c["after"] } for c in plan["changes"]])
    assert len(r_code) < 2000
    assert len(_r_insertion_function(plan).encode("utf-8")) < 2500
    assert len(_r_insertion_function(plan)) == len(_r_insertion_function({**plan, "changes": plan["changes"][:1]}))
    assert _r_insertion_function(plan).count("<- verified_changes$replacement[i]") == 1
    rows = list(csv.reader(StringIO(_as_csv([
        dict(kind="cell", csv_row=str(c["row"]), csv_column=str(c["column"]),
             before=c["before"], after=c["after"]) for c in plan["changes"]], result=False))))
    assert len(rows) == 619 and len(rows[0]) == 5
    original = source.read_bytes()
    review = assess_verified_in_memory(original, original, plan)
    assert len(review["ready_to_apply"]) == 618
    output, result = apply_verified_in_memory(original, original, plan, reviewer="tester")
    assert result["changes_applied"] == 618
    assert output != original and source.read_bytes() == original
    assert all(row[1] == "NA" for row in list(csv.reader(StringIO(output.decode())))[1:])


def test_no_evidence_expansion_without_explicit_full_scope_acceptance(tmp_path):
    source, finding, seed = _inputs(tmp_path)
    plan, added = expand_uniform_evidence_plan(seed, finding, source, full_finding_accepted=False)
    assert added == 0 and len(plan["changes"]) == 32


def test_no_evidence_expansion_if_any_source_value_is_different(tmp_path):
    source, finding, seed = _inputs(tmp_path, one_different=True)
    plan, added = expand_uniform_evidence_plan(seed, finding, source, full_finding_accepted=True)
    assert added == 0 and len(plan["changes"]) == 32


def test_no_evidence_expansion_if_model_proposes_mixed_changes(tmp_path):
    source, finding, seed = _inputs(tmp_path)
    seed["changes"][0]["after"] = "0"
    plan, added = expand_uniform_evidence_plan(seed, finding, source, full_finding_accepted=True)
    assert added == 0 and len(plan["changes"]) == 32
