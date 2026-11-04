
import json
import unittest
from unittest.mock import patch

from healthcare_data_quality.investigation_report import (
    summarize_investigation,
)


class EvidenceReferenceTests(unittest.TestCase):

    def test_rejects_unknown_evidence_id(self):
        history = [{
            "step": 1,
            "tool": "synthetic_check",
            "observations": [{"value": "TEST1"}],
        }]

        report = {
            "findings": [{
                "claim": "A synthetic finding.",
                "evidence_ids": ["T99.O1"],
            }],
            "unresolved_questions": [],
            "recommended_next_step": "Review evidence.",
        }

        with patch(
            "healthcare_data_quality.investigation_report.query_local_model",
            return_value=json.dumps(report),
        ):
            with self.assertRaisesRegex(
                ValueError, "Unknown evidence reference"
            ):
                summarize_investigation(history)
