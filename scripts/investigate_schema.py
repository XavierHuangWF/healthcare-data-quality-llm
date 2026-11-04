
import json
import sys
from pathlib import Path

from healthcare_data_quality.dataset_profiler import profile_csv
from healthcare_data_quality.llm_client import query_local_model


def investigate_schema(path: Path) -> str:
    profile = profile_csv(path)

    prompt = f"""
You are a scientific data-quality investigator.

Examine this unfamiliar dataset profile:

{json.dumps(profile, indent=2)}

For each column, identify:
1. Its likely semantic role or data type.
2. Evidence supporting your interpretation.
3. Uncertainty or alternative interpretations.
4. The next validation check you recommend.

Requirements:
- Infer semantics from the supplied evidence.
- Do not assume a column is invalid because it looks unusual.
- Distinguish observations from hypotheses.
- Do not invent dataset definitions.
- Do not propose automatic corrections.
- Keep the response concise, under 250 words.
"""

    return query_local_model(prompt)


if __name__ == "__main__":
    result = investigate_schema(Path(sys.argv[1]))
    print(result)
