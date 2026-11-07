# Healthcare Data Quality LLM

## Personal developer quick-start — commands I use most

> **Personal developer notes (not secret).** These commands are my fast reference for launching and testing this project in **WSL Ubuntu** from the **single** PyCharm project folder. They are not the end-user setup instructions. **If this README is committed to a public GitHub repository, these notes will be public too; put secrets and truly private configuration in an ignored local file instead.** All model inference is local; the Streamlit page is a **local web UI**, not an Internet-hosted service. See the detailed sections below for prerequisites, permission safeguards, and troubleshooting.

**Project root (WSL):** `/mnt/c/Users/Xavier/PycharmProjects/healthcare-data-quality-llm`  
**Existing Python environment:** `/home/xavier/.venv`  
**Local model API:** `http://127.0.0.1:8000` · **Streamlit UI:** `http://127.0.0.1:8501`  
**Git branches:** `backend` = agent/kernel/CLI; `main` = UI + tested backend. **Same folder, switch branches**; do not create another project directory.


### Export my current backend source for ChatGPT review

Run from the project root in the PyCharm WSL terminal.

This packages the **current working files**, including uncommitted code changes, without connecting to GitHub.

```bash
git branch --show-current
git status --short

ARCHIVE="/mnt/c/Users/Xavier/Downloads/hdq-backend-source-$(date +%Y%m%d-%H%M%S).zip"

git ls-files --cached --others --exclude-standard -- \
  src scripts tests app.py README.md .gitignore \
  | sort -u \
  | zip -q "$ARCHIVE" -@

python -m zipfile -t "$ARCHIVE"
python -m zipfile -l "$ARCHIVE"

echo "Source archive: $ARCHIVE"
```

Upload the resulting ZIP from Windows Downloads to ChatGPT.

**Included:** Source code, scripts, tests, application, README, and Git ignore rules.

**Excluded:** Dataset folders, local models, virtual environments, Git history, and ignored files.

**Important:**
- The `zip` utility must be installed in WSL.
- Inspect the archive listing before uploading. Files within the included directories could still contain confidential information.
- This exports the working tree, unlike `git archive HEAD`, which excludes uncommitted changes.
- Never include confidential datasets, patient records, credentials, API keys, or private logs.
- Do not commit or push merely to package code for review.


### Terminal A — Start Gemma / vLLM first (keep running)

```bash
cd /mnt/c/Users/Xavier/PycharmProjects/healthcare-data-quality-llm
source /home/xavier/.venv/bin/activate

# Use my existing LOCAL model launcher, if present and configured:
bash scripts/serve_gemma4.sh
```

> `scripts/serve_gemma4.sh` is a local launcher seen during development but was **not included in the reviewed source archive**. Confirm that it exists, points at the downloaded local model, and serves the expected OpenAI-compatible API; otherwise use my established vLLM startup command. Do **not** launch a hosted-model service instead. Keep this terminal running.

**Check the server from another terminal:**

```bash
curl -fsS http://127.0.0.1:8000/v1/models
```

The model ID advertised by the server must agree with `src/healthcare_data_quality/llm_client.py` (reviewed configuration: `gemma4-12b`). A response here confirms the endpoint is reachable, not that inference is successful.

### Terminal B — Start the offline CLI/kernel development workflow

```bash
cd /mnt/c/Users/Xavier/PycharmProjects/healthcare-data-quality-llm
source /home/xavier/.venv/bin/activate

git status --short         # Commit or stash changes BEFORE switching branches
git switch backend        # Backend-only development/testing version

# Make a small, known-good synthetic dataset + provenance.
PYTHONPATH=src python scripts/create_poc_dataset.py

# Run the real local agent. Progress prints in terminal; full JSON goes to /tmp.
PYTHONPATH=src python scripts/run_investigation.py \
  data/synthetic/poc_duplicate_columns.csv \
  > /tmp/hdq_investigation.json

# Confirm output is valid JSON and inspect findings/proposals.
python -m json.tool /tmp/hdq_investigation.json > /dev/null \
  && echo "CLI INVESTIGATION JSON: PASS"
python -m json.tool /tmp/hdq_investigation.json | less
```

The CLI uses the same `src/healthcare_data_quality/` engine as the UI. The progress messages on **stderr** should stay visible even when the JSON on **stdout** is redirected. Press `q` to exit `less`.

**Use a different local dataset for investigation (read-only):**

```bash
PYTHONPATH=src python scripts/run_investigation.py \
  data/my_nonconfidential_test.csv > /tmp/hdq_custom_investigation.json
```

Replace the example CSV path with a dataset you are authorized to use. Never commit real patient/research data, derived copies, or private decision-memory logs. The CLI currently has a limited investigation-step budget; valid JSON does not imply full coverage.


### Run a customer's dataset — current and future workflows

The backend is designed to accept unfamiliar scientific CSV datasets without requiring dataset-specific Python modifications.

#### Current workflow — implemented

Start the local Gemma server in Terminal A.

From the project root in Terminal B:

```bash
source /home/xavier/.venv/bin/activate

DATASET="/mnt/c/Users/Xavier/Downloads/customer_data.csv"

PYTHONPATH=src python scripts/run_investigation.py \
  "$DATASET" \
  > /tmp/hdq_customer_audit.json

python -m json.tool /tmp/hdq_customer_audit.json | less
```

The current implementation:
- Profiles the dataset and invokes LLM-selected read-only tools.
- Collects tool observations and evidence-backed findings.
- Identifies unresolved questions.
- Generates and validates remediation proposals.
- Writes investigation results to JSON.
- Does not modify the source dataset.

To review the proposals:

```bash
PYTHONPATH=src python scripts/review_proposals.py \
  /tmp/hdq_customer_audit.json \
  "$DATASET"
```

This records Accept / Reject / Defer decisions in persistent decision memory.

IMPORTANT: Human review does not authorize or execute cleaning. Supported cleaning operations currently require additional authorization, execution, and independent verification steps.

#### Unified interactive backend — IMPLEMENTED, END-TO-END VALIDATION PENDING

The final customer-facing CLI should require only one command:

```bash
PYTHONPATH=src python scripts/run_workflow.py \
  --input "/path/to/customer_data.csv" \
  --interactive
```

scripts/run_workflow.py is the executable interactive backend entry point.

The backend integrates dataset auditing, LLM-directed investigation,
evidence-grounded proposals, human review, R draft generation,
Bubblewrap-isolated synthetic verification, separate authorization,
supported remediation, and audit recording.

Individual components and authorization gates have passed targeted
tests. The complete interactive workflow still requires an end-to-end
integration test using a synthetic dataset.

Synthetic verification does not establish scientific correctness
or authorize changes to customer data.

The unified interactive workflow must:

1. Ingest the customer's dataset and establish its fingerprint.
2. Automatically execute all applicable baseline data-quality checks.
3. Run LLM-directed investigations using the existing tool registry.
4. Track investigation coverage and unresolved evidence gaps.
5. Attempt further read-only investigation before requesting human input.
6. Present specific issues, affected records/columns, evidence, uncertainty, and proposed corrections.
7. Ask the customer for scientific clarification only when needed.
8. Offer Accept / Reject / Defer for proposals requiring review.
9. Request separate, explicit authorization before any data-changing operation.
10. Execute only supported, validated, authorized transformations on a derived copy.
11. Independently verify the result and preserve the original dataset.
12. Record decisions, authorizations, executions, and verified outcomes in persistent memory.
13. Deliver a cleaned dataset, investigation report, and audit trail when applicable.

If an operation is unsupported, evidence is insufficient, or authorization is denied, the system must not execute the change.

The customer should not need to manually run separate Python scripts for review, authorization, cleaning, or verification.

#### Shared backend requirement

The CLI and Streamlit UI must invoke the same workflow controller and backend functions.

- `backend` branch: implement and test the workflow using the terminal.
- `main` branch: expose the tested workflow through Streamlit.

Do not create separate cleaning, authorization, or memory engines for the two interfaces.

#### Current limitations

- The agent can stop at its configured investigation-step budget.
- Comprehensive coverage of arbitrary datasets is not established.
- General-purpose cleaning operations are not yet implemented.
- Only the currently supported remediation operations may execute.
- A successful investigation does not imply the dataset has been cleaned.

Next milestone: complete synthetic end-to-end backend validation,
then integrate the verified shared backend with the Streamlit UI
on the main branch.


### Terminal B — Start the Streamlit UI instead

```bash
cd /mnt/c/Users/Xavier/PycharmProjects/healthcare-data-quality-llm
source /home/xavier/.venv/bin/activate

git status --short         # Save uncommitted backend changes first
git switch main           # UI + latest MERGED backend version

PYTHONPATH=src python -m streamlit run app.py \
  --server.address 127.0.0.1 \
  --server.port 8501 \
  --browser.gatherUsageStats false
```

**Open in Windows Chrome:** <http://127.0.0.1:8501>. Keep **terminal A (Gemma)** running; use **terminal B (Streamlit)** for the website. The page is local to this PC. Upload a nonconfidential CSV to start the current read-only audit, inspect proposals and evidence, and use the human-review form. **The full CLI execution-authorization/remediation workflow is not yet connected to the UI.** Press `Ctrl+C` to stop Streamlit; stop the model separately in terminal A.

### Most frequent backend tests (no UI required)

```bash
# From project root, with the WSL virtual environment active:
PYTHONPATH=src python -m py_compile \
  src/healthcare_data_quality/agent_loop.py \
  src/healthcare_data_quality/workflow.py \
  scripts/run_investigation.py

PYTHONPATH=src python tests/test_decision_memory.py
PYTHONPATH=src python tests/test_remediation_validator.py
PYTHONPATH=src python tests/test_apply_duplicate_removal.py

git diff --check
```

Some tests may need local fixtures or prior setup. Read the test result; **a passing syntax check does not prove functional correctness**.

### Recall the complete human-approved cleaning POC (CLI)

**Do not skip gates or use a real dataset as the first test.** This supported POC concerns the synthetic duplicate-column pair at positions **2 (keep)** and **4 (drop)**. First run the investigation above and confirm it produced a matching **validated `data_change` proposal**; an `investigation` recommendation is not an executable proposal.

```bash
# 1. Preview: read-only.
PYTHONPATH=src python scripts/preview_duplicate_removal.py \
  data/synthetic/poc_duplicate_columns.csv 2 4

# 2. Recommendation review: Accept / Reject / Defer (no data change).
PYTHONPATH=src python scripts/review_proposals.py \
  /tmp/hdq_investigation.json \
  data/synthetic/poc_duplicate_columns.csv

# 3. Separate authorization: paste the ACTUAL accepted Decision ID.
PYTHONPATH=src python scripts/authorize_duplicate_removal.py \
  data/synthetic/poc_duplicate_columns.csv \
  data/derived/poc_cleaned.csv \
  --keep-position 2 --drop-position 4 \
  --decision-id YOUR_ACCEPTED_DECISION_ID

# 4. Execute: paste the ACTUAL Authorization ID from step 3.
PYTHONPATH=src python scripts/apply_duplicate_removal.py \
  data/synthetic/poc_duplicate_columns.csv \
  data/derived/poc_cleaned.csv \
  --authorization-id YOUR_AUTHORIZATION_ID

# 5. Independent post-cleaning verification.
PYTHONPATH=src python scripts/verify_remediation.py \
  data/synthetic/poc_duplicate_columns.csv \
  data/derived/poc_cleaned.csv \
  data/decision_memory/remediation_runs.jsonl \
  --outcome-log data/decision_memory/remediation_outcomes.jsonl
```

Do not reuse approval IDs, manually fabricate evidence, or overwrite an existing derived output. If `poc_cleaned.csv` already exists, choose a fresh output filename **consistently in both authorization and execution**. See [Complete CLI POC: review, authorize, remediate, verify](#complete-cli-poc-review-authorize-remediate-verify) for full safety and interpretation details.

### Quick Git commands — one folder, two development versions

```bash
git status --short

# Develop/test kernel in terminal:
git switch backend
# ... edit/test ...
git add src/ scripts/ tests/ README.md  # Review staged files; never stage data/
git diff --cached --check
git commit -m "feat: describe verified backend change"
git push origin backend

# Update the UI version once backend checks pass:
git switch main
git merge backend
# ... integrate/test Streamlit ...
git push origin main
```

Before `git switch`, **commit or stash** unfinished work. `backend` and `main` contain independent Git versions in the **same directory**, not separate copies of the engine. A backend commit is **not** available in `main` until merged. See [Git workflow](#git-workflow-one-folder-two-branches).

---


**Local-first, evidence-driven, human-governed data-quality investigation and remediation for healthcare and scientific research.**

The project combines deterministic Python checks, a locally hosted LLM, a tool-using investigation agent, evidence-backed reports, persistent human decision records, scoped execution authorization, independently verified remediation, and a Streamlit application. The intention is to help a researcher understand **what is wrong, how it was detected, what remains uncertain, and exactly what would change** before any data modification takes place.

> **Development status — November 2026:** This is an engineering prototype / proof of concept (POC), **not** a validated clinical product or a fully automated general-purpose cleaning service. Read-only investigation and the controlled duplicate-column removal path have been demonstrated. Broader cleaning operations, systematic audit coverage, arbitrary customer-data provenance, R-code generation, and complete end-to-end UI authorization are still under development. See [Current capabilities and limitations](#current-capabilities-and-limitations).

## Contents

1. [Objectives and intended users](#objectives-and-intended-users)
2. [Core design principles](#core-design-principles)
3. [System architecture](#system-architecture)
4. [Current capabilities and limitations](#current-capabilities-and-limitations)
5. [Repository layout](#repository-layout)
6. [Prerequisites and installation](#prerequisites-and-installation)
7. [Start and verify the local LLM](#start-and-verify-the-local-llm)
8. [Quick start: read-only investigation in the terminal](#quick-start-read-only-investigation-in-the-terminal)
9. [Complete CLI POC: review, authorize, remediate, verify](#complete-cli-poc-review-authorize-remediate-verify)
10. [Operate the Streamlit application](#operate-the-streamlit-application)
11. [Understanding the investigation results](#understanding-the-investigation-results)
12. [Tools, evidence, and context authorization](#tools-evidence-and-context-authorization)
13. [Human review and execution permissions](#human-review-and-execution-permissions)
14. [Decision memory and verified-outcome memory](#decision-memory-and-verified-outcome-memory)
15. [Working with different dataset types](#working-with-different-dataset-types)
16. [Tests and validation](#tests-and-validation)
17. [Git workflow: one folder, two branches](#git-workflow-one-folder-two-branches)
18. [Data privacy, security, and operational safeguards](#data-privacy-security-and-operational-safeguards)
19. [Troubleshooting](#troubleshooting)
20. [Development roadmap and acceptance criteria](#development-roadmap-and-acceptance-criteria)
21. [Project provenance and maintainer](#project-provenance-and-maintainer)

---

## Objectives and intended users

The system is intended for research analysts, data stewards, bioinformaticians, biomedical informatics teams, and other scientific dataset owners who need reproducible quality-control (QC) decisions, especially when inputs are inconsistent, poorly documented, or semantically ambiguous.

The project's five original objectives remain:

1. **Dirty ID reconciliation:** Find inconsistent or potentially equivalent subject identifiers while distinguishing plausible matches from verified identities.
2. **Duplicate resolution:** Detect repeated subjects, duplicate observations, duplicate variables, and conflicting measurements without automatically discarding records.
3. **Metadata harmonization:** Investigate inconsistent category labels, missing-value encodings, column types, and associated scientific meaning.
4. **Multi-dataset matching:** Support accurate linking of measurements and metadata, including research workflows involving brain-volume or microbiome data.
5. **LLM-assisted cleaning:** Interpret QC evidence, recommend safe changes, obtain informed human approval, and ultimately support reviewed, reproducible Python/R transformations.

**Do not confuse project objectives with implemented features.** The current end-to-end *executed* transformation is a narrowly specified, human-authorized duplicate-column removal on a derived copy. Generalized metadata harmonization, automatic record merging, and generation/execution of R cleaning scripts are future deliverables.

### What a completed customer workflow should look like

1. Upload one or more approved datasets and, where available, a data dictionary or source documentation.
2. Automatically profile and investigate all **applicable** quality categories.
3. Show verified findings with precise targets, record/column references, supporting evidence, and explicit uncertainties.
4. Continue gathering evidence with authorized read-only tools. Ask the user for domain clarification only when the system cannot establish the answer.
5. Propose exact, previewable operations; distinguish **investigation** from **data change**.
6. Let the user accept, reject, or defer a proposal; separately authorize each supported data-changing operation.
7. Write a new derived dataset without overwriting the original; independently verify retained values and write an audit record.
8. Preserve decision and outcome history to inform future investigations without silently granting future permissions.

Steps 2, 4, 6, and 7 are **only partially connected in the current UI**; use the CLI instructions below for the working authorization/remediation POC.

## Core design principles

- **Local-first inference:** The LLM is served locally. The application does not require a hosted LLM API for inference.
- **Deterministic facts first:** Python tools count, compare, validate, and fingerprint; the LLM interprets and prioritizes evidence rather than pretending to perform exact arithmetic.
- **Evidence before conclusions:** A claim cites actual observations, such as `T2.O1`, and should not exceed what those observations establish.
- **Science-aware uncertainty:** Identical values or equal-looking IDs do not establish equivalent scientific meaning.
- **Read-only by default:** Agent tools inspect data; they are not authorized to modify a dataset.
- **Separate human gates:** Accepting an investigation recommendation is **not** the same as authorizing its execution.
- **Narrow, scoped execution:** Authorization binds the source fingerprint, target columns, operation, and output path. A derived copy is created rather than editing the source.
- **Independent post-change verification:** Reinspect the output and verify that all retained cells remain unchanged.
- **Append-only provenance of decisions:** Human reviews, execution authorizations, remediation runs, and verified outcomes are recorded separately.
- **One backend, two interfaces:** The CLI and Streamlit call the same Python domain modules. They are **not** independent implementations of the cleaning rules.
- **Untrusted input:** Data values and source-document text must not be treated as instructions that override authorized tool and execution rules.

## System architecture

```text
                    Local dataset(s) + optional documentation
                                       |
                                       v
                       Ingestion / CSV structure checks
                                       |
                                       v
                  Dataset profiling + verified tool contexts
                                       |
                                       v
                 Local LLM agent (read-only tool selection)
                    |           |             |
                    v           v             v
              Column tools  Duplicate tools   Context/provenance tools
                    \           |             /
                     +----------+------------+
                                |
                                v
                    Tool observations + evidence IDs
                                |
                                v
                 Investigation report / unresolved questions
                                |
                                v
                   Remediation proposals + validator
                                |
                                v
                     Human review / decision memory
                                |
                      [separate permission gate]
                                |
                                v
                  Supported operation + derived CSV copy
                                |
                                v
                     Independent outcome verification
                                |
                                v
                     Audit and verified-outcome memory
```

### Execution modes

| Mode | Main entry point | Intended use | Important restriction |
| --- | --- | --- | --- |
| **Backend / CLI** | `scripts/run_investigation.py` and related scripts | Engineer the kernel, inspect raw evidence, run repeatable tests, and exercise authorization/remediation | Some stages are interactive terminal programs; do not run production data without review |
| **Streamlit UI** | `app.py` | Upload a dataset, view summaries/findings/proposals, and record human recommendation reviews | The full authorization → cleaning → verification controls are **not yet all wired to the UI** |
| **Shared modules** | `src/healthcare_data_quality/` | Domain logic used from both execution modes | Avoid reimplementing authorization or remediation decisions in `app.py` |

The project uses **one Git repository and one working folder** with separate `backend` and `main` branches. These are different development versions; they do not run simultaneously unless deliberately deployed from separate checkouts. See [Git workflow](#git-workflow-one-folder-two-branches).

## Current capabilities and limitations

| Capability | Current state | Notes |
| --- | --- | --- |
| CSV header/row-width inspection | **Implemented** | Duplicate names, blank names, inconsistent widths, row/column counts |
| Per-column profiling | **Implemented** | Missing/nonmissing counts, numeric prevalence, distinct-value counts, examples |
| Read-only LLM agent loop | **Implemented** | Uses registered tools; selects actions based on profile and observations |
| Evidence-indexed report | **Implemented** | `findings`, `unresolved_questions`, `recommended_next_step` |
| Remediation proposal generation | **Implemented** | Produces proposals from investigation history; includes prior review context |
| Deterministic proposal validation | **Implemented, narrow scope** | Verifies references, targets, and selected operation prerequisites |
| CLI progress events | **Implemented** | Emits progress to stderr so stdout can remain machine-readable JSON |
| Human Accept / Reject / Defer | **Implemented** | CLI and initial Streamlit form; persisted to decision memory |
| Separate data-change authorization | **Implemented in CLI** | Requires previously accepted qualifying review plus fresh, explicit confirmation |
| Remove an identical duplicate column | **Implemented in CLI** | New derived copy, exact authorized positions; source remains intact |
| Independent output verification | **Implemented in CLI** | Confirms source/output hashes and unchanged retained CSV cells |
| Verified-outcome recall | **Implemented in backend** | Recalled only for the exact source fingerprint, and never grants autonomous permission |
| Streamlit upload/preview/audit | **Implemented, partial** | UI upload limit in the reviewed version: **20 MiB**; preliminary audit, not full coverage |
| Streamlit full execution workflow | **Not yet complete** | Authorization, derived-copy execution, verification and download need UI integration |
| Full systematic audit coverage | **Not yet complete** | `max_steps=4` is a tool-call budget, not a completeness guarantee |
| Arbitrary user documentation provenance | **Not yet complete** | The current provenance tool validates a companion **synthetic-POC-only** format |
| Cross-dataset subject-ID matching | **Partial utility available** | Direct Python module exists, but is not yet a comprehensive multi-upload agent workflow |
| Taxonomy/microbiome checks | **Partial utilities available** | Some checks are implemented; agent registration/coverage remains limited |
| General metadata cleaning / record merging | **Not yet complete** | Detection and hypotheses must not be represented as safe merge authorization |
| R script generation and R execution | **Planned** | Named in project objectives; not demonstrated in current Python execution path |
| Multi-user SaaS security / compliance | **Not implemented** | No claim of HIPAA compliance, tenant isolation, or production readiness |

### A critical interpretation rule

An LLM-generated proposal with `validation.status == "needs_human_review"` is **not executable authorization**. A technically valid comparison does not independently prove semantic redundancy; the outcome verifier also records `semantic_verification: "not_established"` for the demonstrated operation.

## Repository layout

The following paths are based on the inspected source snapshot and subsequent backend/UI work. The checkout on a given branch may differ slightly.

```text
healthcare-data-quality-llm/
├── README.md
├── .gitignore
├── app.py                            # Streamlit customer-facing prototype
├── src/
│   └── healthcare_data_quality/
│       ├── agent_loop.py             # Local LLM tool-selection/investigation loop
│       ├── workflow.py               # Shared audit orchestration (recent addition)
│       ├── llm_client.py             # Local OpenAI-compatible model HTTP client
│       ├── ingestion.py              # CSV structural inspection
│       ├── dataset_profiler.py       # Per-column statistics
│       ├── profile_summary.py        # Select compact profile for agent context
│       ├── investigation_report.py   # Evidence-indexed LLM report
│       ├── remediation_proposal.py   # Proposed actions based on evidence/memory
│       ├── remediation_validator.py  # Deterministic proposal/target checks
│       ├── decision_memory.py        # Append-only review/outcome memory and hashes
│       ├── identifier_matching.py    # Two-dataset ID comparison utility
│       ├── duplicate_validation.py   # Repeated IDs and conflicting records
│       ├── taxonomy_validation.py    # Taxonomy diagnostics
│       ├── count_validation.py       # Numeric count-matrix checks
│       └── tools/
│           ├── registry.py           # Available tools and argument schemas
│           ├── context.py            # Verified tool applicability contexts
│           ├── column_inspection.py  # inspect_column(s), compare_columns
│           ├── microbiome.py         # Matrix-structure/invalid-count summaries
│           └── provenance.py         # Dataset-bound synthetic lineage check
├── scripts/
│   ├── run_investigation.py          # Read-only CLI audit -> JSON
│   ├── investigate_schema.py         # Separate schema interpretation experiment
│   ├── plan_investigation.py         # Earlier tool-planning experiment
│   ├── create_poc_dataset.py         # Generate synthetic fixture + lineage JSON
│   ├── preview_duplicate_removal.py  # Compare pair, preview only
│   ├── review_proposals.py           # Record Accept / Reject / Defer
│   ├── authorize_duplicate_removal.py # Fresh, operation-specific approval
│   ├── apply_duplicate_removal.py    # Execute authorized derived-copy change
│   └── verify_remediation.py         # Independent verification and outcome log
├── tests/
│   ├── test_decision_memory.py
│   ├── test_investigation_report.py
│   ├── test_remediation_validator.py
│   └── test_apply_duplicate_removal.py
└── data/                             # Runtime data: keep private/ignored
    ├── synthetic/                    # Synthetic local POC fixtures
    ├── derived/                      # Generated outputs, never source files
    └── decision_memory/              # Local append-only JSONL histories
```

The local model files, Python virtual environment, caches, uploaded research data, and output histories are operational assets—not code to publish.

## Prerequisites and installation

### Recommended development environment

- Windows with **WSL2 / Ubuntu** or an equivalent Linux installation.
- Python **3.12** in a virtual environment (the development environment demonstrated with WSL used Python 3.12; verify package compatibility if using another version).
- NVIDIA GPU and compatible driver/CUDA environment for the selected **local vLLM** model server; actual model requirements depend on the model/quantization.
- A running local OpenAI-compatible chat-completions endpoint.
- Git; GitHub CLI (`gh`) is optional for local development but convenient for publishing branches.
- `jsonschema` for tool argument validation; `streamlit` and `pandas` for the UI. Other local inference dependencies are part of your separately managed model-server environment.

> The reviewed source snapshot does **not** include a pinned `requirements.txt` or `pyproject.toml`; therefore the command below installs the identified direct application dependencies but is **not** a complete reproducible lockfile. Create and pin one before wider distribution.

### 1. Open the correct project directory

In WSL:

```bash
cd /mnt/c/Users/Xavier/PycharmProjects/healthcare-data-quality-llm
pwd
git branch --show-current
```

Replace the path with your own clone path on another computer. All subsequent relative paths assume the **repository root** is the current directory.

### 2. Activate the project environment

If your existing virtual environment is already configured, activate it rather than creating another:

```bash
source /home/xavier/.venv/bin/activate
python --version
```

For a new, isolated environment, an alternative is:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install the identified Python application dependencies if they are absent:

```bash
python -m pip install jsonschema pandas streamlit
```

For a production-quality setup, dependency versions and GPU-model dependencies need to be locked and tested. `python -m pip install -e .` is **not** documented because no verified packaging manifest exists in the reviewed snapshot.

### 3. Run from the repository root with the source path configured

The project uses the `src/` layout. For individual commands, prefix with:

```bash
PYTHONPATH=src python ...
```

Or export it for the current shell session:

```bash
export PYTHONPATH="$PWD/src"
```

This ensures imports such as `healthcare_data_quality.agent_loop` resolve correctly.

## Start and verify the local LLM

All model-assisted operations require the **separately running** local model service. The reviewed backend client configuration is:

| Setting | Value in reviewed `llm_client.py` |
| --- | --- |
| Chat-completions URL | `http://127.0.0.1:8000/v1/chat/completions` |
| Model identifier | `gemma4-12b` |
| Inference protocol | OpenAI-compatible HTTP JSON |
| Sampling temperature | `0` |
| Maximum generated tokens per call | `1536` |
| HTTP timeout per request | `180` seconds |

The API is configured in `src/healthcare_data_quality/llm_client.py`. If your server publishes a different model identifier, **make the client and server names consistent**. The code expects the returned message to end with `finish_reason == "stop"`; an incomplete generation causes an error.

### Start the existing local server

Use your already configured vLLM serving procedure in **terminal A**. Some working directories contain `scripts/serve_gemma4.sh`; that file was not in the inspected source archive, so the README deliberately does not invent its model path or GPU parameters. Review that script locally before running it.

### Check endpoint accessibility

In **terminal B**:

```bash
curl -fsS http://127.0.0.1:8000/v1/models
```

Confirm that the model service is reachable and that its configured model ID matches the client. A successful `/v1/models` response confirms endpoint reachability, **not** that every structured-output generation will work.

If the endpoint is unavailable, do not troubleshoot the agent first—start or repair the local model server.

### Offline requirement

Model inference is designed to use loopback networking only. Initial installation/model downloads may require Internet access; **normal inference must not depend on an external LLM API**. Do not expose the local LLM service publicly by changing its binding address without appropriate security controls.

## Quick start: read-only investigation in the terminal

This is the preferred path for developing and testing the backend kernel.

### 1. Generate an entirely synthetic test dataset

```bash
PYTHONPATH=src python scripts/create_poc_dataset.py
```

The generator creates:

- `data/synthetic/poc_duplicate_columns.csv`: 80 synthetic records and 4 columns.
- `data/synthetic/poc_duplicate_columns.provenance.json`: synthetic lineage evidence bound to the CSV SHA-256 hash.
- A duplicate `Score` column pair at **1-based positions 2 and 4**.

The companion provenance document records that both columns were generated from the same underlying synthetic score variable. This is **not** a template for making unsupported claims about real research data.

**Important:** Regenerating the fixture may change data files or fingerprints if the generator changes. Existing approvals are tied to specific source content. Do not reuse an authorization against a changed dataset.

### 2. Inspect CSV structure without the LLM

```bash
PYTHONPATH=src python src/healthcare_data_quality/ingestion.py \
  data/synthetic/poc_duplicate_columns.csv
```

This inspects column headers, row widths, blank header positions, and repeated header names; it does **not** remove columns or require inference.

Optional deeper profiling:

```bash
PYTHONPATH=src python src/healthcare_data_quality/dataset_profiler.py \
  data/synthetic/poc_duplicate_columns.csv
```

### 3. Run the complete *current* read-only agent investigation

Start the local LLM first, then:

```bash
PYTHONPATH=src python scripts/run_investigation.py \
  data/synthetic/poc_duplicate_columns.csv \
  > /tmp/hdq_investigation.json
```

The CLI emits agent progress to **stderr** (visible in the terminal), preserving machine-readable JSON on **stdout** (redirected to the file). Depending on the model's decisions you may see progress resembling:

```text
[Agent] Profiling dataset structure...
[Agent] Dataset profiling completed.
[Agent] Step 1/4: Gemma is selecting the next investigation tool...
[Agent] Step 1/4: Running compare_columns...
[Agent] Step 1/4: compare_columns completed; observations recorded.
...
[Agent] Investigation ended: model_stopped
```

Tool names, action order, and termination status vary. Common stop statuses include `model_stopped`, `max_steps_reached`, `no_new_action`, and `tools_exhausted`. **`max_steps_reached` is not evidence of a comprehensive audit.**

### 4. Validate the output JSON

```bash
python -m json.tool /tmp/hdq_investigation.json > /dev/null \
  && echo "INVESTIGATION JSON: PASS"
```

Inspect the full artifact:

```bash
python -m json.tool /tmp/hdq_investigation.json | less
```

The investigation artifact normally contains `status`, `history`, `report`, `ingestion`, and `remediation.proposals`. Each tool-history item records the tool, arguments, reason, and observations.

To inspect only the proposals:

```bash
python -c 'import json; d=json.load(open("/tmp/hdq_investigation.json")); print(json.dumps(d.get("remediation", {}).get("proposals", []), indent=2))'
```

> A valid JSON file is **not** proof that every claim, proposed operation, or dataset-quality category has been validated. Use the validator status and underlying tool evidence.

## Complete CLI POC: review, authorize, remediate, verify

This section documents the existing safe, *narrow* executable path. Use the **synthetic test fixture** and only proceed if the agent produced a qualifying `data_change` proposal with the exact expected operation and target. An `investigation` proposal cannot be substituted for a data-change approval.

### Stage 0 — Confirm prerequisites and choose output paths

The source must be the exact dataset used for investigation and review. Use a **new output filename** because the remediation code refuses to overwrite an existing output.

```text
SOURCE  = data/synthetic/poc_duplicate_columns.csv
OUTPUT  = data/derived/poc_cleaned.csv
KEEP    = 2
REMOVE  = 4
```

Inspect the generated proposal for:

- `action_kind: "data_change"`;
- `operation.kind: "drop_duplicate_column"`;
- `operation.keep_position: 2` and `operation.drop_position: 4`;
- matching `target_ref.positions: [2, 4]`;
- verified facts and `validation.status: "needs_human_review"`;
- `execution_authorized: false` (normal before separate approval).

The current deterministic validator authorizes this **proposal type for review** only when the tool history supports both identical columns and matching **synthetic-fixture** lineage. A real customer dataset with no recognized provenance may correctly receive an investigation recommendation rather than an executable change proposal. **Do not forge lineage files or hand-edit result JSON to bypass validation.**

### Stage 1 — Preview the candidate operation (no change)

```bash
PYTHONPATH=src python scripts/preview_duplicate_removal.py \
  data/synthetic/poc_duplicate_columns.csv 2 4
```

The preview checks exact row-wise equality and identical column names. It explicitly records that semantic equivalence is **not independently verified**, and that execution remains unauthorized.

### Stage 2 — Human proposal review (first approval gate)

```bash
PYTHONPATH=src python scripts/review_proposals.py \
  /tmp/hdq_investigation.json \
  data/synthetic/poc_duplicate_columns.csv
```

The reviewer supplies an ID and reviews each validated proposal:

- `A` = **accepted**: agree with the recommendation; **does not authorize cleaning**.
- `R` = **rejected**: decline the recommendation.
- `D` = **deferred**: more evidence or consideration is required.
- `S` = skip, `Q` = quit.

The reviewer may add a rationale without entering confidential subject information. The script detects previous reviews for the same dataset and proposal. Re-reviewing appends a **new** decision record; earlier history remains intact.

If a qualifying `data_change` proposal is accepted, copy the printed **Decision ID** for the next stage. If the model produced only an investigation recommendation, stop here; do not attempt to authorize a deletion.

Review records are written to:

```text
data/decision_memory/decisions.jsonl
```

### Stage 3 — Separate, explicit authorization (second approval gate)

Run with the **actual accepted Decision ID** displayed by the previous command; replace the placeholder, never the dataset/operation details:

```bash
PYTHONPATH=src python scripts/authorize_duplicate_removal.py \
  data/synthetic/poc_duplicate_columns.csv \
  data/derived/poc_cleaned.csv \
  --keep-position 2 \
  --drop-position 4 \
  --decision-id YOUR_ACCEPTED_DECISION_ID
```

The program verifies the original source fingerprint, accepted current review, exact column pair, supported evidence, output path, and that the review was not superseded. It then prompts for:

1. Reviewer ID.
2. Supporting scientific documentation reference; for the synthetic fixture, inspect `data/synthetic/poc_duplicate_columns.provenance.json` first.
3. The **exact** phrase `AUTHORIZE DROP 4 TO COPY`.

On success, it prints an **Authorization ID** and **Operation ID** and records authorization in:

```text
data/decision_memory/operation_authorizations.jsonl
```

**No data is modified at authorization time.** Missing evidence, incorrect confirmation, or an outdated review must result in refusal/deferment.

### Stage 4 — Execute only the authorized operation on a copy

Replace the placeholder with the actual Authorization ID:

```bash
PYTHONPATH=src python scripts/apply_duplicate_removal.py \
  data/synthetic/poc_duplicate_columns.csv \
  data/derived/poc_cleaned.csv \
  --authorization-id YOUR_AUTHORIZATION_ID
```

The execution script checks that:

- The authorization exists exactly once and matches the source hash, output path, operation, and positions.
- The latest qualifying human review still authorizes the proposal; a later revocation/superseding review blocks the change.
- The authorization/operation has not already been used (replay protection).
- The compared columns still have identical headers and row-wise values.
- The source cannot be overwritten, and the requested output does not already exist.
- The derived CSV has the expected rows/cells; the source hash remains unchanged.

A successful synthetic example has **80 data rows, 4 source columns, and 3 derived columns**. The script records its execution in:

```text
data/decision_memory/remediation_runs.jsonl
```

A technical cell-preservation check is **not** independent validation of scientific semantics. Follow organizational data-governance rules before any real-world deletion decision.

### Stage 5 — Independently verify the derived dataset

```bash
PYTHONPATH=src python scripts/verify_remediation.py \
  data/synthetic/poc_duplicate_columns.csv \
  data/derived/poc_cleaned.csv \
  data/decision_memory/remediation_runs.jsonl \
  --outcome-log data/decision_memory/remediation_outcomes.jsonl
```

Verification compares source/output fingerprints to the execution audit and checks every retained CSV cell against the expected transformation. The result should report:

```text
POST-REMEDIATION VERIFICATION: PASS
```

The verified-outcome log records technical success, retained-value preservation, and `semantic_verification: "not_established"`. It explicitly records `autonomous_execution_eligible: false`.

If verification fails, **do not distribute the derived output**. Inspect the failure, source/output hashes, and associated audit record; do not modify logs manually to create a passing outcome.

### Stage 6 — View the resulting artifacts

| Artifact | Meaning | Source of authority |
| --- | --- | --- |
| `/tmp/hdq_investigation.json` | Investigation history, evidence-indexed report, proposals | Agent output + validator |
| `decisions.jsonl` | Human recommendation reviews | Human review records |
| `operation_authorizations.jsonl` | Explicit, scoped execution permissions | Human authorization records |
| `data/derived/poc_cleaned.csv` | The newly generated result | Authorized transformation |
| `remediation_runs.jsonl` | Actual operation/audit event | Execution process |
| `remediation_outcomes.jsonl` | Independently verified technical outcome | Verification process |

All `data/decision_memory/*.jsonl` records are **append-only application history**, not configuration files to edit manually.

## Operate the Streamlit application

**UI development lives on the `main` branch.** Keep the project clean, then switch to `main` before running or editing the latest interface:

```bash
git status --short
git switch main
```

If you have uncommitted backend work, commit/stash it before switching. Do not use `git switch -f` or discard working changes.

### 1. Start the model service

Start the local Gemma/vLLM server in terminal A and confirm `/v1/models` responds as described above.

### 2. Start Streamlit in terminal B

```bash
PYTHONPATH=src python -m streamlit run app.py \
  --server.address 127.0.0.1 \
  --server.port 8501 \
  --browser.gatherUsageStats false
```

Open:

```text
http://127.0.0.1:8501
```

The `127.0.0.1` binding is deliberate: it limits the application to local access rather than making the research-data interface accessible from another network device.

### 3. Upload a dataset

The reviewed UI supports a **CSV file** and currently rejects files larger than **20 MiB**. A local preview shows column names, sample rows, row count, column count, and structural issues. This is a POC ingestion constraint, not an enterprise file-size guarantee.

The application also runs a read-only LLM investigation, presents findings and proposed actions, and exposes a human review form. The newer UI has an elapsed-time spinner, but true category-by-category progress/completeness and complete permission-to-execution controls are not yet finished.

### 4. Read the current results correctly

- **Structural quality findings:** Deterministic checks. A green message means no checked *header/row-width* problem was found; it does **not** certify overall dataset quality.
- **AI Data Quality Audit:** The agent's investigation output; status may include `model_stopped` or `max_steps_reached`.
- **Evidence-backed findings:** LLM report claims attached to actual tool evidence references.
- **Unresolved questions:** Issues requiring more evidence, documentation, or user clarification; they are **not** necessarily data defects.
- **Proposed actions:** Suggested investigation or data changes, with validation and uncertainty.
- **Human review:** Record `accepted`, `rejected`, or `deferred`; this is **not** execution authorization.
- **Previous decisions / outcomes:** Recalled history associated with the exact dataset. A previously seen dataset may have records; a new dataset may not.
- **Detailed investigation evidence:** Tool-by-tool observations available for technical review.

### 5. What is not yet available end to end in the UI

The current UI does not replace the complete terminal sequence in [Complete CLI POC](#complete-cli-poc-review-authorize-remediate-verify). In particular, the operator should **not** assume that pressing **Accept** removes any column or changes data. The UI still needs properly wired operation-specific authorization, derived-copy creation, independent verification, and downloadable audit artifacts.

### 6. Shut down cleanly

Press `Ctrl+C` in the Streamlit terminal. Stop the vLLM server separately using its own startup process. Stopping the UI does not imply that model weights, review histories, or derived files are deleted.

## Understanding the investigation results

### Evidence identifiers

Each tool step records observations in a `history` array. `investigation_report.build_evidence_index()` assigns IDs of the form:

```text
T1.O1  = first observation in tool step 1
T2.O1  = first observation in tool step 2
T2.O2  = second observation in tool step 2
```

An evidence ID permits traceability to a specific tool observation. The report checks that referenced IDs exist. It does **not** independently establish that every natural-language interpretation is scientifically correct.

### Findings vs. hypotheses vs. decisions

| Concept | Example | Meaning |
| --- | --- | --- |
| **Verified observation** | Two columns have identical cell strings and missingness | A deterministic comparison established this fact |
| **Hypothesis** | Both columns represent the same scientific variable | Plausible, but requires lineage/domain confirmation |
| **Unresolved question** | Was the second column intentionally duplicated? | Relevant evidence is missing or insufficient |
| **Investigation proposal** | Request source documentation and inspect columns | Read-only next step |
| **Data-change proposal** | Drop a specific redundant column from a derived copy | Candidate action subject to validation and human gates |
| **Accepted review** | Reviewer agrees with the candidate | Persisted human decision; **not execution permission** |
| **Authorization** | Reviewer approves one exact operation/output | Separate, scoped permission |
| **Verified outcome** | Retained CSV values match expectation | Technical result, not general semantic approval |

### Common statuses and flags

- `model_stopped`: Gemma selected stop; **not necessarily complete coverage**.
- `max_steps_reached`: The configured tool-call budget was exhausted.
- `tools_exhausted`: No currently permitted tool remained in the loop.
- `no_new_action`: The loop could not obtain a nonrepeated new action.
- `needs_human_review`: The validator permits review but **not execution**.
- `rejected`: A proposal failed current validator checks.
- `execution_authorized: false`: The proposal itself is not an execution permit.
- `execution_status: "not_executed"`: A review/authorization stage has not yet applied a change.
- `technical_verification: "passed"`: The specified output checks passed.
- `semantic_verification: "not_established"`: Scientific equivalence was not independently proven by the verifier.

Do not treat a low LLM temperature as a proof of reproducibility. Model server versions, prompts, tool results, and environment can still change outcomes.

## Tools, evidence, and context authorization

The registered investigation tools in the inspected `tools/registry.py` include:

| Tool | Function | Preconditions |
| --- | --- | --- |
| `observe_matrix_structure` | Inspect matrix shape and numeric characteristics without assuming semantics | CSV header available |
| `inspect_column` | Profile one 1-based column | CSV header available |
| `inspect_columns` | Profile up to four distinct 1-based columns in a call | CSV header available |
| `compare_columns` | Compare two columns row by row | CSV header available |
| `inspect_provenance` | Check a dataset-bound synthetic lineage companion document | CSV header available |
| `find_duplicate_ids` | Detect repeated `SampleID`/`NPDNum` identifiers | Both metadata fields present exactly once |
| `find_duplicate_conflicts` | Identify conflicting fields among duplicate `SampleID` records | Both metadata fields present exactly once |
| `summarize_invalid_counts` | Summarize invalid values in a confirmed raw count matrix | Context must be independently verified |

### Why some tools do not appear

`verify_contexts()` currently recognizes a general CSV-header context and the specific metadata context containing exactly one `SampleID` and one `NPDNum` header. Not every scientific dataset qualifies for every tool. A raw count-matrix tool must not be selected solely because a CSV contains numbers.

Tool execution validates JSON-schema arguments and rejects unregistered or context-ineligible tools. The LLM chooses from what the backend permits; it cannot expand its own permission scope by writing a persuasive instruction.

### Other existing utilities not yet integrated into full automatic coverage

- `identifier_matching.py`: exact/normalized cross-dataset ID comparison and suspected matches that require review.
- `taxonomy_validation.py`: duplicate taxa, unclassified-species, and naming-pattern diagnostics.
- `count_validation.py`: count-value checks for appropriate numeric matrices.

These utilities exist, but **a function existing in the repository does not imply the agent currently runs it for every applicable dataset**. The planned coverage controller must make applicability and completion explicit.

### One-based positions and CSV record numbering

Column positions are **1-based** (the first column is `1`), while Python list indexes are **0-based**. Some reported CSV record numbers include the header row, so the first data record is record `2`. Review operation targets carefully—particularly when two columns have the same header name.

## Human review and execution permissions

The system deliberately separates four distinct decisions:

1. **Investigation:** Which safe, read-only evidence should the agent gather?
2. **Recommendation review:** Does a named human accept, reject, or defer a proposed action?
3. **Execution authorization:** Does a human explicitly permit one *validated, supported* change to one source, one target, and one output path?
4. **Outcome verification:** Did the authorized change actually produce the expected, preserved output?

The current duplicate-column CLI requires an accepted **current** review and an additional authorization record; the authorizer rejects a superseded decision and the executor checks authorization replay, output redirection, current review state, and source integrity.

### Why two identical columns are not automatically dropped

Identical cell values do not prove that two columns are exchangeable. They might represent separate replicates, time points, measurement methods, or intentional duplicated reporting fields. Domain documentation or a responsible reviewer must establish the rationale. Even with documented provenance, the independent technical verifier reports scientific semantic validation as **not established**.

### Supported vs. unsupported remediation

**Supported executable CLI operation in this POC:** `drop_duplicate_column` for an explicitly authorized, fully identical pair, with safe derived-copy execution and independent verification.

**Not currently a general executable cleaning API:** arbitrary column deletion, uncertain identifier merges, duplicate subject deletion, taxonomy corrections, metadata harmonization, row-level imputation, or model-generated R transformations. These must remain proposals/investigations until validated operation handlers are developed.

## Decision memory and verified-outcome memory

### Memory storage

| File | Typical contents | Written by |
| --- | --- | --- |
| `data/decision_memory/decisions.jsonl` | Decision ID, dataset/proposal hashes, target/evidence, reviewer, accept/reject/defer, notes | Human review |
| `data/decision_memory/operation_authorizations.jsonl` | Authorization ID, operation ID, current review link, source hash, output path and exact operation | Separate authorizer |
| `data/decision_memory/remediation_runs.jsonl` | Executed operation, source/output fingerprints, source preservation, derived-copy audit | Executor |
| `data/decision_memory/remediation_outcomes.jsonl` | Independent technical verification and autonomous-execution ineligibility | Outcome recorder |

JSONL means **one independent JSON object per line**. Appending records rather than overwriting existing decisions enables review history and supersession checks.

### How review memory is matched

- `fingerprint_dataset()` hashes source file **bytes** with SHA-256.
- `fingerprint_proposal()` hashes specified proposal fields, including issue/action/target/evidence IDs.
- `find_latest_decision()` looks for the exact dataset + proposal fingerprints and returns the last matching record.
- `get_decision_context()` returns recent valid reviews for the **exact dataset fingerprint**, with the latest record per proposal.
- `get_verified_outcome_context()` returns previously verified technical outcomes for that exact dataset, with autonomous execution still disabled.

**Important limits:** A one-byte file change changes its hash, and differently worded proposals may produce different proposal hashes. This is deliberate conservative matching, not semantic similarity memory. No current memory entry should automatically authorize cleaning a new dataset or a new operation.

### What “learning from memory” currently means

The proposal generator can receive relevant historical review decisions to improve recommendations. Verified outcomes can be recalled as evidence of *previous technical execution*. The present system is **not** continuously training model weights, proving general scientific rules from isolated approvals, or autonomously approving future changes. Generalization and safe policy learning are future research and engineering work.

### Retention and corruption considerations

Memory records may contain reviewer identifiers, notes, and dataset fingerprints. Protect them as potentially sensitive operational metadata. Hashes reduce accidental mismatch but are **not** encryption, access control, or an authenticity signature. The files are local and append-only by application convention, not tamper-proof against a filesystem administrator.

## Working with different dataset types

| Dataset situation | Recommended current entry point | Caveat |
| --- | --- | --- |
| Any well-formed CSV | `ingestion.py` / `dataset_profiler.py` | Structural checks alone do not establish quality |
| Unknown scientific CSV | `scripts/run_investigation.py` | Read-only exploratory audit with limited tool budget |
| Metadata with `SampleID`, `NPDNum` | `duplicate_validation.py` and registered duplicate tools | Duplicate IDs do not prove duplicate subjects |
| Taxonomy-style table | `taxonomy_validation.py` | Interpret findings in the correct scientific context |
| Two datasets with subject identifiers | `identifier_matching.py` | Utility produces candidates; not an authorized auto-merge |
| Suspected duplicate columns | `preview_duplicate_removal.py` and `compare_columns` | Exact equality != scientific equivalence |
| Validated synthetic POC fixture | Full CLI review/authorization/verification sequence | Synthetic lineage evidence only; not a real-data validation policy |

For a user-supplied dataset, a suitable workflow should first identify structural type, applicable checks, data dictionary availability, and expected relationships. There is currently no guarantee that arbitrary uploaded datasets receive every relevant check or that all uncertainties can be resolved without human input.

## Tests and validation

### Syntax checks (offline, no model required)

```bash
PYTHONPATH=src python -m py_compile \
  src/healthcare_data_quality/agent_loop.py \
  src/healthcare_data_quality/decision_memory.py \
  src/healthcare_data_quality/remediation_validator.py \
  scripts/run_investigation.py \
  scripts/authorize_duplicate_removal.py \
  scripts/apply_duplicate_removal.py \
  scripts/verify_remediation.py
```

If working on `main`, also compile `app.py` and `workflow.py`:

```bash
PYTHONPATH=src python -m py_compile \
  app.py src/healthcare_data_quality/workflow.py
```

### Targeted automated tests

Run these **portable synthetic/mocked tests** from the repository root:

```bash
PYTHONPATH=src python tests/test_decision_memory.py
PYTHONPATH=src python -m unittest tests/test_investigation_report.py
PYTHONPATH=src python tests/test_apply_duplicate_removal.py
```

The last test exercises synthetic review, separate authorization, unsafe-output rejection, replay/revocation protections, derived-copy preservation, independent verification, and verified-outcome memory. These tests were exercised against the reviewed snapshot without a running model server.

**Fixture-dependent test:** The reviewed version of `tests/test_remediation_validator.py` references a separately supplied local dataset under `data/` (not part of the public source archive). It will fail with `FileNotFoundError` in a clean checkout. Only run it when you have the specifically authorized local fixture, or first refactor the test to generate synthetic inputs:

```bash
# Run only if the test's required authorized local fixture exists:
PYTHONPATH=src python tests/test_remediation_validator.py
```

Do **not** commit a confidential fixture merely to make a test pass. Converting remaining fixture-dependent tests to generated synthetic data is a reproducibility priority. The script-style tests and unittest-style tests use different entry patterns; do not assume all tests are automatically discovered in the same way. The **live investigation** does require the model server.

### Live functional smoke test

```bash
PYTHONPATH=src python scripts/create_poc_dataset.py
PYTHONPATH=src python scripts/run_investigation.py \
  data/synthetic/poc_duplicate_columns.csv \
  > /tmp/hdq_smoke.json
python -m json.tool /tmp/hdq_smoke.json > /dev/null
```

For a full functional assessment, review the history, validator statuses, and expected evidence—not only a successful exit code. Record exact command, source commit, model name, observations, and outcome outside the confidential-data repository.

### Before committing code

```bash
git status --short
git diff --check
git diff --cached --check
```

If editing source files on Windows/WSL, use **LF line endings** to avoid CRLF-related Git whitespace warnings. `sed -i 's/\r$//' path/to/file.py` can normalize a known affected text file, but first review what will change.

## Git workflow: one folder, two branches

The project intentionally uses **one working directory** with two Git versions:

| Branch | Purpose | Files it contains |
| --- | --- | --- |
| `backend` | Develop and validate the core offline Python kernel and CLI | Complete repository history; primarily change `src/`, relevant `scripts/`, tests, and shared documentation |
| `main` | Develop the customer-facing Streamlit UI, using tested backend updates | Complete integrated application, including the backend; primarily change `app.py` and UI/integration assets |

**Git branches are repository versions, not an API/client boundary.** `main` includes the backend code it needs. Do not split the code into unrelated engines or duplicate the cleaning checks in Streamlit.

### Normal backend development cycle

```bash
# Start with a clean working tree
git status --short

git switch backend
# Edit src/, scripts/, tests/ and run appropriate CLI tests

git add src/ scripts/ tests/ README.md
git diff --cached --check
git diff --cached --stat
git commit -m "feat: describe tested backend change"
git push origin backend
```

`git add` is illustrative: stage only the files actually changed and reviewed. Do **not** stage private datasets or runtime JSONL files. If no README changes were made, omit it.

### Bring tested backend changes into the Streamlit version

```bash
# Commit or stash all unfinished backend edits first
git status --short
git switch main
git merge backend

# Test integration and make UI changes here
PYTHONPATH=src python -m py_compile app.py

git push origin main
```

If `main` has independent commits, Git may create a merge commit or report conflicts. Review conflicts manually, especially imports and function signatures; never overwrite UI changes reflexively. **Do not develop separate implementations of the same backend rule on both branches.**

### Switching safely in one folder

Always check `git status --short` before switching. Commit a coherent checkpoint or use `git stash push -m "wip: reason"` for unfinished changes. Do not use `git switch --force`, `git reset --hard`, or `git worktree add` merely to alternate between the two requested versions.

### README maintenance

This README is shared documentation. Update it on `backend` for backend behavior and merge into `main`, then add UI-specific instructions on `main` when the interface changes. The two branches may temporarily describe slightly different code revisions; confirm the current branch when following version-dependent commands.

## Data privacy, security, and operational safeguards

1. **Never commit confidential research or patient datasets.** The repository is intended for source code and synthetic or explicitly authorized samples.
2. **Keep local state out of Git:** `data/`, `models/`, `.venv/`, model caches, derived outputs, decision memory, notebook checkpoints, secrets, and uploaded files should be ignored. Verify the effective rules with `git status --short` and `git check-ignore -v <path>`; do not assume `.gitignore` is correct without checking.
3. **Local-only web and model endpoints:** Use `127.0.0.1` for Streamlit and the inference server in development. Remote or multi-user exposure needs authentication, TLS, isolation, authorization controls, and appropriate network policy.
4. **No automatic edits from prompts:** Model instructions to delete, merge, or rewrite are proposals, never implicit permission.
5. **Sensitive-text risk:** Prompts, error logs, model input, and UI previews may still contain dataset-derived information. Local hosting does not automatically make the workflow compliant with any specific privacy law or institutional data-use agreement.
6. **Dataset integrity:** SHA-256 fingerprints identify exact input/output bytes. The executor preserves original source content by writing a derived output path and checking it; independent verification checks retained CSV values.
7. **Separation of roles:** Human review and operation authorization are distinct states. Production use will require authenticated identities, role-based permissions, audit integrity, and session/workspace isolation.
8. **Least-authority tools:** Only validated, context-eligible, read-only tools may run during investigation. The cleaning executor is a separate controlled path.
9. **No trust escalation from memory:** Previous approval and prior technical success are not blanket permission for future datasets.
10. **Data governance before production:** Organizational approvals, access controls, retention/deletion policy, software validation, backups, monitoring, incident response, and a threat model are still required.

The present project does **not** claim HIPAA certification, HIPAA compliance by default, clinical validation, security certification, or safe autonomous operation on confidential institutional datasets.

## Troubleshooting

### `ModuleNotFoundError: No module named 'healthcare_data_quality'`

Run from the repository root with `PYTHONPATH=src`, or export `PYTHONPATH="$PWD/src"`. Verify the correct virtual environment is active.

### `Connection refused` / local model errors / no response

Verify the server process, `http://127.0.0.1:8000/v1/models`, `MODEL_URL`, and `MODEL_NAME`. A 180-second request timeout can result from slow inference or a stalled model; do not mistake the spinner for a guarantee of forward progress.

### `Incomplete generation` or JSON parsing errors

The client requires a complete response with `finish_reason == "stop"`. Investigate token budget, server logs, structured-output support, and model identifier compatibility before modifying downstream validation.

### Audit says `max_steps_reached`

This is the current budget, **not** a completed audit of all applicable categories. Raising `max_steps` alone is not a coverage policy. The planned controller must explicitly report category applicability, attempts, completion, evidence gaps, and unsupported checks.

### A `provenance` result says `not_available`

The current provenance tool looks for a companion file matching `<dataset-stem>.provenance.json` and validates a restricted synthetic-POC format. Absence is **not** proof that the scientific dataset lacks provenance, only that this tool did not find usable evidence in its supported format.

### Agent proposes investigation instead of deletion

This is often correct. Missing lineage, insufficient semantic evidence, incorrect target references, or unsupported operation types should block execution. Do not change the proposal JSON manually to force `data_change`.

### Review accepted, but cleaning did not run

Expected behavior. A recommendation review is not execution authorization. Complete the separate authorization CLI with the exact current accepted decision ID and supporting documentation, then execute using the resulting Authorization ID.

### `Review decision has been superseded`

A newer decision for the same dataset/proposal exists. Reevaluate and obtain authorization from the **latest** qualified review. Do not reuse the older ID.

### `Authorization already used`, existing output, or output-path mismatch

These are intentional safety checks. Do not replay an authorization, redirect an output, or overwrite an existing derived file. Create a new validated operation and obtain fresh scoped permission if another run is truly required.

### Streamlit spinner shows elapsed time but not tool activity

The elapsed-time indicator is useful but does not prove the model is progressing. Backend CLI progress callbacks exist; full live UI activity-event propagation and coverage-state reporting are still being refined. Treat a long period without backend events as something to diagnose through the terminal/model logs, not evidence of successful analysis.

### WSL logs `gio: http://127.0.0.1:8501: Operation not supported`

Streamlit's attempt to open a browser automatically may fail in WSL. If the server announces the local URL, open `http://127.0.0.1:8501` manually in Windows Chrome. This message alone does not indicate that the web application failed.

### Git says `trailing whitespace` or displays `^M`

A source file may have CRLF line endings. Normalize only the intended text files to LF, re-stage them, and rerun `git diff --cached --check`. Do not apply bulk rewrites to confidential datasets or generated artifacts.

### GitHub authentication fails on password prompt

GitHub HTTPS Git operations require token-based authentication. When appropriate, use `gh auth login`, then `gh auth setup-git`; never paste GitHub credentials into source, README, or screenshots. GitHub authentication is separate from the Linux password used by `sudo`.

## Development roadmap and acceptance criteria

This roadmap separates working prototypes from required engineering work. It does not imply fixed release dates.

### Phase A — Backend kernel coverage (next priority)

- Define an explicit quality-check catalog and applicability policy.
- Execute all safe, applicable deterministic baseline checks automatically, **independent of whether the model happens to select them**.
- Track each check as `completed`, `needs_more_evidence`, `not_applicable`, `unsupported`, or `failed` rather than pretending four tool calls equal comprehensive coverage.
- Let the LLM investigate unusual or ambiguous findings with bounded, nonrepeating, read-only tool calls.
- Implement structured escalation: additional evidence → documentation request → human clarification → validated proposal.
- Emit truthful progress events and coverage summaries for both the CLI and UI.

**Acceptance criterion:** For every uploaded test dataset, a machine-readable coverage report identifies each applicable check, its status, evidence, and reason for any gap. No percentage is invented for an open-ended agent loop.

### Phase B — Generalize evidence and safe remediation

- Support uploaded data dictionaries and dataset-specific provenance with explicit trust boundaries.
- Add domain-review workflows for subject-ID reconciliation, duplicate subjects, categorical normalization, and multi-dataset matching.
- Add separate, deterministic executors and independent verifiers *per* supported operation.
- Record exact before/after hashes, row/column effects, retained-data invariants, and human approval scope.
- Treat generated R code as a reviewed artifact; introduce schema validation, tests, constrained execution, and reproducibility before enabling it.

**Acceptance criterion:** Each newly supported data-change operation has negative-path tests, exact targets, an explicit authorization record, rollback/recovery handling, and independent post-change verification.

### Phase C — Complete the customer UI

- Provide one unified findings screen with verified facts, suspected problems, affected records, uncertainties, and detailed evidence on demand.
- Automatically run the comprehensive audit after upload without requiring users to select investigation buttons.
- Show an actual activity log (tool/stage events), elapsed time, and explicit incomplete-coverage warnings.
- Allow clarification/document upload and targeted re-investigation when uncertainty cannot be resolved internally.
- Surface existing Accept/Reject/Defer, separate execution authorization, derived-copy transformation, verification, audit, and downloads through reusable backend functions.
- Refresh review and outcome memory without repeating unnecessary model calls.

**Acceptance criterion:** A user can complete the same authorized POC from the UI as from the CLI, and both paths yield equivalent audit/verification records.

### Phase D — Security, testing, and deployment readiness

- Pin dependencies, provide environment bootstrap scripts and model/server configuration docs.
- Add realistic synthetic dataset families, schema-validation/robustness tests, and targeted adversarial-input checks.
- Define multi-tenant storage/workspace isolation, authenticated reviewers, authorization policies, concurrency control, and privacy-preserving logging.
- Implement trustworthy audit persistence, recovery from partial writes, cancellation/timeouts, and operational monitoring.
- Assess domain-specific regulatory and institutional requirements before handling confidential customer data.

**Acceptance criterion:** Documented security review, reproducible deployment, validated permissions, safe handling of concurrent operations, and evidence-based acceptance criteria for relevant quality categories.

## Project provenance and maintainer

This README expands the project's original five objectives and development approach while documenting the current Python CLI and Streamlit prototypes. The initial README described work in Python and R; the **reviewed executable code is primarily Python**, with R-based cleaning still an objective rather than a demonstrated feature.

**Maintainer:** Xavier Huang  
**GitHub:** [XavierHuangWF](https://github.com/XavierHuangWF)  
**Project repository:** [healthcare-data-quality-llm](https://github.com/XavierHuangWF/healthcare-data-quality-llm)  
**Contact:** william.techlover@gmail.com

When updating this document, prefer **verified functionality and exact commands** over aspirational claims. Distinguish tested, partially implemented, and planned features; never include confidential dataset names, contents, private paths, patient information, model credentials, or customer evidence in public documentation.
