"""Session-local identities for expensive review rendering work."""
from hashlib import sha256
import json
from pathlib import Path


def checkpoint_identity(state, sources, titles):
    """Hash every durable input. Do not cache failures or share across sessions."""
    from healthcare_data_quality.cleaning.cleaning_stage_checkpoint import _reviewed
    fields = ('cleaning_approved_columns_by_dataset', 'cleaning_approved_rows_by_dataset',
              'cleaning_history', 'cleaning_approved_reviewer',
              'cleaning_completed_audits_by_finding', 'cleaning_decisions')
    payload = {key: state.get(key) for key in fields}
    payload['statuses'] = {k: v for k, v in state.get('cleaning_review_statuses', {}).items()
                           if isinstance(v, str) and _reviewed(v) and k in titles}
    payload['titles'] = titles
    payload['sources'] = [(str(Path(p)), sha256(Path(p).read_bytes()).hexdigest()) for p in sources]
    payload['staged'] = {k: sha256(v).hexdigest() for k, v in state.get('cleaning_staged_bytes_by_dataset', {}).items()}
    return sha256(json.dumps(payload, sort_keys=True, default=str, ensure_ascii=False).encode()).hexdigest()
