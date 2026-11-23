"""A failed Agent B request can be retried once without replacing earlier successes."""
from healthcare_data_quality.jobs.cleaning_jobs import CleaningRecommendationJobs
from healthcare_data_quality.services.llm_client import _RequestGate


def _finding(number):
    return {'id': f'dataset_1:qc:{number}', 'dataset_id': 'dataset_1',
            'source': 'QC', 'title': f'Finding {number}', 'check_key': 'counts',
            'actionable': True, 'evidence': []}


def test_one_explicit_retry_only_for_failed_finding(tmp_path):
    calls = {}
    def recommender(item):
        fid = item['id']; calls[fid] = calls.get(fid, 0) + 1
        if fid.endswith(':1') and calls[fid] == 1:
            raise ValueError('Simulated model schema error')
        return {'suggestion': 'Replace evidenced values with NA after review',
                'decision_needed': 'Needs approval', 'choices': []}
    worker = CleaningRecommendationJobs(tmp_path / 'queues', recommender=recommender)
    findings = [_finding(1), _finding(2)]
    key = worker.start('test-run', findings, run_id='a', retry_failed=False)
    worker._workers[key].join(3)
    snap = worker.snapshot(key)['entries']
    assert snap[findings[0]['id']]['status'] == 'failed'
    assert snap[findings[1]['id']]['status'] == 'ready'
    # Ordinary reruns are inert: no model calls repeat.
    worker.start('test-run', findings, run_id='a')
    assert calls == {findings[0]['id']: 1, findings[1]['id']: 1}
    worker.start('test-run', findings, run_id='a', retry_failed=True)
    worker._workers[key].join(3)
    new = worker.snapshot(key)['entries']
    assert new[findings[0]['id']]['status'] == 'ready'
    assert new[findings[0]['id']]['manual_retries'] == 1
    assert calls == {findings[0]['id']: 2, findings[1]['id']: 1}
    worker.start('test-run', findings, run_id='a', retry_failed=True)
    assert calls == {findings[0]['id']: 2, findings[1]['id']: 1}
