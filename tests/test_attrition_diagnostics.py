from src import data as D
from src.diagnose_attrition import make_report, STAGES


def test_attrition_counts_reconcile_to_build_pipeline():
    raw = D.filter_frame(D.make_synthetic_raid(n_per_domain=12, seed=4),
                         decoding="greedy", repetition_penalty="no")
    stages, truncation, matched = make_report(raw, max_docs=12)
    assert set(stages.stage.unique()) == set(STAGES)
    assert (stages.rows_removed_since_previous.dropna() >= 0).all()
    assert len(truncation) == len(D.DOMAINS) * (len(D.GENERATORS) + 1)
    built = D.build_dataset(raw, max_docs=12)
    actual = built.groupby(["domain", "generator", "split", "label"]).size()
    for r in matched.itertuples():
        assert int(actual.get((r.domain, r.generator, r.split, r.label), 0)) == r.n_after_matching
