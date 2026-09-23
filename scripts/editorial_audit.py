"""Recompute manuscript evidence from saved predictions; no training or selection.

Run with the experiment environment: python scripts/editorial_audit.py --root PATH.
Outputs contain aggregate evidence and hashes, never raw spectra.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def ap_columns(y, p, ties=False):
    order = np.argsort(-p, axis=0, kind="stable")
    ys = np.take_along_axis(y, order, axis=0)
    tp = np.cumsum(ys, axis=0, dtype=float)
    if not ties:
        return (tp / np.arange(1, len(y)+1)[:, None] * ys).sum(0) / ys.sum(0)
    ps = np.take_along_axis(p, order, axis=0)
    result = []
    for j in range(y.shape[1]):
        ends = np.r_[np.flatnonzero(ps[1:, j] != ps[:-1, j]), len(y)-1]
        increments = np.diff(np.r_[0, tp[ends, j]])
        result.append(np.sum(tp[ends, j] / (ends+1) * increments) / tp[-1, j])
    return np.array(result)


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    repo = Path(__file__).resolve().parents[1]
    dest = repo / "results" / "editorial_audit"
    dest.mkdir(exist_ok=True)
    seeds = [2026, 7, 13]
    arms = ["M0", "M0T", "M1D", "M1R", "M1", "M2"]
    means, aps, rows, strata = {}, {}, [], []
    reference = None
    inputs = []
    for arm in arms:
        seed_ap = []
        for seed in seeds:
            tag = arm + ("" if seed == 2026 else f"_s{seed}")
            folder = root / "dbf2_runs_open" / tag
            path = folder / f"{tag}_test_predictions.npz"
            d = np.load(path, allow_pickle=False)
            if reference is None:
                reference = {k:d[k].copy() for k in ("target", "group_id", "n_views", "n_views_total")}
            for key in reference:
                if not np.array_equal(reference[key], d[key]):
                    raise ValueError(f"Unpaired predictions: {tag}: {key}")
            y, p = d["target"], d["prob"]
            if not np.isfinite(p).all():
                raise ValueError(f"Nonfinite predictions in {tag}")
            n = y.sum(0)
            scored = (n >= 5) & (n <= len(y)-5)
            ap = ap_columns(y[:,scored], p[:,scored])
            tie_ap = ap_columns(y[:,scored], p[:,scored], ties=True)
            seed_ap.append(ap)
            report_path = folder / f"{tag}_report.json"
            report = json.loads(report_path.read_text())
            assert abs(ap.mean()-report['test']['macro_auprc']) < 1e-10, tag
            cfg = json.loads((folder/'config.json').read_text())
            rows.append(dict(arm=arm,seed=seed,macro_auprc=float(ap.mean()),
                             threshold_ap=float(tie_ap.mean()),parameters=report['n_parameters'],
                             views_per_molecule=cfg['train']['views_per_molecule'],
                             **{k:report['test'][k] for k in ('macro_auroc','micro_auprc','brier')}))
            if arm in ('M1','M2'):
                for label, mask in [('1',d['n_views']==1),('2',d['n_views']==2),('3-4',d['n_views']>=3)]:
                    sy, sp = y[mask], p[mask]
                    sn = sy.sum(0)
                    keep = (sn>=5)&(sn<=len(sy)-5)
                    strata.append(dict(arm=arm,seed=seed,views=label,n_molecules=int(mask.sum()),
                                       n_targets=int(keep.sum()),macro_auprc=float(ap_columns(sy[:,keep],sp[:,keep]).mean())))
            inputs.append(dict(path=str(path.relative_to(root)),sha256=sha(path)))
            print(f"{tag}: {ap.mean():.8f}; threshold AP {tie_ap.mean():.8f}", flush=True)
        aps[arm] = np.stack(seed_ap)
        values = aps[arm].mean(1)
        means[arm] = dict(mean=float(values.mean()),sd=float(values.std(ddof=1)),values=values.tolist())
    rng = np.random.default_rng(20260923)
    ix = rng.integers(0,int(scored.sum()),size=(5000,int(scored.sum())))
    pairs = [("M0T","M0"),("M0T","M1"),("M0T","M2"),("M1","M1R"),("M2","M1"),
             ("M1R","M0"),("M1D","M0"),("M1R","M1D"),("M1","M0")]
    contrasts = {}
    for hi,lo in pairs:
        delta=(aps[hi]-aps[lo]).mean(0)
        boot=delta[ix].mean(1)
        values=(aps[hi]-aps[lo]).mean(1)
        contrasts[f'{hi}-{lo}']=dict(mean=float(delta.mean()),lo=float(np.quantile(boot,.025)),
            hi=float(np.quantile(boot,.975)),seed_sd=float(values.std(ddof=1)),values=values.tolist(),
            fraction_improved=float((delta>0).mean()))
    old=json.loads((repo/'results/seed_analysis_open.json').read_text())['by_rung']['M0']
    # The overwritten unconditioned prediction files are not reconstructed from summaries.
    historical=dict(**old,status='historical summary only; paired target uncertainty unavailable')
    hist_delta=np.array(means['M0']['values'])-np.array(old['values'])
    historical['metadata_change_mean']=float(hist_delta.mean())
    historical['metadata_change_seed_sd']=float(hist_delta.std(ddof=1))
    pd.DataFrame(rows).to_csv(dest/'arm_metrics.csv',index=False)
    pd.DataFrame(strata).to_csv(dest/'aggregation_by_views.csv',index=False)
    per_target=pd.DataFrame({'target_index':np.flatnonzero(scored)})
    for arm in arms:
        for i,seed in enumerate(seeds): per_target[f'{arm}_seed{seed}']=aps[arm][i]
    per_target.to_csv(dest/'per_target_ap.csv',index=False)
    raw_path=root/'data/train_open.parquet'
    raw=pq.read_table(raw_path,columns=['inchikey14','normalized_smiles','ingest_lib']).to_pandas()
    variants=raw.groupby('inchikey14').normalized_smiles.nunique()
    cols=pq.ParquetFile(raw_path).schema_arrow.names
    provenance=dict(source_file='data/train_open.parquet',sha256=sha(raw_path),n_rows=len(raw),
        source_counts={k:int(v) for k,v in raw.ingest_lib.value_counts().items()},
        license_fields=[c for c in cols if 'licen' in c.lower()],
        source_id_fields=[c for c in cols if any(s in c.lower() for s in ('accession','spectrum_id','usi'))],
        identity_groups_with_multiple_smiles=int((variants>1).sum()),
        redistribution_verified=False,
        limitation='Library selection does not establish record-level licences or independent reconstruction.')
    mol=pq.read_table(root/'dbf2_prepared_open/molecules.parquet').to_pandas()
    provenance['prepared_hashes']={name:sha(root/'dbf2_prepared_open'/name) for name in
        ['manifest.json','molecules.parquet','targets.parquet','target_dictionary.csv']}
    result=dict(n_molecules=len(reference['group_id']),n_targets=int(scored.sum()),
                prevalence_reference=float(reference['target'][:,scored].mean()),
                max_views_used=int(reference['n_views'].max()),
                molecules_with_more_views_than_used=int((reference['n_views_total']>reference['n_views']).sum()),
                historical_unconditioned=historical,arms=means,contrasts=contrasts,
                target_resampling=dict(draws=5000,seed=20260923,unit='target',fixed_molecules=True),
                provenance=provenance,inputs=inputs)
    (dest/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    # Analytical checks of AP tie handling (rank-order AP remains the historical estimand).
    assert np.isclose(ap_columns(np.array([[1],[0]]),np.array([[.5],[.5]]),True)[0],.5)
    assert np.isclose(ap_columns(np.array([[1],[0]]),np.array([[.9],[.1]]),True)[0],1)
    assert np.isclose(ap_columns(np.array([[0],[1]]),np.array([[.9],[.1]]),True)[0],.5)
    print(json.dumps({k:v for k,v in result.items() if k!='inputs'},indent=2))


if __name__=='__main__': main()
