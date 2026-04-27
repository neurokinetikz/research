"""
Pediatric SR × β coupling artifact controls (paper-staging Q2).

The cross-cohort meta-analysis flagged that pediatric ρ > 0.8 could be
artifact-driven (HBN R2 +0.811 explicitly flagged; HBN R7 +0.782 and R11
+0.825 are the two highest-ρ cohorts). Full ICA-cleaned re-extraction is
multi-hour batch work that belongs on the VM. This script runs four
lightweight robustness controls locally on HBN R7 / R11 (and R1 / R10
as comparator releases) using the existing per-subject CSVs:

1. **n_events threshold sensitivity** — recompute Spearman ρ at successive
   minimum-event-count cutoffs (≥ 3, 5, 7, 10). If ρ collapses at higher
   thresholds, low-data subjects drive the effect; if ρ is stable, the
   coupling reflects subjects with substantial data, not a few noisy
   ones.

2. **Per-event template_ρ filtering (Q4-only proxy)** — restrict subjects
   to those whose median per-event template_ρ exceeds a quality
   threshold (e.g., 0.3). Tests whether the coupling is driven by
   subjects with canonical-shape events rather than noise-like
   detections.

3. **Outlier exclusion (winsorization)** — exclude subjects with
   extreme amplitude ratios (top 1% or top 5% in either SR or β
   amplitude). Tests whether a small number of overflow-driven
   subjects drive the coupling.

4. **Per-release age tertile stratification** — within each release,
   split subjects into age tertiles and compute ρ within each. Tests
   whether the coupling concentrates in younger (more-motion) subjects
   or is broadly distributed.

5. **Two-half split-half stability** — randomly split each cohort 50/50
   (10 iterations), report mean ρ and SD across halves. Tests whether
   ρ is a stable cohort-level statistic or driven by a few subjects.

Output: outputs/2026-04-25-hbn-artifact-controls-R7-R11.md (report)
        outputs/2026-04-25-hbn-artifact-controls-R7-R11.csv (data)
"""

from pathlib import Path
import csv
import math
import random
import statistics

REPO = Path(__file__).resolve().parent.parent
HBN_DATA = Path("/Volumes/T9/hbn_data")
COMPOSITE_DIR = REPO / "outputs/schumann/images/psd_timelapse"
QUALITY_DIR = REPO / "outputs/schumann/images/quality"
OUTPUT_REPORT = REPO / "outputs/2026-04-25-hbn-artifact-controls-R7-R11.md"
OUTPUT_CSV = REPO / "outputs/2026-04-25-hbn-artifact-controls-R7-R11.csv"

TARGET_RELEASES = ["R7", "R11"]  # primary
COMPARATOR_RELEASES = ["R1", "R10"]  # also high-ρ vs lower-ρ adult-like
ALL_RELEASES = TARGET_RELEASES + COMPARATOR_RELEASES


def _rank(values):
    indexed = sorted(enumerate(values), key=lambda t: t[1])
    ranks = [0.0] * len(values)
    i = 0
    n = len(values)
    while i < n:
        j = i
        while j + 1 < n and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        i = j + 1
    return ranks


def _norm_cdf(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def spearman(x, y):
    n = len(x)
    if n < 3:
        return float("nan"), float("nan"), n
    xr = _rank(x)
    yr = _rank(y)
    return pearson(xr, yr)


def pearson(x, y):
    n = len(x)
    if n < 3:
        return float("nan"), float("nan"), n
    mx = sum(x) / n
    my = sum(y) / n
    sx = sum((xi - mx) ** 2 for xi in x)
    sy = sum((yi - my) ** 2 for yi in y)
    sxy = sum((x[i] - mx) * (y[i] - my) for i in range(n))
    denom = math.sqrt(sx * sy)
    if denom == 0:
        return float("nan"), float("nan"), n
    r = sxy / denom
    if abs(r) >= 1.0:
        return r, 0.0, n
    z = math.atanh(r)
    se = 1.0 / math.sqrt(n - 3)
    p = 2.0 * (1.0 - _norm_cdf(abs(z / se)))
    return r, p, n


def load_release(rel):
    """Load per-subject coupling, age, and per-event template_rho summary."""
    coupling_csv = COMPOSITE_DIR / f"hbn_{rel}_composite/beta_peak_iaf_coupling.csv"
    participants = HBN_DATA / f"cmi_bids_{rel}/participants.tsv"
    quality_csv = QUALITY_DIR / f"per_event_quality_hbn_{rel}_composite.csv"

    coupling = {}
    if coupling_csv.exists():
        with coupling_csv.open() as f:
            for row in csv.DictReader(f):
                try:
                    coupling[row["subject_id"]] = {
                        "iaf_hz": float(row["iaf_hz"]),
                        "sr_peak_ratio": float(row["sr_peak_ratio"]),
                        "beta_peak_ratio": float(row["beta_peak_ratio"]),
                        "n_events": int(row["n_events"]),
                    }
                except (ValueError, KeyError):
                    continue

    ages = {}
    if participants.exists():
        with participants.open() as f:
            for row in csv.DictReader(f, delimiter="\t"):
                try:
                    ages[row["participant_id"]] = float(row["age"])
                except (ValueError, KeyError):
                    continue

    # Per-event template_rho — aggregate to per-subject median
    median_rho = {}
    n_events_q = {}
    if quality_csv.exists():
        per_subject_rhos = {}
        with quality_csv.open() as f:
            for row in csv.DictReader(f):
                try:
                    sid = row["subject_id"]
                    rho = float(row["template_rho"])
                    per_subject_rhos.setdefault(sid, []).append(rho)
                except (ValueError, KeyError):
                    continue
        for sid, rhos in per_subject_rhos.items():
            median_rho[sid] = statistics.median(rhos)
            n_events_q[sid] = len(rhos)

    rows = []
    for sid, c in coupling.items():
        if sid not in ages:
            continue
        rows.append({
            "subject_id": sid,
            "release": rel,
            "age": ages[sid],
            "iaf_hz": c["iaf_hz"],
            "sr_peak_ratio": c["sr_peak_ratio"],
            "beta_peak_ratio": c["beta_peak_ratio"],
            "n_events": c["n_events"],
            "median_template_rho": median_rho.get(sid, float("nan")),
        })
    return rows


def baseline_rho(rows):
    sr = [r["sr_peak_ratio"] for r in rows]
    beta = [r["beta_peak_ratio"] for r in rows]
    return spearman(sr, beta)


def threshold_sensitivity(rows, label, thresholds=(3, 5, 7, 10)):
    """Recompute ρ at successive minimum-event-count cutoffs."""
    print(f"\n## n_events threshold sensitivity ({label})")
    print(f"{'min_events':>12} {'N':>5} {'ρ':>8} {'p':>10}")
    results = []
    for thr in thresholds:
        sub = [r for r in rows if r["n_events"] >= thr]
        if len(sub) < 5:
            continue
        rho, p, n = baseline_rho(sub)
        print(f"{thr:>12} {n:>5} {rho:>8.3f} {p:>10.2e}")
        results.append({"threshold": thr, "n": n, "rho": rho, "p": p})
    return results


def template_rho_quality_filter(rows, label, thresholds=(0.0, 0.1, 0.2, 0.3)):
    """Restrict to subjects with median per-event template_rho ≥ threshold."""
    print(f"\n## Median template_rho quality threshold ({label})")
    print(f"{'min_med_rho':>12} {'N':>5} {'ρ':>8} {'p':>10}")
    results = []
    for thr in thresholds:
        sub = [r for r in rows if not math.isnan(r["median_template_rho"]) and r["median_template_rho"] >= thr]
        if len(sub) < 5:
            continue
        rho, p, n = baseline_rho(sub)
        print(f"{thr:>12.2f} {n:>5} {rho:>8.3f} {p:>10.2e}")
        results.append({"threshold": thr, "n": n, "rho": rho, "p": p})
    return results


def outlier_winsorization(rows, label, percentiles=(99, 95, 90)):
    """Exclude subjects with extreme amplitude ratios above given percentile."""
    print(f"\n## Outlier exclusion (top-percentile cap; {label})")
    print(f"{'cap_pct':>10} {'N':>5} {'ρ':>8} {'p':>10}")
    results = []
    sr_all = [r["sr_peak_ratio"] for r in rows]
    beta_all = [r["beta_peak_ratio"] for r in rows]
    for pct in percentiles:
        sr_cap = _percentile(sr_all, pct)
        beta_cap = _percentile(beta_all, pct)
        sub = [r for r in rows if r["sr_peak_ratio"] <= sr_cap and r["beta_peak_ratio"] <= beta_cap]
        if len(sub) < 5:
            continue
        rho, p, n = baseline_rho(sub)
        print(f"{pct:>10}% {n:>5} {rho:>8.3f} {p:>10.2e}")
        results.append({"cap_percentile": pct, "n": n, "rho": rho, "p": p})
    return results


def _percentile(values, pct):
    s = sorted(values)
    if not s:
        return float("nan")
    k = (len(s) - 1) * pct / 100.0
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return s[int(k)]
    return s[f] + (s[c] - s[f]) * (k - f)


def age_tertile_stratification(rows, label):
    """Within-release age tertiles."""
    print(f"\n## Age tertile stratification ({label})")
    print(f"{'tertile':>10} {'age_range':>15} {'N':>5} {'ρ':>8} {'p':>10}")
    if len(rows) < 9:
        print("  (insufficient n)")
        return []
    ages = sorted(r["age"] for r in rows)
    n = len(ages)
    t1_cut = ages[n // 3]
    t2_cut = ages[2 * n // 3]
    tertiles = [
        ("T1 (younger)", lambda r: r["age"] <= t1_cut, ages[0], t1_cut),
        ("T2 (middle)", lambda r: t1_cut < r["age"] <= t2_cut, t1_cut, t2_cut),
        ("T3 (older)", lambda r: r["age"] > t2_cut, t2_cut, ages[-1]),
    ]
    results = []
    for tname, tfn, tlow, thigh in tertiles:
        sub = [r for r in rows if tfn(r)]
        if len(sub) < 5:
            continue
        rho, p, m = baseline_rho(sub)
        age_range = f"{tlow:.1f}-{thigh:.1f}"
        print(f"{tname:>10} {age_range:>15} {m:>5} {rho:>8.3f} {p:>10.2e}")
        results.append({"tertile": tname, "age_range": age_range, "n": m, "rho": rho, "p": p})
    return results


def split_half_stability(rows, label, n_iterations=200, seed=42):
    """Random 50/50 splits; report mean ρ and 95% interval."""
    if len(rows) < 20:
        return {}
    rng = random.Random(seed)
    rhos1 = []
    rhos2 = []
    for _ in range(n_iterations):
        shuffled = rows[:]
        rng.shuffle(shuffled)
        mid = len(shuffled) // 2
        h1 = shuffled[:mid]
        h2 = shuffled[mid:]
        rho1, _, _ = baseline_rho(h1)
        rho2, _, _ = baseline_rho(h2)
        if not math.isnan(rho1):
            rhos1.append(rho1)
        if not math.isnan(rho2):
            rhos2.append(rho2)
    rhos = rhos1 + rhos2
    if len(rhos) < 5:
        return {}
    mean = statistics.mean(rhos)
    sd = statistics.stdev(rhos) if len(rhos) > 1 else 0
    rhos_sorted = sorted(rhos)
    p2_5 = rhos_sorted[int(0.025 * len(rhos))]
    p97_5 = rhos_sorted[int(0.975 * len(rhos))]
    print(f"\n## Split-half stability ({label}; {n_iterations} iterations)")
    print(f"  Mean ρ across halves: {mean:.3f} (SD {sd:.3f})")
    print(f"  95% interval: [{p2_5:.3f}, {p97_5:.3f}]")
    return {
        "n_iterations": n_iterations,
        "mean_rho": mean,
        "sd_rho": sd,
        "ci_lo_2_5": p2_5,
        "ci_hi_97_5": p97_5,
    }


def main():
    all_results = {}
    for rel in ALL_RELEASES:
        print(f"\n{'='*60}\nRelease {rel}\n{'='*60}")
        rows = load_release(rel)
        if len(rows) < 20:
            print(f"  (insufficient data, skipping)")
            continue
        baseline_r, baseline_p, baseline_n = baseline_rho(rows)
        print(f"\nBaseline (all subjects, n={baseline_n}): ρ = {baseline_r:.3f}, p = {baseline_p:.2e}")

        n_events_dist = [r["n_events"] for r in rows]
        rho_dist = [r["median_template_rho"] for r in rows if not math.isnan(r["median_template_rho"])]
        print(f"  n_events: median {statistics.median(n_events_dist):.0f}, "
              f"max {max(n_events_dist)}, "
              f"% subjects with ≥7 events: {100 * sum(1 for n in n_events_dist if n >= 7) / len(n_events_dist):.1f}%")
        if rho_dist:
            print(f"  median template_rho per subject: median {statistics.median(rho_dist):.3f}, "
                  f"% > 0.3: {100 * sum(1 for r in rho_dist if r > 0.3) / len(rho_dist):.1f}%")

        results = {
            "baseline": {"n": baseline_n, "rho": baseline_r, "p": baseline_p},
            "n_events_thresholds": threshold_sensitivity(rows, rel),
            "template_rho_quality": template_rho_quality_filter(rows, rel),
            "outlier_winsorization": outlier_winsorization(rows, rel),
            "age_tertiles": age_tertile_stratification(rows, rel),
            "split_half": split_half_stability(rows, rel),
        }
        all_results[rel] = results

    # Save results
    print(f"\nSaving CSV to {OUTPUT_CSV}")
    with OUTPUT_CSV.open("w") as f:
        f.write("# HBN R7/R11 (and R1/R10 comparators) — artifact-control sensitivity analyses\n\n")
        for rel, results in all_results.items():
            f.write(f"\n# Release {rel}\n")
            f.write(f"baseline_n,{results['baseline']['n']}\n")
            f.write(f"baseline_rho,{results['baseline']['rho']:.4f}\n")
            f.write(f"baseline_p,{results['baseline']['p']:.4e}\n")
            f.write("\n## n_events threshold sensitivity\n")
            f.write("threshold,n,rho,p\n")
            for r in results["n_events_thresholds"]:
                f.write(f"{r['threshold']},{r['n']},{r['rho']:.4f},{r['p']:.4e}\n")
            f.write("\n## Template_rho quality threshold\n")
            f.write("threshold,n,rho,p\n")
            for r in results["template_rho_quality"]:
                f.write(f"{r['threshold']},{r['n']},{r['rho']:.4f},{r['p']:.4e}\n")
            f.write("\n## Outlier winsorization\n")
            f.write("cap_percentile,n,rho,p\n")
            for r in results["outlier_winsorization"]:
                f.write(f"{r['cap_percentile']},{r['n']},{r['rho']:.4f},{r['p']:.4e}\n")
            f.write("\n## Age tertile stratification\n")
            f.write("tertile,age_range,n,rho,p\n")
            for r in results["age_tertiles"]:
                f.write(f"{r['tertile']},{r['age_range']},{r['n']},{r['rho']:.4f},{r['p']:.4e}\n")
            sh = results["split_half"]
            if sh:
                f.write("\n## Split-half stability\n")
                f.write(f"n_iterations,{sh['n_iterations']}\n")
                f.write(f"mean_rho,{sh['mean_rho']:.4f}\n")
                f.write(f"sd_rho,{sh['sd_rho']:.4f}\n")
                f.write(f"ci_2_5,{sh['ci_lo_2_5']:.4f}\n")
                f.write(f"ci_97_5,{sh['ci_hi_97_5']:.4f}\n")
    return all_results


if __name__ == "__main__":
    main()
