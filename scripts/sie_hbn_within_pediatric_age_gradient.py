"""
Within-pediatric age gradient analysis: SR × β coupling ρ vs age within HBN.

Tests the developmental de-coupling hypothesis prediction (paper-staging Q3):
the cross-subject SR × β amplitude coupling ρ that distinguishes pediatric
HBN releases (ρ +0.57 to +0.825) from adult cohorts (ρ +0.16 to +0.45)
should decline monotonically with age within the pediatric range if the
underlying mechanism is developmental de-coupling.

Approach:
1. Load per-subject β-IAF coupling CSVs for HBN R1-R11 under composite-v2.
2. Load per-subject HBN demographics (participants.tsv) for age.
3. Per-release: replicate the cross-cohort meta's between-subject ρ as a
   sanity check.
4. Within-pediatric age binning (pooled across releases):
   - Bin subjects into age strata (5-7, 8-10, 11-13, 14-17, 18-21).
   - Compute Pearson r between sr_peak_ratio and beta_peak_ratio within
     each age stratum.
   - Test for monotonic decline of r with age via Spearman correlation
     between bin midpoint and bin r.
5. Per-release age regression alternative: within each release with
   sufficient age range, fit OLS:
       beta_peak_ratio ~ sr_peak_ratio * age
   and test the interaction term (negative interaction = de-coupling).

Output: report to outputs/2026-04-25-hbn-within-pediatric-age-gradient.md
"""

from pathlib import Path
import csv
import math
import os
import statistics

SCOPE = os.environ.get("SCOPE", "all")  # 'all', 'q4', 'sw'
SUFFIX = {"all": "", "q4": "_q4", "sw": "_sw"}[SCOPE]
_TAG = "" if SCOPE == "all" else f"_{SCOPE}"

# --- Paths ---
REPO = Path(__file__).resolve().parent.parent
HBN_DATA = Path("/Volumes/T9/hbn_data")
COMPOSITE_DIR = REPO / "outputs/schumann/images/psd_timelapse"
OUTPUT_REPORT = REPO / f"outputs/2026-04-25-hbn-within-pediatric-age-gradient{_TAG}.md"
OUTPUT_CSV = REPO / f"outputs/2026-04-25-hbn-within-pediatric-age-gradient{_TAG}.csv"

RELEASES = [f"R{i}" for i in range(1, 12)]
AGE_BINS = [(5, 7), (8, 10), (11, 13), (14, 17), (18, 21)]


def load_release_data(rel):
    """Load per-subject SR/β coupling and age for one HBN release."""
    coupling_csv = COMPOSITE_DIR / f"hbn_{rel}_composite/beta_peak_iaf_coupling{SUFFIX}.csv"
    participants = HBN_DATA / f"cmi_bids_{rel}/participants.tsv"

    if not coupling_csv.exists():
        return []
    if not participants.exists():
        return []

    # Load coupling per-subject
    coupling = {}
    with coupling_csv.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = row["subject_id"]
            try:
                coupling[sid] = {
                    "iaf_hz": float(row["iaf_hz"]),
                    "sr_peak_hz": float(row["sr_peak_hz"]),
                    "sr_peak_ratio": float(row["sr_peak_ratio"]),
                    "beta_peak_hz": float(row["beta_peak_hz"]),
                    "beta_peak_ratio": float(row["beta_peak_ratio"]),
                    "beta_over_iaf": float(row["beta_over_iaf"]),
                    "n_events": int(row["n_events"]),
                }
            except (ValueError, KeyError):
                continue

    # Load ages
    ages = {}
    sexes = {}
    with participants.open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            sid = row["participant_id"]
            try:
                ages[sid] = float(row["age"])
                sexes[sid] = row.get("sex", "")
            except (ValueError, KeyError):
                continue

    # Join on subject_id
    rows = []
    for sid, c in coupling.items():
        if sid in ages:
            rows.append({
                "subject_id": sid,
                "release": rel,
                "age": ages[sid],
                "sex": sexes.get(sid, ""),
                **c,
            })
    return rows


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
    # Fisher z-transformation for p-value
    if abs(r) >= 1.0:
        return r, 0.0, n
    z = math.atanh(r)
    se = 1.0 / math.sqrt(n - 3)
    p = 2.0 * (1.0 - _norm_cdf(abs(z / se)))
    return r, p, n


def spearman(x, y):
    """Simple Spearman correlation: rank then Pearson."""
    if len(x) < 3:
        return float("nan"), float("nan"), len(x)
    xr = _rank(x)
    yr = _rank(y)
    return pearson(xr, yr)


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
    """Standard normal CDF via erf approximation."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def fisher_z_diff_test(r1, n1, r2, n2):
    """Fisher z difference test between two correlations."""
    if not (-1 < r1 < 1 and -1 < r2 < 1):
        return float("nan"), float("nan")
    z1 = math.atanh(r1)
    z2 = math.atanh(r2)
    se = math.sqrt(1.0 / (n1 - 3) + 1.0 / (n2 - 3))
    z = (z1 - z2) / se
    p = 2.0 * (1.0 - _norm_cdf(abs(z)))
    return z, p


def ols_with_interaction(sr, beta, age):
    """OLS: beta ~ a + b*sr + c*age + d*sr*age, return d, p_d."""
    n = len(sr)
    if n < 10:
        return float("nan"), float("nan"), n
    # Center predictors
    sr_m = sum(sr) / n
    age_m = sum(age) / n
    x1 = [s - sr_m for s in sr]
    x2 = [a - age_m for a in age]
    x3 = [x1[i] * x2[i] for i in range(n)]
    y = beta
    y_m = sum(y) / n
    yc = [yi - y_m for yi in y]

    # Build normal equations
    X = [[1.0, x1[i], x2[i], x3[i]] for i in range(n)]
    XtX = [[sum(X[k][i] * X[k][j] for k in range(n)) for j in range(4)] for i in range(4)]
    Xty = [sum(X[k][i] * y[k] for k in range(n)) for i in range(4)]

    # Solve by Gauss elimination
    M = [row[:] + [Xty[i]] for i, row in enumerate(XtX)]
    for i in range(4):
        pivot = M[i][i]
        if abs(pivot) < 1e-12:
            return float("nan"), float("nan"), n
        for j in range(i + 1, 4):
            if abs(M[j][i]) < 1e-12:
                continue
            f = M[j][i] / pivot
            for k in range(i, 5):
                M[j][k] -= f * M[i][k]
    coef = [0.0] * 4
    for i in range(3, -1, -1):
        s = M[i][4]
        for j in range(i + 1, 4):
            s -= M[i][j] * coef[j]
        coef[i] = s / M[i][i]

    # SE for interaction term
    yhat = [coef[0] + coef[1] * x1[i] + coef[2] * x2[i] + coef[3] * x3[i] for i in range(n)]
    resid = [y[i] - yhat[i] for i in range(n)]
    rss = sum(r * r for r in resid)
    sigma2 = rss / (n - 4)

    # Diagonal of (XtX)^-1
    inv = _matrix_inverse(XtX)
    if inv is None:
        return coef[3], float("nan"), n
    se_d = math.sqrt(sigma2 * inv[3][3])
    if se_d == 0:
        return coef[3], float("nan"), n
    t = coef[3] / se_d
    df = n - 4
    p = 2.0 * (1.0 - _t_cdf(abs(t), df))
    return coef[3], p, n


def _matrix_inverse(M):
    n = len(M)
    A = [row[:] + [1.0 if i == j else 0.0 for j in range(n)] for i, row in enumerate(M)]
    for i in range(n):
        pivot = A[i][i]
        if abs(pivot) < 1e-12:
            return None
        for k in range(2 * n):
            A[i][k] /= pivot
        for j in range(n):
            if j == i:
                continue
            f = A[j][i]
            for k in range(2 * n):
                A[j][k] -= f * A[i][k]
    return [row[n:] for row in A]


def _t_cdf(t, df):
    """Approximation of t-distribution CDF via normal for large df."""
    if df > 30:
        return _norm_cdf(t)
    # Simple beta-function-based approximation for moderate df
    x = df / (df + t * t)
    # Use regularized incomplete beta — approximate
    return 1.0 - 0.5 * _ibeta(0.5 * df, 0.5, x)


def _ibeta(a, b, x):
    """Approximation of regularized incomplete beta via series; rough but adequate for p-value reporting."""
    # Continued-fraction expansion (Lentz)
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    # Symmetry
    if x > (a + 1) / (a + b + 2):
        return 1.0 - _ibeta(b, a, 1 - x)
    # Series
    bt = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1 - x))
    # Lentz CF
    fpmin = 1e-30
    qab = a + b
    qap = a + 1
    qam = a - 1
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, 200):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        del_ = d * c
        h *= del_
        if abs(del_ - 1.0) < 3e-7:
            break
    return bt * h / a


# --- Main ---
def main():
    print("Loading per-subject coupling + age data per release...")
    all_rows = []
    per_release_n = {}
    for rel in RELEASES:
        rows = load_release_data(rel)
        per_release_n[rel] = len(rows)
        all_rows.extend(rows)
        print(f"  {rel}: {len(rows)} subjects with both coupling and age")

    total_n = len(all_rows)
    print(f"Total: {total_n} subjects across {sum(1 for n in per_release_n.values() if n > 0)} releases")

    # --- 1. Per-release between-subject ρ (sanity check vs cross-cohort meta numbers) ---
    # The amplitude ratios have severe outliers from near-zero-baseline overflow
    # (some values >10^6). Cross-cohort meta uses Spearman (rank-based, robust).
    # We report both Pearson and Spearman for transparency.
    print("\n## 1. Per-release between-subject ρ (replication check; Spearman primary)")
    print(f"{'Release':<8} {'N':>5} {'ρ Spear':>10} {'p_S':>10} {'r Pear':>10} {'p_P':>10} {'age_mean':>10}")
    per_release_rho = []
    for rel in RELEASES:
        rows = [r for r in all_rows if r["release"] == rel]
        if len(rows) < 5:
            continue
        sr = [r["sr_peak_ratio"] for r in rows]
        beta = [r["beta_peak_ratio"] for r in rows]
        ages = [r["age"] for r in rows]
        rho_s, p_s, n = spearman(sr, beta)
        r_p, p_p, _ = pearson(sr, beta)
        age_m = sum(ages) / len(ages)
        age_sd = statistics.stdev(ages) if len(ages) > 1 else 0
        print(f"{rel:<8} {n:>5} {rho_s:>10.3f} {p_s:>10.2e} {r_p:>10.3f} {p_p:>10.2e} {age_m:>10.2f}")
        per_release_rho.append({
            "release": rel,
            "n": n,
            "rho_spearman": rho_s,
            "p_spearman": p_s,
            "r_pearson": r_p,
            "p_pearson": p_p,
            "age_mean": age_m,
            "age_sd": age_sd,
        })

    # --- 2. Pooled-across-releases age-binned ρ (the central test) ---
    print("\n## 2. Pooled age-binned Spearman ρ across all HBN releases")
    print(f"{'Age bin':<10} {'N':>5} {'ρ Spear':>10} {'p_S':>10} {'r Pear':>10} {'age_mean':>10}")
    age_bin_results = []
    for low, high in AGE_BINS:
        rows = [r for r in all_rows if low <= r["age"] <= high]
        if len(rows) < 5:
            print(f"{low}-{high:<5} {len(rows):>5} (insufficient)")
            continue
        sr = [r["sr_peak_ratio"] for r in rows]
        beta = [r["beta_peak_ratio"] for r in rows]
        ages = [r["age"] for r in rows]
        rho_s, p_s, n = spearman(sr, beta)
        r_p, p_p, _ = pearson(sr, beta)
        age_m = sum(ages) / n
        print(f"{low}-{high:<5} {n:>5} {rho_s:>10.3f} {p_s:>10.2e} {r_p:>10.3f} {age_m:>10.2f}")
        age_bin_results.append({
            "age_bin": f"{low}-{high}",
            "age_mean": age_m,
            "n": n,
            "rho_spearman": rho_s,
            "p_spearman": p_s,
            "r_pearson": r_p,
        })

    # --- 3. Monotonic decline test (Spearman of bin ρ vs bin midpoint) ---
    if len(age_bin_results) >= 3:
        midpoints = [b["age_mean"] for b in age_bin_results]
        rs = [b["rho_spearman"] for b in age_bin_results]
        rho_spear, p_spear, _ = spearman(midpoints, rs)
        print("\n## 3. Monotonic decline test (Spearman of bin ρ vs bin midpoint)")
        print(f"  Spearman ρ = {rho_spear:.3f}, p = {p_spear:.3f}")
        print("  Negative ρ indicates declining SR-β coupling with age")

    # --- 4. Per-release OLS interaction test on LOG-amplitudes ---
    # Raw ratios have severe right-tail outliers; log-transform makes the
    # SR-β coupling tractable for OLS interaction testing.
    print("\n## 4. Per-release OLS interaction test (log β ~ log SR + age + log SR × age)")
    print(f"{'Release':<8} {'N':>5} {'d (logSR×age)':>16} {'p':>10}")
    interaction_results = []
    for rel in RELEASES:
        rows = [r for r in all_rows if r["release"] == rel]
        rows = [r for r in rows if r["sr_peak_ratio"] > 0 and r["beta_peak_ratio"] > 0]
        if len(rows) < 15:
            continue
        log_sr = [math.log(r["sr_peak_ratio"]) for r in rows]
        log_beta = [math.log(r["beta_peak_ratio"]) for r in rows]
        ages = [r["age"] for r in rows]
        d, p, n = ols_with_interaction(log_sr, log_beta, ages)
        d_str = f"{d:>16.4f}" if not math.isnan(d) else "nan".rjust(16)
        p_str = f"{p:>10.2e}" if not math.isnan(p) else "nan".rjust(10)
        print(f"{rel:<8} {n:>5} {d_str} {p_str}")
        interaction_results.append({
            "release": rel,
            "n": n,
            "interaction_d": d,
            "p": p,
        })

    # --- 5. Pooled OLS interaction test (across all HBN, log-amplitudes) ---
    rows_pool = [r for r in all_rows if r["sr_peak_ratio"] > 0 and r["beta_peak_ratio"] > 0]
    if len(rows_pool) > 30:
        log_sr = [math.log(r["sr_peak_ratio"]) for r in rows_pool]
        log_beta = [math.log(r["beta_peak_ratio"]) for r in rows_pool]
        ages = [r["age"] for r in rows_pool]
        d_pool, p_pool, n_pool = ols_with_interaction(log_sr, log_beta, ages)
        d_str = f"{d_pool:.4f}" if not math.isnan(d_pool) else "nan"
        p_str = f"{p_pool:.2e}" if not math.isnan(p_pool) else "nan"
        print("\n## 5. Pooled across all HBN releases (log-amplitudes)")
        print(f"  N = {n_pool}, interaction d = {d_str}, p = {p_str}")
        print("  Negative d = developmental de-coupling (log-β response to log-SR weakens with age)")

    # --- Save results CSV ---
    print(f"\nSaving results to {OUTPUT_CSV}")
    with OUTPUT_CSV.open("w") as f:
        f.write("# Within-pediatric age gradient analysis\n")
        f.write("# Per-release between-subject ρ (Spearman primary; Pearson reference)\n")
        f.write("release,n,rho_spearman,p_spearman,r_pearson,p_pearson,age_mean,age_sd\n")
        for row in per_release_rho:
            f.write(
                f"{row['release']},{row['n']},"
                f"{row['rho_spearman']:.4f},{row['p_spearman']:.4e},"
                f"{row['r_pearson']:.4f},{row['p_pearson']:.4e},"
                f"{row['age_mean']:.2f},{row['age_sd']:.2f}\n"
            )
        f.write("\n# Pooled age-binned Spearman ρ\n")
        f.write("age_bin,age_mean,n,rho_spearman,p_spearman,r_pearson\n")
        for row in age_bin_results:
            f.write(
                f"{row['age_bin']},{row['age_mean']:.2f},{row['n']},"
                f"{row['rho_spearman']:.4f},{row['p_spearman']:.4e},"
                f"{row['r_pearson']:.4f}\n"
            )
        f.write("\n# Per-release OLS log-SR × age interaction (log β ~ log SR + age + log SR × age)\n")
        f.write("release,n,interaction_d,p\n")
        for row in interaction_results:
            d_str = f"{row['interaction_d']:.4f}" if not math.isnan(row['interaction_d']) else "nan"
            p_str = f"{row['p']:.4e}" if not math.isnan(row['p']) else "nan"
            f.write(f"{row['release']},{row['n']},{d_str},{p_str}\n")

    return {
        "per_release_rho": per_release_rho,
        "age_bin_results": age_bin_results,
        "interaction_results": interaction_results,
    }


if __name__ == "__main__":
    main()
