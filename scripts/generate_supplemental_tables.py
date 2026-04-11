"""
Generate supplemental tables of complete phi^n frequency predictions.
All positions from 1st through 7th degree nobles and their inverses,
across n=-6 to n=4, f0=7.6 Hz.
"""

import csv

phi = 1.6180339887498948
f0 = 7.6

# Position definitions up to 7th degree noble + inverse
# Noble positions: phi^{-k} for k=1..7
# Inverse nobles: 1 - phi^{-k} for k=3..7 (1st and 2nd degree inverses = nobles themselves)
positions = []

# Boundary
positions.append(("Boundary", 0.000, "phi^0 = 1"))

# Nobles (1st through 7th degree)
for k in range(1, 8):
    offset = phi**(-k)
    label = f"{k} Noble"
    rep = f"phi^(-{k})"
    positions.append((label, round(offset, 6), rep))

# Attractor
positions.append(("Attractor", 0.500, "---"))

# Inverse nobles (1st through 7th degree)
# Inverse of degree k = 1 - phi^{-k}
for k in range(1, 8):
    offset = 1.0 - phi**(-k)
    label = f"{k} Inverse"
    rep = f"1 - phi^(-{k})"
    positions.append((label, round(offset, 6), rep))

# Sort by offset
positions.sort(key=lambda x: x[1])

# Remove duplicates (boundary at 0 and 1 are same; 1st inverse = 1-phi^-1 = 0.382 = 2nd noble area)
# Actually let's keep all and let the table show everything

# Band definitions
bands = []
for n in range(-6, 4):
    f_lo = f0 * phi**n
    f_hi = f0 * phi**(n+1)
    bands.append((n, n+1, f_lo, f_hi))

# Generate table rows
rows = []
for n_lo, n_hi, f_lo, f_hi in bands:
    for label, offset, rep in positions:
        n_val = n_lo + offset
        freq = f0 * phi**n_val
        if freq < 0.3 or freq > 60:
            continue
        rows.append({
            'band_n_lo': n_lo,
            'band_n_hi': n_hi,
            'band_f_lo': round(f_lo, 2),
            'band_f_hi': round(f_hi, 2),
            'position_type': label,
            'offset': offset,
            'representation': rep,
            'n': round(n_val, 6),
            'phi_n': round(phi**n_val, 6),
            'frequency_hz': round(freq, 3),
        })

# Write CSV
csv_path = '/Users/neurokinetikz/Code/research/outputs/supplemental_phi_lattice_positions.csv'
with open(csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=[
        'band_n_lo', 'band_n_hi', 'band_f_lo', 'band_f_hi',
        'position_type', 'offset', 'representation', 'n', 'phi_n', 'frequency_hz'
    ])
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote {len(rows)} positions to {csv_path}")

# Also generate LaTeX tables grouped by band
tex_path = '/Users/neurokinetikz/Code/research/outputs/supplemental_phi_lattice_tables.tex'
with open(tex_path, 'w') as f:
    f.write("% Supplemental Tables: Complete phi^n Frequency Predictions\n")
    f.write(f"% f_0 = {f0} Hz, phi = {phi:.10f}\n")
    f.write(f"% Positions: Boundary, 1st-7th degree Noble, Attractor, 1st-7th degree Inverse Noble\n\n")

    for n_lo, n_hi, f_lo, f_hi in bands:
        band_rows = [r for r in rows if r['band_n_lo'] == n_lo]
        if not band_rows:
            continue

        # Determine band name
        if f_hi < 2.9:
            bname = "Sub-delta"
        elif f_hi <= 4.7:
            bname = "Delta"
        elif f_hi <= 7.6:
            bname = "Theta"
        elif f_hi <= 12.3:
            bname = "Alpha"
        elif f_hi <= 19.9:
            bname = r"Low $\beta$"
        elif f_hi <= 32.2:
            bname = r"High $\beta$"
        elif f_hi <= 52.1:
            bname = r"$\gamma$"
        else:
            bname = r"High $\gamma$"

        f.write(r"\begin{table}[H]" + "\n")
        f.write(r"\centering" + "\n")
        f.write(r"\caption*{Table S" + f"{n_lo+7}" + f": {bname} Band ($n = {n_lo}$ to ${n_hi}$, "
                f"{f_lo:.2f}--{f_hi:.2f} Hz)" + "}\n")
        f.write(r"\begin{tabular}{@{}lllll@{}}" + "\n")
        f.write(r"\toprule" + "\n")
        f.write(r"Position Type & Offset & $n$ Value & $\varphi^n$ & Frequency (Hz) \\" + "\n")
        f.write(r"\midrule" + "\n")

        for r in band_rows:
            ptype = r['position_type']
            # Bold boundaries and attractors
            if ptype == "Boundary":
                ptype_fmt = r"\textbf{Boundary}"
            elif ptype == "Attractor":
                ptype_fmt = r"\textbf{Attractor}"
            elif ptype == "1 Noble":
                ptype_fmt = r"\textit{1\degree\ Noble}"
            else:
                ptype_fmt = ptype.replace(" Noble", r"\degree\ Noble").replace(" Inverse", r"\degree\ Inverse")

            f.write(f"{ptype_fmt} & {r['offset']:.3f} & ${r['n']:.3f}$ & "
                    f"${r['phi_n']:.4f}$ & {r['frequency_hz']:.2f} \\\\\n")

        f.write(r"\bottomrule" + "\n")
        f.write(r"\end{tabular}" + "\n")
        f.write(r"\end{table}" + "\n\n")

print(f"Wrote LaTeX tables to {tex_path}")

# Print summary to console
print(f"\n{'='*80}")
print(f"COMPLETE PHI LATTICE: f0 = {f0} Hz, n = -6 to +4")
print(f"{'='*80}")
print(f"{'Position':<18} {'Offset':>8} {'n':>8} {'phi^n':>10} {'Freq (Hz)':>10}")
print(f"{'-'*18} {'-'*8} {'-'*8} {'-'*10} {'-'*10}")

current_band = None
for r in rows:
    band_key = (r['band_n_lo'], r['band_n_hi'])
    if band_key != current_band:
        current_band = band_key
        print(f"\n--- Band n={r['band_n_lo']} to {r['band_n_hi']} "
              f"({r['band_f_lo']:.2f}-{r['band_f_hi']:.2f} Hz) ---")

    marker = ""
    if r['position_type'] == "Boundary":
        marker = " ***"
    elif r['position_type'] == "Attractor":
        marker = " **"
    elif r['position_type'] == "1 Noble":
        marker = " *"

    print(f"{r['position_type']:<18} {r['offset']:>8.3f} {r['n']:>8.3f} "
          f"{r['phi_n']:>10.4f} {r['frequency_hz']:>10.3f}{marker}")
