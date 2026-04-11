"""Regenerate supplemental tables with 7th degree nobles/inverses, replace in frontiers_revision.tex"""
import re

phi = 1.6180339887498948
f0 = 7.6

positions = [
    ("\\textbf{Boundary}", 0.000),
    ("7\\degree\\ Noble", round(phi**-7, 3)),
    ("6\\degree\\ Noble", round(phi**-6, 3)),
    ("5\\degree\\ Noble", round(phi**-5, 3)),
    ("4\\degree\\ Noble", round(phi**-4, 3)),
    ("3\\degree\\ Noble", round(phi**-3, 3)),
    ("2\\degree\\ Noble / 1\\degree\\ Inverse", round(phi**-2, 3)),
    ("\\textbf{Attractor}", 0.500),
    ("\\textit{1\\degree\\ Noble} / 2\\degree\\ Inverse", round(phi**-1, 3)),
    ("3\\degree\\ Inverse", round(1 - phi**-3, 3)),
    ("4\\degree\\ Inverse", round(1 - phi**-4, 3)),
    ("5\\degree\\ Inverse", round(1 - phi**-5, 3)),
    ("6\\degree\\ Inverse", round(1 - phi**-6, 3)),
    ("7\\degree\\ Inverse", round(1 - phi**-7, 3)),
]

bands = [
    ("Sub-delta", -6, -5),
    ("Sub-delta", -5, -4),
    ("Sub-delta", -4, -3),
    ("Delta", -3, -2),
    ("Delta", -2, -1),
    ("Theta", -1, 0),
    ("Alpha", 0, 1),
    ("Low $\\beta$", 1, 2),
    ("High $\\beta$", 2, 3),
    ("$\\gamma$", 3, 4),
]

tables = []
for i, (bname, nlo, nhi) in enumerate(bands, 1):
    flo = f0 * phi**nlo
    fhi = f0 * phi**nhi
    lines = []
    lines.append("\\begin{table}[H]")
    lines.append("\\centering")
    lines.append(f"\\caption*{{Table S{i}: {bname} Band ($n = {nlo}$ to ${nhi}$, {flo:.2f}--{fhi:.2f} Hz)}}")
    lines.append("\\begin{tabular}{@{}lllll@{}}")
    lines.append("\\toprule")
    lines.append("Position Type & Offset & $n$ Value & $\\varphi^n$ & Frequency (Hz) \\\\")
    lines.append("\\midrule")
    for label, offset in positions:
        n_val = nlo + offset
        phi_n = phi**n_val
        freq = f0 * phi_n
        lines.append(f"{label} & {offset:.3f} & ${n_val:.3f}$ & ${phi_n:.4f}$ & {freq:.2f} \\\\")
    # Upper boundary
    phi_n_upper = phi**nhi
    freq_upper = f0 * phi_n_upper
    lines.append(f"\\textbf{{Upper Boundary}} & 1.000 & ${nhi:.3f}$ & ${phi_n_upper:.4f}$ & {freq_upper:.2f} \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    lines.append("")
    tables.append("\n".join(lines))

supp_content = "\n".join(tables)

# Read the file
with open("papers/frontiers_revision.tex", "r") as f:
    content = f.read()

# Find and replace the supplemental tables section
intro_text = """nobles (1\\degree\\ through 5\\degree\\ at $n + \\phisym^{-k}$ for $k = 1 \\ldots 5$), and inverse nobles (1\\degree\\ through 5\\degree\\ at $n + 1 - \\phisym^{-k}$)"""
new_intro = """nobles (1\\degree\\ through 7\\degree\\ at $n + \\phisym^{-k}$ for $k = 1 \\ldots 7$), and inverse nobles (1\\degree\\ through 7\\degree\\ at $n + 1 - \\phisym^{-k}$)"""
content = content.replace(intro_text, new_intro)

# Replace everything between the intro paragraph end and \end{document}
marker = "reflecting the fundamental symmetry of the noble hierarchy about the attractor."
end_marker = "\\end{document}"
idx_start = content.index(marker) + len(marker)
idx_end = content.rindex(end_marker)
content = content[:idx_start] + "\n\n" + supp_content + "\n" + content[idx_end:]

with open("papers/frontiers_revision.tex", "w") as f:
    f.write(content)

print("Done - regenerated 10 supplemental tables with 7th degree nobles/inverses")
