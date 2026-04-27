#!/usr/bin/env python3
"""
Second-pass patcher: handles cohort_config variants that the main patcher
missed (different variable names, 2-tuple returns, etc.).

Uses regex to find the "if cohort.startswith('dortmund_')" line and its
return statement, then inserts chbmp/hbn/plain-dortmund branches that mirror
the existing return signature.
"""
from __future__ import annotations
import os
import re


SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))

IMPORT_OLD = ('from scripts.run_sie_extraction import (\n'
              '    load_lemon, load_dortmund, load_srm, load_tdbrain,\n'
              ')')
IMPORT_NEW = ('from scripts.run_sie_extraction import (\n'
              '    load_lemon, load_dortmund, load_srm, load_tdbrain,\n'
              '    load_chbmp, load_hbn_by_subject,\n'
              ')')

# Match the dortmund_ branch, capture the return signature, insert
# chbmp/hbn/dortmund branches before the raise.
DORTMUND_RE = re.compile(
    r"(    if cohort\.startswith\('dortmund_'\):\n"
    r"        return load_dortmund, \{\}, (?P<tail>[^\n]+)\n)"
    r"    raise ValueError\(f\"unsupported cohort \{cohort!r\}\"\)"
)


def patch_file(path):
    with open(path, encoding='utf-8') as f:
        text = f.read()

    changed = False

    if 'load_chbmp' not in text and IMPORT_OLD in text:
        text = text.replace(IMPORT_OLD, IMPORT_NEW)
        changed = True

    if "cohort == 'chbmp'" not in text:
        m = DORTMUND_RE.search(text)
        if m:
            tail = m.group('tail')
            replacement = (
                f"    if cohort.startswith('dortmund_'):\n"
                f"        return load_dortmund, {{}}, {tail}\n"
                f"    if cohort == 'dortmund':\n"
                f"        return load_dortmund, {{}}, {tail}\n"
                f"    if cohort == 'chbmp':\n"
                f"        return load_chbmp, {{}}, {tail}\n"
                f"    if cohort.startswith('hbn_'):\n"
                f"        return load_hbn_by_subject, "
                f"{{'release': cohort.split('_', 1)[1]}}, {tail}\n"
                f"    raise ValueError(f\"unsupported cohort {{cohort!r}}\")"
            )
            text = DORTMUND_RE.sub(replacement, text, count=1)
            changed = True

    # _init_worker dispatch (only if it exists and doesn't already have chbmp)
    INITW_OLD = (
        "    _LOADER = {\n"
        "        'load_lemon': load_lemon,\n"
        "        'load_tdbrain': load_tdbrain,\n"
        "        'load_srm': load_srm,\n"
        "        'load_dortmund': load_dortmund,\n"
        "    }[loader_name]"
    )
    INITW_NEW = (
        "    _LOADER = {\n"
        "        'load_lemon': load_lemon,\n"
        "        'load_tdbrain': load_tdbrain,\n"
        "        'load_srm': load_srm,\n"
        "        'load_dortmund': load_dortmund,\n"
        "        'load_chbmp': load_chbmp,\n"
        "        'load_hbn_by_subject': load_hbn_by_subject,\n"
        "    }[loader_name]"
    )
    if INITW_OLD in text:
        text = text.replace(INITW_OLD, INITW_NEW)
        changed = True

    if changed:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(text)
        return True
    return False


def main():
    skip = {
        'composite_analysis_manifest.py',
        'composite_cohort_runner.py',
        'gcp_analysis_orchestrator.py',
        'sie_compute_onset_from_composite.py',
        'sie_b47_composite_cohorts.py',
        'sie_b58_composite_canonicality.py',
        'sie_paper_figure3_composite.py',
        '_patch_composite_scripts_for_cohorts.py',
        '_patch_composite_scripts_remaining.py',
    }
    all_files = sorted(os.listdir(SCRIPTS_DIR))
    targets = [f for f in all_files
               if ('composite' in f) and f.endswith('.py') and f not in skip]
    print(f"Target scripts: {len(targets)}")
    patched = 0
    for f in targets:
        path = os.path.join(SCRIPTS_DIR, f)
        if patch_file(path):
            patched += 1
            print(f"  [patched] {f}")
    print(f"\nPatched {patched}/{len(targets)} files (2nd pass)")


if __name__ == '__main__':
    main()
