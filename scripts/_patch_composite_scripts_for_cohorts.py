#!/usr/bin/env python3
"""
One-shot patcher: adds support for chbmp, hbn_R1..R11, and plain 'dortmund'
cohorts to all sie_*_composite*.py scripts.

Handles two tuple patterns in cohort_config:
  - 4-tuple: (loader, kw, events, qual)   [newer scripts with quality CSV]
  - 3-tuple: (loader, kw, events)         [older scripts without quality CSV]

Idempotent: checks for existing patched state before applying edits.
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

# 4-tuple pattern (newer): ends with events, qual
COHORT_OLD_4 = (
    "    if cohort.startswith('dortmund_'):\n"
    "        return load_dortmund, {}, events, qual\n"
    "    raise ValueError(f\"unsupported cohort {cohort!r}\")"
)
COHORT_NEW_4 = (
    "    if cohort.startswith('dortmund_'):\n"
    "        return load_dortmund, {}, events, qual\n"
    "    if cohort == 'dortmund':\n"
    "        return load_dortmund, {}, events, qual\n"
    "    if cohort == 'chbmp':\n"
    "        return load_chbmp, {}, events, qual\n"
    "    if cohort.startswith('hbn_'):\n"
    "        return load_hbn_by_subject, "
    "{'release': cohort.split('_', 1)[1]}, events, qual\n"
    "    raise ValueError(f\"unsupported cohort {cohort!r}\")"
)

# 3-tuple pattern (older): ends with events only
COHORT_OLD_3 = (
    "    if cohort.startswith('dortmund_'):\n"
    "        return load_dortmund, {}, events\n"
    "    raise ValueError(f\"unsupported cohort {cohort!r}\")"
)
COHORT_NEW_3 = (
    "    if cohort.startswith('dortmund_'):\n"
    "        return load_dortmund, {}, events\n"
    "    if cohort == 'dortmund':\n"
    "        return load_dortmund, {}, events\n"
    "    if cohort == 'chbmp':\n"
    "        return load_chbmp, {}, events\n"
    "    if cohort.startswith('hbn_'):\n"
    "        return load_hbn_by_subject, "
    "{'release': cohort.split('_', 1)[1]}, events\n"
    "    raise ValueError(f\"unsupported cohort {cohort!r}\")"
)

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


def patch_file(path):
    with open(path) as f:
        text = f.read()
    changed = False

    if 'load_chbmp' not in text:
        if IMPORT_OLD in text:
            text = text.replace(IMPORT_OLD, IMPORT_NEW)
            changed = True
        else:
            print(f"  [import] pattern NOT FOUND in {os.path.basename(path)}")

    # Skip if chbmp branch already exists
    if "cohort == 'chbmp'" not in text:
        if COHORT_OLD_4 in text:
            text = text.replace(COHORT_OLD_4, COHORT_NEW_4)
            changed = True
        elif COHORT_OLD_3 in text:
            text = text.replace(COHORT_OLD_3, COHORT_NEW_3)
            changed = True
        else:
            print(f"  [cohort] pattern NOT FOUND in {os.path.basename(path)}")

    if INITW_OLD in text and 'load_chbmp' not in text.split('_LOADER = {')[-1]:
        text = text.replace(INITW_OLD, INITW_NEW)
        changed = True

    if changed:
        with open(path, 'w') as f:
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
    print(f"\nPatched {patched}/{len(targets)} files")


if __name__ == '__main__':
    main()
