#!/bin/bash
# Pull ALL analysis output subdirs from GCS for cohorts with _DONE marker.
#
# Downloads per cohort, all these subdirs:
#   coupling, perionset, mechanism_battery, iei, psd_timelapse,
#   single_event, multistream, source, if_corrections, spectrum, quality
# Plus:
#   quality.csv  → outputs/schumann/images/quality/per_event_quality_<cohort>_composite.csv
#   _logs        → outputs/schumann/images/coupling/<cohort>_composite/_logs/
#
# Usage:
#   bash scripts/pull_analysis_results.sh                 # pull all DONE
#   bash scripts/pull_analysis_results.sh lemon_EO chbmp  # pull specific cohorts

set -e
BUCKET=gs://sie-composite-v2-extractions
ROOT=$(cd "$(dirname "$0")/.." && pwd)

SUBDIRS=(coupling perionset mechanism_battery iei psd_timelapse
         single_event multistream source if_corrections spectrum quality)

if [ $# -eq 0 ]; then
  # Find all cohorts with _DONE marker
  COHORTS=$(gcloud storage ls ${BUCKET}/analysis-results/ 2>/dev/null \
    | sed 's#.*/analysis-results/##; s#/$##' | grep -v '^$')
else
  COHORTS="$@"
fi

for cohort in $COHORTS; do
  done=$(gcloud storage ls ${BUCKET}/analysis-results/${cohort}/_DONE 2>/dev/null | wc -l | tr -d ' ')
  if [ "$done" != "1" ]; then
    echo "[skip] $cohort (no _DONE marker)"
    continue
  fi
  echo "[pull] $cohort"
  # All output subdirs
  for sd in "${SUBDIRS[@]}"; do
    local_dir=${ROOT}/outputs/schumann/images/${sd}/${cohort}_composite
    mkdir -p "$local_dir"
    gcloud storage cp -r "${BUCKET}/analysis-results/${cohort}/${sd}/${cohort}_composite/*" \
      "$local_dir/" 2>&1 | tail -1 || true
  done
  # Quality CSV (the per-event one used for stratification)
  local_qual=${ROOT}/outputs/schumann/images/quality/per_event_quality_${cohort}_composite.csv
  mkdir -p $(dirname "$local_qual")
  gcloud storage cp "${BUCKET}/analysis-results/${cohort}/quality.csv" "$local_qual" 2>&1 | tail -1 || true
  # Logs bundle next to coupling outputs
  local_logs=${ROOT}/outputs/schumann/images/coupling/${cohort}_composite/_logs
  mkdir -p "$local_logs"
  gcloud storage cp -r "${BUCKET}/analysis-results/${cohort}/_logs/*" \
    "$local_logs/" 2>&1 | tail -1 || true
done
echo "Done."
