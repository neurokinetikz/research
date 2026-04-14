#!/bin/bash
# Run all IRASA extractions with dynamic VM scheduling.
# Launches up to MAX_VMS concurrent VMs, queuing the rest.
# As each VM finishes, the next job launches automatically.
#
# Usage: bash scripts/gcp_run_all_irasa.sh

MAX_VMS=8
RUNNING=0
COMPLETED=0
FAILED=0

# Job queue: "dataset condition session" per line
# Ordered smallest SSD first to maximize concurrency
JOBS=(
    "eegmmidb  . 1"
    "chbmp     . 1"
    "lemon     . 1"
    "lemon     EO 1"
    "dortmund  . 1"
    "dortmund  EO-pre 1"
    "dortmund  EC-post 1"
    "dortmund  EO-post 1"
    "dortmund  . 2"
    "dortmund  EO-pre 2"
    "hbn_r1    . 1"
    "hbn_r2    . 1"
    "hbn_r6    . 1"
    "hbn_r3    . 1"
    "hbn_r4    . 1"
)
TOTAL=${#JOBS[@]}
NEXT=0

launch_job() {
    local spec="${JOBS[$NEXT]}"
    local dataset=$(echo $spec | awk '{print $1}')
    local condition=$(echo $spec | awk '{print $2}')
    local session=$(echo $spec | awk '{print $3}')

    # Map "." to empty string for condition
    [ "$condition" = "." ] && condition=""

    local label="${dataset}${condition:+/$condition}/ses-${session}"
    echo "[$(date +%H:%M)] LAUNCH [$((NEXT+1))/$TOTAL] $label"

    bash scripts/gcp_run.sh "$dataset" "$condition" "$session" irasa > /tmp/irasa_job_${NEXT}.log 2>&1
    local rc=$?

    if [ $rc -eq 0 ]; then
        # Extract summary from log
        local peaks=$(grep "Total peaks:" /tmp/irasa_job_${NEXT}.log | tail -1 | grep -oP '[\d,]+' | tail -1)
        local time=$(grep "DONE:" /tmp/irasa_job_${NEXT}.log | grep -oP '[\d.]+(?= min)' | tail -1)
        echo "[$(date +%H:%M)] DONE  [$((NEXT+1))/$TOTAL] $label — ${peaks:-?} peaks in ${time:-?} min"
        COMPLETED=$((COMPLETED + 1))
    else
        echo "[$(date +%H:%M)] FAIL  [$((NEXT+1))/$TOTAL] $label (exit $rc)"
        FAILED=$((FAILED + 1))
    fi

    NEXT_DONE=$NEXT
}

echo "============================================================"
echo "  IRASA Full Re-extraction: $TOTAL jobs, max $MAX_VMS concurrent"
echo "  Started: $(date)"
echo "============================================================"

# Launch initial batch
while [ $NEXT -lt $TOTAL ] && [ $RUNNING -lt $MAX_VMS ]; do
    launch_job &
    PIDS[$NEXT]=$!
    RUNNING=$((RUNNING + 1))
    NEXT=$((NEXT + 1))
    sleep 2  # stagger VM creation slightly
done

echo "[$(date +%H:%M)] Launched initial batch of $RUNNING jobs"

# Wait for completions and launch replacements
while [ $RUNNING -gt 0 ]; do
    wait -n 2>/dev/null
    RUNNING=$((RUNNING - 1))

    # Launch next job if queue has more
    if [ $NEXT -lt $TOTAL ]; then
        launch_job &
        PIDS[$NEXT]=$!
        RUNNING=$((RUNNING + 1))
        NEXT=$((NEXT + 1))
    fi
done

echo ""
echo "============================================================"
echo "  ALL DONE: $COMPLETED completed, $FAILED failed out of $TOTAL"
echo "  Finished: $(date)"
echo "============================================================"
