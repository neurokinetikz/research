#!/bin/bash
# Run SIE window enrichment on all remaining datasets.
# Dispatches up to MAX_VMS concurrent VMs.
#
# Usage: bash scripts/gcp_run_all_window_enrichment.sh

MAX_VMS=8
COMPLETED=0
FAILED=0

# Remaining jobs (already done: EEGMMIDB local, LEMON EC, HBN R4, CHBMP)
JOBS=(
    "lemon EO 1"
    "hbn_r1 . 1"
    "hbn_r2 . 1"
    "hbn_r3 . 1"
    "hbn_r6 . 1"
    "tdbrain . 1"
    "tdbrain EO 1"
    "dortmund . 1"
    "dortmund EO-pre 1"
    "dortmund EC-post 1"
    "dortmund EO-post 1"
    "dortmund . 2"
    "dortmund EO-pre 2"
    "dortmund EC-post 2"
    "dortmund EO-post 2"
)
TOTAL=${#JOBS[@]}
NEXT=0

declare -a SLOT_PID
declare -a SLOT_JOB
for ((i=0; i<MAX_VMS; i++)); do
    SLOT_PID[$i]=0
    SLOT_JOB[$i]=""
done

run_job() {
    local idx=$1
    local spec="${JOBS[$idx]}"
    local dataset=$(echo $spec | awk '{print $1}')
    local condition=$(echo $spec | awk '{print $2}')
    local session=$(echo $spec | awk '{print $3}')
    [ "$condition" = "." ] && condition=""

    local label="${dataset}${condition:+/$condition}/ses-${session}"
    echo "[$(date +%H:%M)] LAUNCH [$((idx+1))/$TOTAL] $label"

    bash scripts/gcp_run_window_enrichment.sh "$dataset" "$condition" "$session" > /tmp/winenr_all_${idx}.log 2>&1
}

check_result() {
    local idx=$1
    local rc=$2
    local spec="${JOBS[$idx]}"
    local dataset=$(echo $spec | awk '{print $1}')
    local condition=$(echo $spec | awk '{print $2}')
    [ "$condition" = "." ] && condition=""
    local label="${dataset}${condition:+/$condition}"

    if [ $rc -eq 0 ]; then
        echo "[$(date +%H:%M)] DONE  [$((idx+1))/$TOTAL] $label"
        COMPLETED=$((COMPLETED + 1))
    else
        echo "[$(date +%H:%M)] FAIL  [$((idx+1))/$TOTAL] $label (exit $rc)"
        FAILED=$((FAILED + 1))
    fi
}

echo "============================================================"
echo "  Window Enrichment: $TOTAL jobs, max $MAX_VMS concurrent"
echo "  Started: $(date)"
echo "============================================================"

RUNNING=0
for ((slot=0; slot<MAX_VMS && NEXT<TOTAL; slot++)); do
    run_job $NEXT &
    SLOT_PID[$slot]=$!
    SLOT_JOB[$slot]=$NEXT
    NEXT=$((NEXT + 1))
    RUNNING=$((RUNNING + 1))
    sleep 2
done

echo "[$(date +%H:%M)] Launched initial batch of $RUNNING"

while [ $RUNNING -gt 0 ]; do
    sleep 15
    for ((slot=0; slot<MAX_VMS; slot++)); do
        pid=${SLOT_PID[$slot]}
        if [ "$pid" != "0" ] && ! kill -0 $pid 2>/dev/null; then
            wait $pid
            rc=$?
            check_result ${SLOT_JOB[$slot]} $rc
            RUNNING=$((RUNNING - 1))

            if [ $NEXT -lt $TOTAL ]; then
                run_job $NEXT &
                SLOT_PID[$slot]=$!
                SLOT_JOB[$slot]=$NEXT
                NEXT=$((NEXT + 1))
                RUNNING=$((RUNNING + 1))
            else
                SLOT_PID[$slot]=0
            fi
        fi
    done
done

echo ""
echo "============================================================"
echo "  ALL DONE: $COMPLETED completed, $FAILED failed"
echo "  Finished: $(date)"
echo "============================================================"
