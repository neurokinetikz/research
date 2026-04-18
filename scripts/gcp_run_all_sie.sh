#!/bin/bash
# Run SIE detection on EVERY EEG recording across ALL datasets and conditions.
# Launches up to MAX_VMS concurrent VMs, queuing the rest.
# Compatible with macOS bash 3.2 (no wait -n).
#
# Usage: bash scripts/gcp_run_all_sie.sh

MAX_VMS=8
COMPLETED=0
FAILED=0

# Exhaustive job queue: every dataset × condition × session
JOBS=(
    # EEGMMIDB: 110 subjects, motor/imagery tasks (all runs concatenated)
    "eegmmidb  . 1"

    # CHBMP: 282 subjects, EC segments extracted from protmap task
    "chbmp     . 1"

    # LEMON: 220 subjects, EC and EO
    "lemon     . 1"
    "lemon     EO 1"

    # Dortmund session 1: 608 subjects × 4 conditions
    "dortmund  . 1"
    "dortmund  EO-pre 1"
    "dortmund  EC-post 1"
    "dortmund  EO-post 1"

    # Dortmund session 2: 208 subjects × 4 conditions (longitudinal)
    "dortmund  . 2"
    "dortmund  EO-pre 2"
    "dortmund  EC-post 2"
    "dortmund  EO-post 2"

    # HBN: 929 subjects across 5 releases (ages 5-21)
    "hbn_r1    . 1"
    "hbn_r2    . 1"
    "hbn_r3    . 1"
    "hbn_r4    . 1"
    "hbn_r6    . 1"

    # TDBRAIN: 1226 subjects, EC and EO (clinical psychiatric, ages 5-88)
    "tdbrain   . 1"
    "tdbrain   EO 1"

    # Discovery cohort (Michael's self-recordings + 3 consumer-grade public datasets)
    "muse          . 1"
    "epoc_self     . 1"
    "insight_self  . 1"
    "physf         . 1"
    "mpeng         . 1"
    "vep           . 1"
)
TOTAL=${#JOBS[@]}
NEXT=0

# Track PIDs for each slot
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

    bash scripts/gcp_run.sh "$dataset" "$condition" "$session" sie > /tmp/sie_job_${idx}.log 2>&1
}

check_result() {
    local idx=$1
    local rc=$2
    local spec="${JOBS[$idx]}"
    local dataset=$(echo $spec | awk '{print $1}')
    local condition=$(echo $spec | awk '{print $2}')
    local session=$(echo $spec | awk '{print $3}')
    [ "$condition" = "." ] && condition=""
    local label="${dataset}${condition:+/$condition}/ses-${session}"

    if [ $rc -eq 0 ]; then
        local events=$(grep "Total events:" /tmp/sie_job_${idx}.log 2>/dev/null | tail -1 | grep -oE '[0-9,]+' | tail -1)
        echo "[$(date +%H:%M)] DONE  [$((idx+1))/$TOTAL] $label — ${events:-?} events"
        COMPLETED=$((COMPLETED + 1))
    else
        echo "[$(date +%H:%M)] FAIL  [$((idx+1))/$TOTAL] $label (exit $rc)"
        FAILED=$((FAILED + 1))
    fi
}

echo "============================================================"
echo "  SIE Detection: $TOTAL jobs, max $MAX_VMS concurrent"
echo "  Started: $(date)"
echo "============================================================"

# Launch initial batch
RUNNING=0
for ((slot=0; slot<MAX_VMS && NEXT<TOTAL; slot++)); do
    run_job $NEXT &
    SLOT_PID[$slot]=$!
    SLOT_JOB[$slot]=$NEXT
    NEXT=$((NEXT + 1))
    RUNNING=$((RUNNING + 1))
    sleep 2
done

echo "[$(date +%H:%M)] Launched initial batch of $RUNNING jobs"

# Poll for completions and launch replacements
while [ $RUNNING -gt 0 ]; do
    sleep 10
    for ((slot=0; slot<MAX_VMS; slot++)); do
        pid=${SLOT_PID[$slot]}
        if [ "$pid" != "0" ] && ! kill -0 $pid 2>/dev/null; then
            # Process finished
            wait $pid
            rc=$?
            check_result ${SLOT_JOB[$slot]} $rc
            RUNNING=$((RUNNING - 1))

            # Launch next if available
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
echo "  ALL DONE: $COMPLETED completed, $FAILED failed out of $TOTAL"
echo "  Finished: $(date)"
echo "============================================================"
