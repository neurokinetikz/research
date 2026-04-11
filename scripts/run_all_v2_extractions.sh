#!/bin/bash
# Run all v2 extractions sequentially (f0=7.60, max_peaks=15, merged theta+alpha, R²)
# Usage: bash scripts/run_all_v2_extractions.sh 2>&1 | tee outputs/v2_extraction_log.txt

set -e
SCRIPT="scripts/run_f0_760_extraction.py"

echo "============================================================"
echo "  v3 Extraction: f0=7.60, max_peaks=15, merged theta+alpha, bw_floor=2*freq_res"
echo "  Started: $(date)"
echo "============================================================"

echo -e "\n>>> EEGMMIDB"
python $SCRIPT --dataset eegmmidb

echo -e "\n>>> LEMON EC"
python $SCRIPT --dataset lemon

echo -e "\n>>> LEMON EO"
python $SCRIPT --dataset lemon --condition EO

echo -e "\n>>> Dortmund EC-pre (ses-1)"
python $SCRIPT --dataset dortmund

echo -e "\n>>> Dortmund EO-pre (ses-1)"
python $SCRIPT --dataset dortmund --condition EO-pre

echo -e "\n>>> Dortmund EC-post (ses-1)"
python $SCRIPT --dataset dortmund --condition EC-post

echo -e "\n>>> Dortmund EO-post (ses-1)"
python $SCRIPT --dataset dortmund --condition EO-post

echo -e "\n>>> Dortmund EC-pre (ses-2)"
python $SCRIPT --dataset dortmund --session 2

echo -e "\n>>> Dortmund EC-post (ses-2)"
python $SCRIPT --dataset dortmund --condition EC-post --session 2

echo -e "\n>>> Dortmund EO-pre (ses-2)"
python $SCRIPT --dataset dortmund --condition EO-pre --session 2

echo -e "\n>>> Dortmund EO-post (ses-2)"
python $SCRIPT --dataset dortmund --condition EO-post --session 2

echo -e "\n>>> CHBMP"
python $SCRIPT --dataset chbmp

echo -e "\n>>> HBN R1-R6"
python $SCRIPT --dataset hbn --release all

echo -e "\n============================================================"
echo "  ALL EXTRACTIONS COMPLETE"
echo "  Finished: $(date)"
echo "============================================================"
