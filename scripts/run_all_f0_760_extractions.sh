#!/bin/bash
# Run all f0=7.60 extractions sequentially
# Usage: bash scripts/run_all_f0_760_extractions.sh 2>&1 | tee outputs/f0_760_extraction_log.txt

set -e
SCRIPT="scripts/run_f0_760_extraction.py"

echo "============================================================"
echo "  f0=7.60 Re-extraction: All Datasets"
echo "  Started: $(date)"
echo "============================================================"

# EEGMMIDB (already done, will skip existing)
echo -e "\n>>> EEGMMIDB"
python $SCRIPT --dataset eegmmidb

# LEMON EC (may already be running/done)
echo -e "\n>>> LEMON EC"
python $SCRIPT --dataset lemon

# LEMON EO
echo -e "\n>>> LEMON EO"
python $SCRIPT --dataset lemon --condition EO

# Dortmund ses-1: EC-pre (default), EO-pre, EC-post, EO-post
echo -e "\n>>> Dortmund EC-pre (ses-1)"
python $SCRIPT --dataset dortmund

echo -e "\n>>> Dortmund EO-pre (ses-1)"
python $SCRIPT --dataset dortmund --condition EO-pre

echo -e "\n>>> Dortmund EC-post (ses-1)"
python $SCRIPT --dataset dortmund --condition EC-post

echo -e "\n>>> Dortmund EO-post (ses-1)"
python $SCRIPT --dataset dortmund --condition EO-post

# Dortmund ses-2: EC-pre, EC-post, EO-pre, EO-post
echo -e "\n>>> Dortmund EC-pre (ses-2)"
python $SCRIPT --dataset dortmund --session 2

echo -e "\n>>> Dortmund EC-post (ses-2)"
python $SCRIPT --dataset dortmund --condition EC-post --session 2

echo -e "\n>>> Dortmund EO-pre (ses-2)"
python $SCRIPT --dataset dortmund --condition EO-pre --session 2

echo -e "\n>>> Dortmund EO-post (ses-2)"
python $SCRIPT --dataset dortmund --condition EO-post --session 2

# CHBMP
echo -e "\n>>> CHBMP"
python $SCRIPT --dataset chbmp

# HBN (all releases)
echo -e "\n>>> HBN R1-R6"
python $SCRIPT --dataset hbn --release all

echo -e "\n============================================================"
echo "  ALL EXTRACTIONS COMPLETE"
echo "  Finished: $(date)"
echo "============================================================"
