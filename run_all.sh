#!/usr/bin/env bash
# Regenerates everything in outputs/ from scratch.
set -e
mkdir -p outputs
echo "Running Big-M Simplex..."
python3 big_m_simplex.py   > outputs/big_m_simplex_output.txt
echo "Running Transportation (VAM + MODI)..."
python3 transportation.py  > outputs/transportation_output.txt
echo "Running verification suite..."
python3 verify.py          > outputs/verification_output.txt
echo "Done. See outputs/"
