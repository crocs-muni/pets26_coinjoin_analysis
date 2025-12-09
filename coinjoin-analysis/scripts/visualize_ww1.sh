#!/usr/bin/env bash

# Run generation of aggregated plots 
python3 -m cj_process.parse_dumplings --cjtype ww1 --action plot_coinjoins --env_vars "PLOT_REMIXES_AGGREGATE=True" --target-path $TMP_DIR/ | tee parse_dumplings.py.log

# Run generation of plots only for specific intervals
python3 -m cj_process.parse_dumplings --cjtype ww1 --action plot_coinjoins --target-path $TMP_DIR/ --env_vars "PLOT_REMIXES_SINGLE_INTERVAL=True" | tee parse_dumplings.py.log

# Visualization of wallets predictions in time
python3 -m cj_process.parse_dumplings --cjtype ww1 --env_vars "ANALYSIS_WALLET_PREDICTION=True" --target-path $TMP_DIR/ | tee parse_dumplings.py.log
