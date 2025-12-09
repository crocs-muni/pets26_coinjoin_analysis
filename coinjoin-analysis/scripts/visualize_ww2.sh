#!/usr/bin/env bash

# Run generation of aggregated plots for all coordinators
python3 -m cj_process.parse_dumplings --cjtype ww2 --action plot_coinjoins --env_vars "PLOT_REMIXES_AGGREGATE=True" --target-path $TMP_DIR/ | tee parse_dumplings.py.log

# Run generation of plots for single intervals 
python3 -m cj_process.parse_dumplings --cjtype ww2 --action plot_coinjoins --env_vars "PLOT_REMIXES_SINGLE_INTERVAL=True" --target-path $TMP_DIR/ | tee parse_dumplings.py.log

# Visualization of wallets predictions in time
python3 -m cj_process.parse_dumplings --cjtype ww2 --env_vars "ANALYSIS_WALLET_PREDICTION=True" --target-path $TMP_DIR/ | tee parse_dumplings.py.log



