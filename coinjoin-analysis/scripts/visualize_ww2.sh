#!/usr/bin/env bash

# Run generation of aggregated plots for all coordinators
python3 -m cj_process.parse_dumplings --cjtype ww2 --action plot_coinjoins --env_vars "PLOT_REMIXES_AGGREGATE=True" --target-path $TMP_DIR/ | tee parse_dumplings.py.log

# Run generation of plots for single intervals 
python3 -m cj_process.parse_dumplings --cjtype ww2 --action plot_coinjoins --env_vars "PLOT_REMIXES_SINGLE_INTERVAL=True" --target-path $TMP_DIR/ | tee parse_dumplings.py.log

# Visualization of wallets predictions in time
python3 -m cj_process.parse_dumplings --cjtype ww2 --env_vars "ANALYSIS_WALLET_PREDICTION=True" --target-path $TMP_DIR/ | tee parse_dumplings.py.log

# Visualise flows between coordinators
python3 -m cj_process.visualize_coordinators $TMP_DIR

# Visualize notable intervals
python3 -m cj_process.parse_dumplings --cjtype ww2 --env_vars "PROCESS_NOTABLE_INTERVALS=True" --target-path $TMP_DIR/ | tee parse_dumplings.py.log

# Run generation of aggregated plots for all coordinators WITHOUT tx reordering - copy all existing files first, then run generation again
mkdir -p $TMP_DIR/Scanner/wasabi2_norelativereorder/
cp -a $TMP_DIR/Scanner/wasabi2/. $TMP_DIR/Scanner/wasabi2_norelativereorder/
python3 -m cj_process.parse_dumplings --cjtype ww2 --action plot_coinjoins --env_vars "PLOT_REMIXES_SINGLE_INTERVAL=True;MIX_IDS=['wasabi2_norelativereorder']" --target-path $TMP_DIR/ | tee parse_dumplings.py.log