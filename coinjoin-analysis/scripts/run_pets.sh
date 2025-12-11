#!/usr/bin/env bash

if [[ -z "${BASE_PATH}" ]]; then
  echo "BASE_PATH not set"
  exit 1
fi


echo "###############################################" >> $BASE_PATH/summary.log

export TMP_DIR=$BASE_PATH/dumplings_temp
#
# Extract Dumplings results
#
# Create new temporary directory
mkdir $TMP_DIR/
# Unzip processed dumplings files
#unzip $BASE_PATH/btc/dumplings.zip -d $TMP_DIR/
unzip $BASE_PATH/dumplings.zip -d $TMP_DIR/

unzip $BASE_PATH/missing_dumplings_txs.zip -d $BASE_PATH/missing_dumplings_txs


#
# Process Wasabi 2.0
#
$BASE_PATH/coinjoin-analysis/scripts/process_ww2.sh


#
# Process Whirlpool Ashigaru
#
$BASE_PATH/coinjoin-analysis/scripts/process_aw.sh


#
# Process Wasabi 1.0 
#
$BASE_PATH/coinjoin-analysis/scripts/process_ww1.sh

#
# Process Samourai Whirlpool 
#
$BASE_PATH/coinjoin-analysis/scripts/process_sw.sh


#
# Visualize processed coinjoins
#
$BASE_PATH/coinjoin-analysis/scripts/visualize_ww2.sh
$BASE_PATH/coinjoin-analysis/scripts/visualize_aw.sh
$BASE_PATH/coinjoin-analysis/scripts/visualize_ww1.sh
$BASE_PATH/coinjoin-analysis/scripts/visualize_sw.sh


#
# Copy images and data into results directory
#
$BASE_PATH/coinjoin-analysis/scripts/collect_results.sh
