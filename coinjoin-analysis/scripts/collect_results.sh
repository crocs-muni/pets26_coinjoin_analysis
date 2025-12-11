#!/bin/bash

RESULTS_DIR=$TMP_DIR/Scanner
ARTIF_DIR=$BASE_PATH/results
echo "Collecting data from $RESULTS_DIR"
echo "Storing results to $ARTIF_DIR"

mkdir $ARTIF_DIR
for i in {1..22}; do
    mkdir -p "$ARTIF_DIR/fig$i"
done

mkdir -p "$ARTIF_DIR/table1"
mkdir -p "$ARTIF_DIR/table2"
mkdir -p "$ARTIF_DIR/table3_4"

# Data collection

echo "Manually drawn picture" > $ARTIF_DIR/fig1/info.txt

echo "Manually drawn picture" > $ARTIF_DIR/fig2/info.txt

#fig3-5 - non-public dataset ()ww2_as25_38_privacy_progress.png, ww2_as25_38_anonscoregain.png, ww2_realexp_num_inputs.png, ww2_realexp_num_outputs.png, ww2_realexp_as25_inoutheatmap.png, ww2_realexp_as38_inoutheatmap.png (generated from non-public dataset not shared due to privacy reasons as explained 
echo "Generated from non-public dataset not made public due to privacy reasons as explained in Appendix A." > $ARTIF_DIR/fig3/info.txt
echo "Generated from non-public dataset not made public due to privacy reasons as explained in Appendix A." > $ARTIF_DIR/fig4/info.txt
echo "Generated from non-public dataset not made public due to privacy reasons as explained in Appendix A." > $ARTIF_DIR/fig5/info.txt

# Copy missing_dumplings_txs.zip, run type=ww2 DOWNLOAD_MISSING_TRANSACTIONS=True
cp "$RESULTS_DIR/crawl_datasets.png" $ARTIF_DIR/fig6

cp "$RESULTS_DIR/wasabi2/2023-03-01 00-00-00--2023-04-01 00-00-00_unknown-static-100-1utxo/wasabi2_input_types_nums_notnorm.png" $ARTIF_DIR/fig7
echo "For WW2, the picture without sorting is obtained by running with SORT_COINJOINS_BY_RELATIVE_ORDER=False => 'python3 -m cj_process.parse_dumplings --cjtype ww2 --action plot_coinjoins --env_vars "PLOT_REMIXES_SINGLE_INTERVAL=True;SORT_COINJOINS_BY_RELATIVE_ORDER=False" --target-path $TMP_DIR/ | tee parse_dumplings.py.log
' " > $ARTIF_DIR/fig7/info.txt


cp "$RESULTS_DIR/wasabi1_zksnacks/2019-12-01 00-00-00--2020-01-01 00-00-00_unknown-static-100-1utxo/wasabi1_zksnacks_input_types_values_notnorm.png" $ARTIF_DIR/fig8
cp "$RESULTS_DIR/wasabi1_zksnacks/2020-10-01 00-00-00--2020-11-01 00-00-00_unknown-static-100-1utxo/wasabi1_zksnacks_input_types_nums_norm.png" $ARTIF_DIR/fig8

cp "$RESULTS_DIR/wasabi1_zksnacks/wasabi1_zksnacks_cummul_values_norm.png" $ARTIF_DIR/fig9
cp "$RESULTS_DIR/whirlpool/whirlpool_cummul_nums_norm_nolegend.png" $ARTIF_DIR/fig9
cp "$RESULTS_DIR/wasabi2/wasabi2_cummul_values_norm_nolegend.png" $ARTIF_DIR/fig9

cp "$RESULTS_DIR/wasabi2_zksnacks/wasabi2_zksnacks_wallets_predictions_dynamics.png" $ARTIF_DIR/fig10
cp "$RESULTS_DIR/wasabi2_kruw/wasabi2_kruw_wallets_predictions_dynamics.png" $ARTIF_DIR/fig10

cp "$RESULTS_DIR/wasabi2_opencoordinator/2025-03-01 00-00-00--2025-04-01 00-00-00_unknown-static-100-1utxo/wasabi2_opencoordinator_input_types_values_notnorm.png" $ARTIF_DIR/fig11

#fig12 - FIX copy and run new source vizualize coordinators
cp $TPM_DIR/coordinator_flows_*.html $ARTIF_DIR/fig12

cp "$RESULTS_DIR/crawl_all_coordinators_in_out_boxplot.png" $ARTIF_DIR/fig13

cp "$RESULTS_DIR/crawl_in_out_ratio_over_time__kruw.png" $ARTIF_DIR/fig14
cp "$RESULTS_DIR/crawl_in_out_ratio_over_time__gingerwallet.png" $ARTIF_DIR/fig14
cp "$RESULTS_DIR/crawl_in_out_ratio_over_time__opencoordinator.png" $ARTIF_DIR/fig14
cp "$RESULTS_DIR/crawl_in_out_ratio_over_time__wasabicoordinator.png" $ARTIF_DIR/fig14

# Requires ANALYZE_DETECT_COORDINATORS_ALG_DETAILED=True (30+ hours execution, runtime heavily dependant on number of available cores)
# python3 -m cj_process.parse_dumplings --cjtype ww2 --env_vars "ANALYZE_DETECT_COORDINATORS_ALG_DETAILED=True" --target-path $TMP_DIR/ 
# cp "$RESULTS_DIR/all_coord_discovery_analysis___drop__randomsingle_aggregated.html" $ARTIF_DIR/fig15
# cp "$RESULTS_DIR/all_coord_discovery_analysis___drop__randomany_aggregated.html" $ARTIF_DIR/fig16
# cp "$RESULTS_DIR/all_coord_discovery_analysis___drop__tail_aggregated.html" $ARTIF_DIR/fig17

echo "Omitted due to high compute time." > $ARTIF_DIR/fig15/info.txt
echo "Omitted due to high compute time." > $ARTIF_DIR/fig16/info.txt
echo "Omitted due to high compute time." > $ARTIF_DIR/fig17/info.txt

cp "$RESULTS_DIR/wasabi2_unknown_2024_e85631/wasabi2_unknown_2024_e85631_cummul_values_norm.png" $ARTIF_DIR/fig18
cp "$RESULTS_DIR/discovered_in_out_ratio_over_time__unknown_2024_e85631.png" $ARTIF_DIR/fig18

cp "$RESULTS_DIR/wasabi2_unknown_2024_28ce7b/wasabi2_unknown_2024_28ce7b_cummul_values_norm.png" $ARTIF_DIR/fig19
cp "$RESULTS_DIR/discovered_in_out_ratio_over_time__unknown_2024_28ce7b.png" $ARTIF_DIR/fig19

echo "Generated from non-public dataset not made public due to privacy reasons as explained in Appendix A." > $ARTIF_DIR/fig20/info.txt

# python3 -m cj_process.parse_dumplings --cjtype ww2 --env_vars "ANALYSIS_WALLET_PREDICTION_EXT=True" --target-path $TMP_DIR/ 
# cp $BASE_PATH/btc/coinjoin-analysis/data/wasabi2/wallet_estimation_matrix_*.json $RESULTS_DIR
# python3 -m cj_process.parse_dumplings --cjtype ww2 --env_vars "ANALYSIS_WALLET_PREDICTION_EXT=True" --target-path $TMP_DIR/ 
# cp "$RESULTS_DIR/wasabi2_zksnacks/wasabi2_zksnacks_wallets_predictions_drops.png" $ARTIF_DIR/fig21

echo "Omitted due to high compute time." > $ARTIF_DIR/fig21/info.txt

echo "Generated from non-public dataset not made public due to privacy reasons as explained in Appendix A." > $ARTIF_DIR/fig22/info.txt

echo "Table 1 contains only definitions - nothing to reproduce" > $ARTIF_DIR/table1/info.txt

echo "Generated from non-public dataset not made public due to privacy reasons as explained in Appendix A." > $ARTIF_DIR/table2/info.txt

cp $RESULTS_DIR/liquidity_summary_*.json $ARTIF_DIR/table2_3

