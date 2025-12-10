# Artifact Appendix

Paper title: CoinJoin ecosystem insights for Wasabi 1.x, Wasabi 2.x and Whirlpool coordinator-based privacy mixers

Requested Badge(s):
  - **Available**
  - **Functional**
  - **Reproduced**


## Description

These are the digital artifacts of the paper “CoinJoin ecosystem insights for Wasabi 1.x, Wasabi 2.x and Whirlpool coordinator-based privacy mixers” 

```
@article{pets2025_svenda,
  title={CoinJoin ecosystem insights for Wasabi 1.x, Wasabi 2.x and Whirlpool coordinator-based privacy mixers},
  author={Svenda, Petr and Gavenda, Jiri and Mavroudis, Vasilios and Hicks, Chris},
  journal={Proceedings on Privacy Enhancing Technologies},
  year={2025},
  volume={2}
}
```

The artifacts consist of programs for collection of CoinJoin related data – both on-chain and off-chain. Further is features postprocessing scripts for these data and collected datasets.

**Repository directories structure:**
  * **coinjoin-analysis** (commit [369fe7c](https://github.com/crocs-muni/coinjoin-analysis/commit/369fe7cfe80c6ff3e95b8b997dd22174fa0a010a)) - Python processing of Dumplings project results. Up-to-date version: https://github.com/crocs-muni/coinjoin-analysis
  * **dumplings** (commit [edce638](https://github.com/crocs-muni/Dumplings/commit/edce63840fdbba6c5a61c0b23e2cccecc5dbb95f)) - .NET project for scanning Bitcoin blockchain for coinjoin transactions. Up-to-date version: https://github.com/crocs-muni/Dumplings/
  * **API_monitoring** - Tool for monitoring Wasabi 2 coordinators based on [LiquiSabi](https://github.com/turbolay/LiquiSabi) with python script for collecting transaction IDs. 
	
**dumplings** and **API_monitoring** were used only for data collection and are not necessary for artifact evaluation as we provide the necessary datasets

**Datasets**

The following datasets are used as a part of processing.
  * Crawled coordinators + manual additions: /coinjoin-analysis/data/wasabi2/txid_coord.json 
  * Manual false positives: /coinjoin-analysis/data/wasabi1/false_cjtxs.json, /coinjoin-analysis/data/wasabi2/false_cjtxs.json, /coinjoin-analysis/data/whirlpool/false_cjtxs.json
  * Processed Dumplings dataset (cutoff time 2025-11-10, blockheight 923063): https://zenodo.org/records/17870025


### Security/Privacy Issues and Ethical Concerns 

This artifact poses no security or privacy risks. As we do not provide the data from client-side experiments, we believe that there are also no ethical concerns regarding the artifacts. 

## Basic Requirements (Required for Functional and Reproduced badges)

### Hardware Requirements (Required for Functional and Reproduced badges)

_minimal hardware requirements_:

* The main hardware requirement is that around 100 GB of RAM is necessary. 
* Around 150 GB of free disk space is necessary.
* No GPU is required
* No specific CPU is required, however, the execution time highly depends on it. 

_the specifications of the used hardware_:
* RAM: 128 GB
* Processor: 13th Gen Intel(R) Core(TM) i7-13700KF
* The results of our experiments should not be influenced by a specific choice of hardware.

### Software Requirements (Required for Functional and Reproduced badges)
* OS: We used Debian 12, however the artifact should work with any reasonably new Linux distribution. 
* Python 3.11+
* .NET 8 SDK
* Necessary python libraries are listed in: https://github.com/crocs-muni/pets26_coinjoin_analysis/blob/main/coinjoin-analysis/requirements.txt
* Datasets needed to run the artifacts are available at: https://zenodo.org/records/17870025


### Estimated Time and Storage Consumption (Required for Functional and Reproduced badges)

- The overall human and compute times required to run the artifact: 15 minutes
- The overall disk space consumed by the artifact: 150 GB


## Environment 

### Accessibility

All artifacts are available from https://github.com/crocs-muni/pets26_coinjoin_analysis

### Set up the environment 

1. Obtain Linux machine with at least 64G RAM (ideally 128GB). Around 98GB is required at peak (only for very short time, compensation of missing space via swap file is ok)
2. Prepare a base directorie where all the results will be created: e.g. `mkdir work_dir`
3. Set variable `$BASE_PATH` to the working directory -- e.g. `export BASE_PATH=/home/user/work_dir`
4. Clone this repository into `$BASE_PATH/`:
```
cd $BASE_PATH
git clone https://github.com/crocs-muni/pets26_coinjoin_analysis.git
```
5. Download dumplings.zip into $BASE_PATH
```
wget https://zenodo.org/records/17870025/files/dumplings_20251110.zip
mv dumplings_20251110.zip $BASE_PATH/dumplings.zip
```
6. Copy `coinjoin-analysis` into $BASE_PATH
```
cp -r $BASE_PATH/pets26_coinjoin_analysis/coinjoin-analysis $BASE_PATH
```
7. Create python virtual environment
```
python3 -m venv ./venv
source ./venv/bin/activate
```
8. Install python requirements for coinjoin-analysis project:
```
cd $BASE_PATH/coinjoin-analysis
pip install -r requirements.txt
```

### Testing the Environment 

If the installation of libraries ran without errors and all everything is correctly placed in $BASE_PATH (Test 1), the environment should be set up correctly. 

#### Test 1
Run the following:
```
cd $BASE_PATH
ls
```
Expected result:
```
coinjoin-analysis
dumplings.zip
pets26_coinjoin_analysis
venv
```


## Artifact Evaluation

### Main Results and Claims

### Main Result 1: Detection of non-public coordinators
  * Means of detection of non-attributed transactions and their prevalence (Figure 6), API monitoring + Dumplings on-chain extraction
  * Algorithm for attribution to known cooridnators, connected clusters of unattributed (=> unknown coord)
  * Selection of sensible threshold for attribution algorithm (Figure 13, 14)
  * Provide expected list of transactions txid_to_coord_discovered_renamed.json + /Scanner/crawl_datasets.png
  * Reasoning why it is separate cluster (coordinator) - almost no remix to other coords (Figure 18, Figure 19) 


### Main Result 2: Prediction of number of participants
  * Extraction of prediction factor and error bounds from client-side experiments (linear factor + error bounds + selection of outputs due to better stability)
  * Application to coinjoin in time (Figure 10)


### Main Result 3: Liquidity statistics for separate coordinator
  * Provide expected liquidity_xxx.json files 
  * Extraction, false positives, reordering, filtering
  * Liquidity visualizations  


### Experiments

#### Experiment 1: Run all processing scripts 
(~4 hours on Debian Linux, 13th Gen Intel(R) Core(TM) i7-13700KF)

All the analytical scripts can be simply executed using `run_pets.sh` script as follows:

```
cd $BASE_PATH/coinjoin-analysis
./scripts/run_pets.sh
```

##### Experiment results

Inspect results produced in /btc/dumplings_temp2/Scanner/ 
  * Figure 1 and 2 are manually drawn illustration diagrams  
  * Figures 3-5 are composition of ww2_as25_38_privacy_progress.png, ww2_as25_38_anonscoregain.png, ww2_realexp_num_inputs.png, ww2_realexp_num_outputs.png, ww2_realexp_as25_inoutheatmap.png, ww2_realexp_as38_inoutheatmap.png (generated from non-public dataset not shared due to privacy reasons as explained in paper)
  * Figure 6 is /Scanner/crawl_datasets.png
  * Figure 7 is composition of two versions of the same file with SORT_COINJOINS_BY_RELATIVE_ORDER=False (upper graph) and SORT_COINJOINS_BY_RELATIVE_ORDER=True (lower graph) /wasabi2/2023-03-01 00-00-00--2023-04-01 00-00-00_unknown-static-100-1utxo/wasabi2_input_types_nums_notnorm.png
  * Figure 8 from paper is composition of results /wasabi1_zksnacks/2019-12-01 000-00-00--2020-01-01 00-00-00_unknown-static-100-1utxo/wasabi1_zksnacks_input_types_values_notnorm.png and /wasabi1_zksnacks/2020-10-01 00-00-00--2020-11-01 00-00-00_unknown-static-100-1utxo/wasabi1_zksnacks_input_types_nums_norm.png
  * Figure 9 is composition of /wasabi1/wasabi1_zksnacks_cummul_values_norm.png, /whirlpool/whirlpool_cummul_nums_norm_nolegend.png, /wasabi2/wasabi2_cummul_values_norm_nolegend.png
  * Figure 10 is composition of /wasabi2_zksnacks/wasabi2_zksnacks_wallets_predictions_dynamics.png and /wasabi2_kruw/wasabi2_kruw_wallets_predictions_dynamics.png
  * Figure 11 is small part of /wasabi2_opencoordinator/2025-03-01 00-00-00--2025-04-01 00-00-00_unknown-static-100-1utxo/wasabi2_opencoordinator_input_types_values_notnorm.png in days 9-11.3.
  * Figure 12 is ww2_coord_flows_values.png
  * Figure 13 is /Scanner/crawl_all_coordinators_in_out_boxplot.png
  * Figure 14 is composition of crawl_in_out_ratio_over_time__kruw.png, crawl_in_out_ratio_over_time__gingerwallet.png, crawl_in_out_ratio_over_time__opencoordinator.png, crawl_in_out_ratio_over_time__wasabicoordinator.png from /Scanner/ folder 
  * Figure 15 /Scanner/all_coord_discovery_analysis___drop__randomsingle_aggregated.png
  * Figure 16 /Scanner/all_coord_discovery_analysis___drop__randomany2_aggregated.png
  * Figure 17 /Scanner/all_coord_discovery_analysis___drop__tail_aggregated.png
  * Figure 18 is composition of /wasabi2_unknown_2024_e85631/wasabi2_unknown_2024_e85631_cummul_values_norm.png and /Scanner/discovered_in_out_ratio_over_time__unknown_2024_e85631.png
  * Figure 19 is composition of /wasabi2_unknown_2024_28ce7b/wasabi2_unknown_2024_28ce7b_cummul_values_norm.png and /Scanner/discovered_in_out_ratio_over_time__unknown_2024_28ce7b.png
  * Figure 20 is as25_as38_wallet_predict_confidence.png (generated from non-public dataset not shared due to privacy reasons as explained in paper)
  * Figure 21 is composition of /wasabi2_zksnacks/wasabi2_zksnacks_wallets_predictions_drops.png and /wasabi2_kruw/wasabi2_kruw_wallets_predictions_drops.png
  * Figure 22 is composition of ww2_realexp_as25_fees.png and ww2_realexp_as38_fees.png (generated from non-public dataset not shared due to privacy reasons as explained in paper)
  * Table 2 is generated from non-public dataset not shared due to privacy reasons as explained in paper
  * Table 3 is generated from /Scanner/liquidity_summary_xxx.json files where xxx is name of a pool and with usage of "latex_summary", "earliest_cjtx", "earliest_time", "latest_cjtx", "latest_time" and "total_coinjoins" keys.    
  * Table 4 is generated from /Scanner/liquidity_summary_xxx.json files where xxx is name of a pool and with usage of "earliest_cjtx" and "latest_cjtx" keys.

## Limitations

The replication of the following results and datasets extraction is not included in the Evaluations instructions:
  * Client-side characteristics extraction (Figures 3-5, Table 2) -- We do not provide dataset needed for reproduction of these results as it could har users privacy as discussed in the Ethics section of the paper.  
  * Dumplings extraction dataset -- We do not instruct the evaluator to recollect the dumplings dataset as it is a very time consuming process. Instead, we provide the dataset. However, we provide the version of Dumplings including our modifications.
  * API dataset collection is not repeatable. It is impossible to reproduce collection of TXIDs, we did by monitoring coordinator API. We provide the dataset. Furthermore, we provide the scripts used in `API_monitoring` and instructions on how to use them. 


## Notes on Reusability

This artifacts provides multiple methods for analysis and visualization of data about coinjoins collected from the Bitcoin blockchain and from Wasabi Wallet 2 coordinator API. These methods can be extended and applied on future data, providing a good starting point for a future, more advanced analysis of statistical properties of coinjoins and investigation of different mixing patterns. 

