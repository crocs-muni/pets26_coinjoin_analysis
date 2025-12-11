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

The artifacts consist of programs for collection of CoinJoin related data – both on-chain and off-chain. Further it features postprocessing scripts for these data and collected datasets.

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

_Minimal hardware requirements_:

* The main hardware requirement is that around 100 GB of RAM is necessary. 
* Around 130 GB of free disk space is necessary.
* No GPU is required
* No specific CPU is required, however, the execution time highly depends on it. 

_Specifications of the used hardware_:
* RAM: 128 GB
* Processor: 13th Gen Intel(R) Core(TM) i7-13700KF
* The results of our experiments should not be influenced by a specific choice of hardware.

### Software Requirements (Required for Functional and Reproduced badges)
* OS: We used Debian 12, however the artifact should work with any reasonably new Linux distribution. 
* Bash
* Python 3.11+
* Necessary python libraries are listed in: https://github.com/crocs-muni/pets26_coinjoin_analysis/blob/main/coinjoin-analysis/requirements.txt
* Datasets needed to run the artifacts are available at: https://zenodo.org/records/17870025


### Estimated Time and Storage Consumption (Required for Functional and Reproduced badges)

- The overall human time required to run the artifact: ~20 minutes
- The overall compute time required to run the artifact: ~4 hours
- The overall disk space consumed by the artifact: ~130 GB


## Environment 

### Accessibility

All artifacts are available from https://github.com/crocs-muni/pets26_coinjoin_analysis

### Set up the environment 

1. Obtain Linux machine with at least 64G RAM (ideally 128GB). Around 100GB is required at peak (only for very short time, compensation of missing space via swap file should be ok)
2. Prepare a base directorie where all the results will be created: e.g. `mkdir work_dir`
3. Set variable `$BASE_PATH` to the working directory - e.g. `export BASE_PATH=/home/user/work_dir`
4. Clone this repository into `$BASE_PATH/`:
```
cd $BASE_PATH
git clone https://github.com/crocs-muni/pets26_coinjoin_analysis.git
```
5. Download dumplings.zip and missing_dumplings_txs.zip into $BASE_PATH
```
cd $BASE_PATH
wget https://zenodo.org/records/17870025/files/dumplings_20251110.zip
mv dumplings_20251110.zip dumplings.zip
wget https://zenodo.org/records/17870025/files/missing_dumplings_txs.zip
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

If the installation of libraries ran without errors and everything is correctly placed in $BASE_PATH (Test 1), the environment should be set up correctly. 

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
missing_dumplings_txs.zip
pets26_coinjoin_analysis
venv
```


## Artifact Evaluation

### Main Results and Claims

#### Main Result 1: Detection of non-public coordinators
  * Means of detection of non-attributed transactions and their prevalence (Figure 6), API monitoring + Dumplings on-chain extraction
  * Algorithm for attribution of transactions to known cooridnators, connected clusters of unattributed transactions (=> unknown coord)
  * Selection of sensible threshold for attribution algorithm (Figure 13, 14)
  * Provide expected list of transactions txid_to_coord_discovered_renamed.json + /Scanner/crawl_datasets.png
  * Reasoning why it is separate cluster (coordinator) - almost no remix to other coordinators (Figure 18, Figure 19) 


#### Main Result 2: Prediction of number of participants
  * Extraction of prediction factor and error bounds from client-side experiments (linear factor + error bounds + selection of outputs due to better stability)
  * Application to coinjoins in time (Figure 10)


#### Main Result 3: Liquidity statistics for separate coordinator
  * Provide expected liquidity_xxx.json files 
  * Postprocessing of collected coinjoin transactions: Extraction, false positives, reordering, filtering
  * CoinJoin liquidity visualizations  


### Experiments

#### Experiment 1: Run all processing scripts 
Time: ~4 hours on Debian Linux, 13th Gen Intel(R) Core(TM) i7-13700KF
Storage: ~130 GB

All the analytical scripts can be simply executed using `run_pets.sh` script as follows(storing both stdout and stderr into `log.txt` for potential debugging):

```
cd $BASE_PATH/coinjoin-analysis
./scripts/run_pets.sh >>log.txt 2>&1
```
If everything went correctly `$BASE_PATH/summary.log` should contain all executed python scripts always followed with line `SUCCESS (elapsed: XX.XX seconds)`. 
In case that some commands are not followed by such a message, re-running these commands should fix the issue. However, it such situation would occur, please send us the `log.txt` file for further investigation. 

##### Experiment results

The last script invoked in the `run_pets.sh` script copied the results into `$BASE_PATH/results` into individual folders for each figure or table. For some figures, it inserted explanation of why the figure is missing, see Limitations section for more details. 

Graphs from Figures 27, 28, 29, 30, 31, 34, 35, 36 for individual months are located in `$BASE_PATH/dumplings_temp/Scanner/<coordinator>/<month>`. 


## Limitations

The replication of the following results and datasets extraction is not included in the Evaluations instructions:
  * Client-side characteristics extraction (Figures 3-5, Table 2) - We do not provide dataset needed for reproduction of these results as it could har users privacy as discussed in the Ethics section of the paper. 
  * We omitted some figures from Appendix (15, 16, 17, 21) due to a very high compute times. These figures are non-essential for the main claims of the paper.
  * Dumplings extraction dataset - We do not instruct the evaluator to recollect the dumplings dataset as it is a very time consuming process. Instead, we provide the dataset. However, we provide the version of Dumplings including our modifications.
  * API dataset collection is not repeatable. It is impossible to reproduce collection of TXIDs, we did by monitoring coordinator API. We provide the dataset. Furthermore, we provide the scripts used in `API_monitoring` and instructions on how to use them. 


## Notes on Reusability

This artifacts provides multiple methods for analysis and visualization of data about coinjoins collected from the Bitcoin blockchain and from Wasabi Wallet 2 coordinator API. These methods can be extended and applied on future data, providing a good starting point for a future, more advanced analysis of statistical properties of coinjoins and investigation of different mixing patterns. 

