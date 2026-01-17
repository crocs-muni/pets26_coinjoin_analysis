# Software and data artifacts for pets26_coinjoin_analysis (cutoff time 2025-11-10, blockheight 923063)
Code and data artifacts for PETS'26 paper 'CoinJoin ecosystem insights for Wasabi 1.x, Wasabi 2.x and Whirlpool coordinator-based privacy mixers'
 

| :point_up:    | IMPORTANT: This repository does NOT contain the up-to-date version of code, but only version of the code at the time of paper publication. Visit upstream repository for up-to-date version!   |
|---------------|:------------------------|


## Repository directories structure
  * **coinjoin-analysis** (commit [369fe7c](https://github.com/crocs-muni/coinjoin-analysis/commit/369fe7cfe80c6ff3e95b8b997dd22174fa0a010a)) - Python processing of Dumplings project results. Up-to-date version: https://github.com/crocs-muni/coinjoin-analysis
  * **dumplings** (commit [edce638](https://github.com/crocs-muni/Dumplings/commit/edce63840fdbba6c5a61c0b23e2cccecc5dbb95f)) - .NET project for scanning Bitcoin blockchain for coinjoin transactions. Up-to-date version: https://github.com/crocs-muni/Dumplings/
  * **API_monitoring** - Tool for monitoring Wasabi 2 coordinators based on [LiquiSabi](https://github.com/turbolay/LiquiSabi) with python script for collecting transaction IDs. 
	
**Datasets**

The following datasets are used as a part of processing.
  * Crawled coordinators + manual additions: /coinjoin-analysis/data/wasabi2/txid_coord.json 
  * Manual false positives: /coinjoin-analysis/data/wasabi1/false_cjtxs.json, /coinjoin-analysis/data/wasabi2/false_cjtxs.json, /coinjoin-analysis/data/whirlpool/false_cjtxs.json
  * Processed Dumplings dataset (cutoff time 2025-11-10, blockheight 923063): https://zenodo.org/records/17870025


## Replication

For more details and replication instructions see [ARTIFACT-APPENDIX.md](ARTIFACT-APPENDIX.md)

