# Wasabi 2 coordinator API monitoring

This folder contains:
1. Modified version of the opensource tool [LiquiSabi](https://github.com/turbolay/LiquiSabi) used for collecting data from Wasabi 2 coordinators
2. Script for mapping collected information about finished coinjoins to TXIDs

### Running modified LiquiSabi

#### Requirements
.NET 8 SDK

#### Running

```
cd LiquiSabi
dotnet run
```

Data will be collected into `human_db.sqlite` and `status_db.sqlite`

### Getting TXIDs

#### Requirements
Python 3 with the `requests` library -- installation using `pip install requests`

#### Running
The TXIDs can be obtained by requesting mempool.space using `get_txids.py`. It expects that `status_db.sqlite` is located in `./LiquiSabi/`.

It can be run simply as  

`python get_txids.py`

