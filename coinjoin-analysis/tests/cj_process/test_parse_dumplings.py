import os
import shutil
from pathlib import Path

from orjson import orjson

import utils
from cj_process.cj_analysis import load_coinjoins_from_file

TESTS = Path(__file__).resolve().parent.parent # …/repo/tests
REPO_ROOT = TESTS.parent                       # …/repo
DATA = REPO_ROOT / "data"                      # …/repo/data
TEMP_DUMPLINGS = REPO_ROOT.parent / "temp_dumplings"


def test_joinmarket_filter_false_positives():
    source_zip = os.path.abspath(os.path.join(TESTS, "fixtures", "jm_filter_false_positives.zip"))
    extract_dir = TEMP_DUMPLINGS
    target_zip = os.path.abspath(f"{extract_dir}/dumplings.zip")

    # Prepare test data from zip file
    utils.prepare_from_zip_file(extract_dir, source_zip, target_zip)

    returncode = utils.run_parse_dumplings("jm", "process_dumplings", None, extract_dir)
    assert returncode == 0, f"Expected returncode 0, got {returncode}"

    expected_num_cjtxs = 6
    with open(os.path.join(extract_dir, "Scanner", "joinmarket_all", "coinjoin_tx_info.json"), "r") as file:
        coinjoins = orjson.loads(file.read())
        assert len(coinjoins['coinjoins']) == expected_num_cjtxs, f"Expected {expected_num_cjtxs} coinjoins, got {len(coinjoins['coinjoins'])}"


def test_ww2_wasabi_detect_coordinators():
    source_zip = os.path.abspath(os.path.join(TESTS, "fixtures", "ww2_coordinators_052024-072024.zip"))
    extract_dir = TEMP_DUMPLINGS
    target_zip = os.path.abspath(f"{extract_dir}/dumplings.zip")

    # # Prepare test data from zip file
    # utils.prepare_from_zip_file(extract_dir, source_zip, target_zip)
    #
    # returncode = utils.run_parse_dumplings("ww2", "process_dumplings", None, extract_dir)
    # assert returncode == 0, f"Expected returncode 0, got {returncode}"

    for coord in ["wasabi2_others"]:
        target_dir = os.path.join(extract_dir, "Scanner", coord)
        shutil.copy(os.path.join(DATA, "wasabi2", "false_cjtxs.json"), os.path.join(target_dir, "false_cjtxs.json"))
        shutil.copy(os.path.join(DATA, "wasabi2", "fee_rates.json"), os.path.join(target_dir, "fee_rates.json"))
        shutil.copy(os.path.join(DATA, "wasabi2", "txid_coord.json"), os.path.join(target_dir, "txid_coord.json"))

    returncode = utils.run_parse_dumplings("ww2", "detect_coordinators", None, extract_dir)
    assert returncode == 0, f"Expected returncode 0, got {returncode}"

    # returncode = utils.run_parse_dumplings("ww2", "split_coordinators", None, extract_dir)
    # assert returncode == 0, f"Expected returncode 0, got {returncode}"

    # expected_num_cjtxs = 6
    # with open(os.path.join(extract_dir, "Scanner", "joinmarket_all", "coinjoin_tx_info.json"), "r") as file:
    #     coinjoins = orjson.loads(file.read())
    #     assert len(coinjoins['coinjoins']) == expected_num_cjtxs, f"Expected {expected_num_cjtxs} coinjoins, got {len(coinjoins['coinjoins'])}"
