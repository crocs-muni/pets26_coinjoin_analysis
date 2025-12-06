import math
import os
import shutil
import tracemalloc
from datetime import time
from pathlib import Path

from orjson import orjson

import utils
from cj_process.cj_analysis import load_coinjoins_from_file, MIX_EVENT_TYPE
from cj_process import cj_analysis as als, cj_analysis, parse_dumplings
from cj_process.cj_structs import MIX_PROTOCOL

from cj_process.file_check import check_coinjoin_files
from test_full_pipeline import assert_process_dumplings

TESTS = Path(__file__).resolve().parent.parent # …/repo/tests
REPO_ROOT = TESTS.parent                       # …/repo
DATA = REPO_ROOT / "data"                      # …/repo/data
TEMP_DUMPLINGS = REPO_ROOT.parent / "temp_dumplings"


def test_debug():
    utils.run_parse_dumplings("ww1", "process_dumplings","",
                        TEMP_DUMPLINGS)
    utils.run_parse_dumplings("ww1", "split_coordinators","",
                        TEMP_DUMPLINGS)
    return
    parse_dumplings.wasabi_detect_coordinators_evaluation('c:/!blockchains/CoinJoin/Dumplings_Stats_20250820/Scanner/wasabi2_others/')
    return

    als.SORT_COINJOINS_BY_RELATIVE_ORDER = False
    parse_dumplings.process_and_save_intervals_filter(
        'wasabi2_gingerwallet', MIX_PROTOCOL.WASABI2, 'c:/!blockchains/CoinJoin/Dumplings_Stats_20251009/Scanner/',
        '2025-08-23 00:00:07.000', '2025-12-06 23:59:59.000',
    'Wasabi2CoinJoins.txt', 'Wasabi2PostMixTxs.txt', None,
                                             False, True)

    return
    utils.run_parse_dumplings("ww2", "", "ANALYZE_DETECT_COORDINATORS_ALG=True", TEMP_DUMPLINGS)


    return
    #als.analyze_coordinator_detection(os.path.join(TEMP_DUMPLINGS, 'Scanner', 'wasabi2'), ['kruw'])
    als.analyze_coordinator_detection(os.path.join(TEMP_DUMPLINGS, 'Scanner', 'wasabi2_mega'), ['mega'])

    return
#    utils.run_parse_dumplings("ww2", "process_dumplings", "", 'c:/!blockchains/CoinJoin/temp_dumplings/')
    #utils.run_parse_dumplings("ww2", "process_dumplings", "", 'c:/!blockchains/CoinJoin/temp_dumplings/')
    utils.run_parse_dumplings("ww2", "detect_false_positives", "", 'c:/!blockchains/CoinJoin/temp_dumplings/')
    return

    parse_dumplings.wasabi_detect_coordinators("wasabi2", MIX_PROTOCOL.WASABI2,
                                               'c:/!blockchains/CoinJoin/temp_dumplings/Scanner/wasabi2_others/')
    return

    parse_dumplings.wasabi_detect_coordinators("wasabi2", MIX_PROTOCOL.WASABI2,
                                               'c:/!blockchains/CoinJoin/Dumplings_Stats_20250727/Scanner/wasabi2_others/')
    return


    # Base addresses
    base_txs = ['dcbddb28cfe2682e6135be36f0afe6f8e7ec0055d2786cad09806e76c6a95fbf',
                '075b01cf63d35fe58a538511c59e95f4e150f843b582381427b22e6169dd31eb',
                '39960bb706d0233d93013f3e7443eedb496b9325ffd9ba43dbb941d70f72a6cb',
                'd6ebaf5e1b4fdfa149ccc6fefb23037ef582ee6b5892c00c2d31c0a086b6c96d',
                'f4d12e0c1b7fd30c7a7690f715fbb4d1e8bd101ee1b71e1abae3c341f647915b']
    generate_normalized_json(op.target_base_path, base_txs)

    return


    parse_dumplings.wasabi_detect_coordinators("wasabi2", MIX_PROTOCOL.WASABI2,
                                               'c:/!blockchains/CoinJoin/Dumplings_Stats_20250727/Scanner/wasabi2_others/')
    return

    # #WW2 transaction misclassified: dbeba7d9dcc7944ac27dabdb1e01c516c56b449193ed5f59eb8629efc52ac774
    # utils.run_parse_dumplings("ww2", "process_dumplings",
    #                     None,
    #                     'c:/!blockchains/CoinJoin/Dumplings_Stats_postmixfix/')
    # return

    utils.run_parse_dumplings("ww2", None,
                        f"RESTORE_FALSE_POSITIVES_FOR_OTHERS=True",
                        'c:/!blockchains/CoinJoin/Dumplings_Stats_postmixfix/')
    return

    parse_dumplings.wasabi_detect_coordinators("wasabi2", MIX_PROTOCOL.WASABI2,
                                               'c:/!blockchains/CoinJoin/Dumplings_Stats_20250727/Scanner/wasabi2_others/')
    return

    extract_dir = TEMP_DUMPLINGS
    utils.run_parse_dumplings("jm", "plot_coinjoins",
                        f"PLOT_REMIXES_MULTIGRAPH=False",
                        extract_dir)
    return

    data = load_coinjoins_from_file('c:/!blockchains/CoinJoin/temp_dumplings/Scanner/joinmarket_all/', None, False)
    txnull = []
    for txid in data['coinjoins'].keys():
        for index in data['coinjoins'][txid]['outputs']:
            if data['coinjoins'][txid]['outputs'][index]['script_type'] == 'TxNullData':
                print(f'{txid} has TxNullData output')
                txnull.append(txid)
    print(f'Coinjoins with OP_RETURN output (likely false positive): {len(txnull)}')
    print(f'Total coinjoins : {len(data['coinjoins'])}')

    #     utils.run_parse_dumplings("ww2", None,
    # #                        f"VISUALIZE_ALL_COINJOINS_INTERVALS=True;MIX_IDS=['wasabi2', 'wasabi2_zksnacks'];interval_start_date={interval_start_date};interval_stop_date={interval_stop_date}",
    #                         f"VISUALIZE_ALL_COINJOINS_INTERVALS=True;MIX_IDS=['wasabi2_others'];interval_start_date={interval_start_date};interval_stop_date={interval_stop_date}",
    #                         extract_dir)
    #     return


def test_old_debug():
    coord = 'wasabi2_others'
    # coord = 'wasabi1_mystery'
    target_path = ""
    target_load_path = os.path.join(target_path, coord)

    tracemalloc.start()

    start_snapshot = tracemalloc.take_snapshot()
    all_data = als.load_json_from_file(os.path.join(target_load_path, f'coinjoin_tx_info.json'))
    end_snapshot = tracemalloc.take_snapshot()
    stats = end_snapshot.compare_to(start_snapshot, 'lineno')
    for stat in stats[:10]:
        print(stat)

    all_data_slim = all_data

    tic = time.perf_counter()
    txid_map = als.streamline_coinjoins_structure(all_data_slim)
    print(f"streamline_coinjoins_structure() {time.perf_counter() - tic:.4f}s")

    als.save_json_to_file(os.path.join(target_load_path, 'coinjoin_tx_info_slim.json'), all_data_slim)
    als.save_json_to_file_pretty(os.path.join(target_load_path, 'txid_map.json'), txid_map)

    start_snapshot = tracemalloc.take_snapshot()
    data = als.load_json_from_file(os.path.join(target_load_path, f'coinjoin_tx_info_slim.json'))
    end_snapshot = tracemalloc.take_snapshot()
    stats = end_snapshot.compare_to(start_snapshot, 'lineno')

    # Print top memory differences
    for stat in stats[:10]:
        print(stat)
    exit(42)

    coord = 'wasabi2_others'
    # coord = 'wasabi1_mystery'
    target_load_path = os.path.join(target_path, coord)
    all_data = als.load_coinjoins_from_file(target_load_path, None, True)
    print(list(all_data['coinjoins'].keys())[0:3])
    als.PERF_USE_COMPACT_CJTX_STRUCTURE = True
    all_data = als.load_coinjoins_from_file(target_load_path, None, True)
    print(list(all_data['coinjoins'].keys())[0:3])
    exit(42)

    coord = 'wasabi2_others'
    # coord = 'wasabi1_mystery'
    # coord = 'wasabi1_others'
    target_load_path = os.path.join(target_path, coord)

    tic = time.perf_counter()
    all_data = als.load_json_from_file(os.path.join(target_load_path, f'coinjoin_tx_info.json'))
    print(f"load_json_from_file() {time.perf_counter() - tic:.4f}s")

    als.print_liquidity_summary(all_data["coinjoins"], f'{coord}')

    tic = time.perf_counter()
    all_data_slim = all_data
    als.streamline_coinjoins_structure(all_data_slim)
    print(f"streamline_coinjoins_structure() {time.perf_counter() - tic:.4f}s")

    als.print_liquidity_summary(all_data["coinjoins"], f'{coord}')

    als.save_json_to_file(os.path.join(target_load_path, 'coinjoin_tx_info_slim.json'), all_data_slim)
    als.save_json_to_file_pretty(os.path.join(target_load_path, 'txid_map.json'), txid_map)
    exit(42)

    coord = 'wasabi2_others'
    coord = 'wasabi1_mystery'
    # coord = 'wasabi1_others'
    target_load_path = os.path.join(target_path, coord)

    # In RAM processing
    all_data = als.load_coinjoins_from_file(target_load_path, None, False)
    tic = time.perf_counter()
    als.print_liquidity_summary(all_data["coinjoins"], f'{coord}')
    print(f"print_liquidity_summary() for legacy in RAM {time.perf_counter() - tic:.4f}s")
    del (all_data)

    # SQL processing
    all_data = als.load_coinjoins_from_file_sqlite(target_load_path, None, False)
    tic = time.perf_counter()
    als.print_liquidity_summary(all_data["coinjoins"], f'{coord}')
    print(f"print_liquidity_summary() for SQL {time.perf_counter() - tic:.4f}s")

    # print(list(all_data["coinjoins"].keys())[3])
    # als.print_liquidity_summary(all_data["coinjoins"], f'{coord}')

    exit(42)

    data = detect_additional_cjtxs('wasabi1_mystery', MIX_PROTOCOL.WASABI1,
                                   os.path.join(target_path, 'wasabi1_mystery'))
    exit(42)

    estimate_wallet_prediction_factor(target_path, 'wasabi2_kruw')
    estimate_wallet_prediction_factor(target_path, 'wasabi2_zksnacks')
    estimate_wallet_prediction_factor(target_path, 'wasabi1')
    exit(42)

    example_path = 'c:/!blockchains/CoinJoin/coinjoin_tx_info.json'
    data = als.load_json_from_file(example_path)
    txids = data['coinjoins'].keys()
    for txid in txids:
        indexes = data['coinjoins'][txid]['inputs'].keys()

        def set_artifical_values(item: dict, value, mix_type, burn_time):
            item['value'] = value
            item['mix_event_type'] = mix_type
            item['burn_time_cjtxs'] = burn_time
            return item

        def variant1(data: dict):
            index = 0

            item = data['coinjoins'][txid]['inputs'][str(index)]
            item = set_artifical_values(item, 3000, MIX_EVENT_TYPE.MIX_ENTER.name, 1)
            index = index + 1

            item = data['coinjoins'][txid]['inputs'][str(index)]
            item = set_artifical_values(item, 1200, MIX_EVENT_TYPE.MIX_REMIX.name, 1)
            data['coinjoins'][txid]['inputs'][str(index)]['is_standard_denom'] = False
            index = index + 1

            item = data['coinjoins'][txid]['inputs'][str(index)]
            item = set_artifical_values(item, 2000, MIX_EVENT_TYPE.MIX_REMIX.name, 1)
            index = index + 1

            item = data['coinjoins'][txid]['inputs'][str(index)]
            item = set_artifical_values(item, 2500, MIX_EVENT_TYPE.MIX_REMIX.name, 2)
            index = index + 1

            item = data['coinjoins'][txid]['inputs'][str(index)]
            item = set_artifical_values(item, 1500, MIX_EVENT_TYPE.MIX_REMIX.name, 3)
            index = index + 1

            item = data['coinjoins'][txid]['inputs'][str(index)]
            item = set_artifical_values(item, 1400, MIX_EVENT_TYPE.MIX_REMIX.name, 6)
            index = index + 1

            item = data['coinjoins'][txid]['inputs'][str(index)]
            item = set_artifical_values(item, 800, MIX_EVENT_TYPE.MIX_REMIX.name, 20)
            index = index + 1

            item = data['coinjoins'][txid]['inputs'][str(index)]
            item = set_artifical_values(item, 900, MIX_EVENT_TYPE.MIX_REMIX.name, 1000)
            index = index + 1

            item = data['coinjoins'][txid]['inputs'][str(index)]
            item = set_artifical_values(item, 800, MIX_EVENT_TYPE.MIX_REMIX.name, 2000)
            index = index + 1

            item = data['coinjoins'][txid]['inputs'][str(index)]
            item = set_artifical_values(item, 4000, MIX_EVENT_TYPE.MIX_REMIX_FRIENDS_WW1.name, 10)
            index = index + 1

        def variant2(data: dict):
            index = 0

            item = data['coinjoins'][txid]['inputs'][str(index)]
            item = set_artifical_values(item, 2000, MIX_EVENT_TYPE.MIX_ENTER.name, 1)
            index = index + 1

            item = data['coinjoins'][txid]['inputs'][str(index)]
            item = set_artifical_values(item, 500, MIX_EVENT_TYPE.MIX_REMIX.name, 1)
            data['coinjoins'][txid]['inputs'][str(index)]['is_standard_denom'] = False
            index = index + 1

            item = data['coinjoins'][txid]['inputs'][str(index)]
            item = set_artifical_values(item, 1500, MIX_EVENT_TYPE.MIX_REMIX.name, 1)
            index = index + 1

            item = data['coinjoins'][txid]['inputs'][str(index)]
            item = set_artifical_values(item, 1500, MIX_EVENT_TYPE.MIX_REMIX.name, 2)
            index = index + 1

            item = data['coinjoins'][txid]['inputs'][str(index)]
            item = set_artifical_values(item, 1500, MIX_EVENT_TYPE.MIX_REMIX.name, 3)
            index = index + 1

            item = data['coinjoins'][txid]['inputs'][str(index)]
            item = set_artifical_values(item, 1400, MIX_EVENT_TYPE.MIX_REMIX.name, 6)
            index = index + 1

            item = data['coinjoins'][txid]['inputs'][str(index)]
            item = set_artifical_values(item, 800, MIX_EVENT_TYPE.MIX_REMIX.name, 20)
            index = index + 1

            item = data['coinjoins'][txid]['inputs'][str(index)]
            item = set_artifical_values(item, 900, MIX_EVENT_TYPE.MIX_REMIX.name, 1000)
            index = index + 1

            item = data['coinjoins'][txid]['inputs'][str(index)]
            item = set_artifical_values(item, 800, MIX_EVENT_TYPE.MIX_REMIX.name, 2000)
            index = index + 1

            item = data['coinjoins'][txid]['inputs'][str(index)]
            item = set_artifical_values(item, 2000, MIX_EVENT_TYPE.MIX_REMIX_FRIENDS_WW1.name, 10)
            index = index + 1

            return index

        index = variant2(data)

        for in_index in list(data['coinjoins'][txid]['inputs'].keys()):
            if in_index not in [str(i) for i in range(0, index)]:
                del data['coinjoins'][txid]['inputs'][in_index]

    als.save_json_to_file(example_path + '.trim', data)

    exit(42)
    wasabi_plot_remixes('whirlpool_5M', MIX_PROTOCOL.WHIRLPOOL, os.path.join(target_path, 'whirlpool_5M'),
                        'coinjoin_tx_info.json', True, True, None,
                        None, op.PLOT_REMIXES_MULTIGRAPH, op.PLOT_REMIXES_SINGLE_INTERVAL)
    exit(42)
    op.target_base_path = 'c:/!blockchains/CoinJoin/!Jirka_small_coinjoins/'
    base_txs = []
    base_txs.extend(als.load_json_from_file(os.path.join(op.target_base_path, 'small1.json')).keys())
    base_txs.extend(als.load_json_from_file(os.path.join(op.target_base_path, 'small2.json')).keys())
    base_txs.extend(als.load_json_from_file(os.path.join(op.target_base_path, 'small3.json')).keys())
    generate_normalized_json(op.target_base_path, base_txs)

    exit(42)
    wasabi2_recompute_inputs_outputs_other_pools(['kruw'], target_path, MIX_PROTOCOL.WASABI2)
    exit(42)



    address_out, _ = als.get_address_legacy('0014194311ad28daaedfd1346bdf6cb2603b848f5701', 'TxWitnessV0Keyhash')
    address_out, _ = als.get_address('0014194311ad28daaedfd1346bdf6cb2603b848f5701')
    expected = 'bc1qr9p3rtfgm2hdl5f5d00kevnq8wzg74cpzuzj2m'
    assert address_out == expected, f'{expected} expected, but {address_out} obtained'

    address_in, _ = als.get_address_legacy('0014ba5241b6abf4fbbaaf5b99b855e645bb464f18a7', 'TxWitnessV0Keyhash')
    expected = 'bc1qhffyrd4t7nam4t6mnxu9tej9hdry7x98mdn4a9'
    assert address_in == expected, f'{expected} expected, but {address_in} obtained'

    analyze_zksnacks_output_clusters('wasabi2', target_path)
    exit(42)

    # process_inputs_distribution2('wasabi2_test', MIX_PROTOCOL.WASABI2, target_path, 'Wasabi2CoinJoins.txt', True)
    process_estimated_wallets_distribution('wasabi2', target_path, [1.8, 2.0, 2.3, 2.7], True)
    exit(42)

def test_run_cj_process_sw():
    interval_start_date = "2024-03-01 00:00:00.000000"
    interval_stop_date = "2024-03-31 00:00:00.000000"
    source_zip = os.path.abspath(os.path.join(TESTS, "fixtures", "dumplings__sw_202403.zip"))
    extract_dir = TEMP_DUMPLINGS
    target_zip = os.path.abspath(f"{extract_dir}/dumplings.zip")

    # Prepare test data from zip file
    utils.prepare_from_zip_file(extract_dir, source_zip, target_zip)

    #
    # Run initial processing
    #
    utils.run_parse_dumplings("sw", "process_dumplings",
                        f"interval_start_date='{interval_start_date}';interval_stop_date='{interval_stop_date}'",
                        extract_dir, False)
    # BUGBUG: Basic extraction of all transactions will succeed, but pool splittuing will fail as we are not starting
    # from initial seed transactions from from 03/2024. The whirlpool folder structure is created

    # ASSERT
    assert_process_dumplings(extract_dir, 'whirlpool', 196, 26033, 26033,
                             4231, (333000000, 30), (500000000, 3), None,
                             None,
                             None)

    for coord in ["wasabi1"]:
        target_dir = os.path.join(extract_dir, "Scanner", coord)
        shutil.copy(os.path.join(DATA, "wasabi1", "false_cjtxs.json"), os.path.join(target_dir, "false_cjtxs.json"))
        shutil.copy(os.path.join(DATA, "wasabi1", "fee_rates.json"), os.path.join(target_dir, "fee_rates.json"))

    #
    # Run false positives detection
    #
    utils.run_parse_dumplings("sw", "detect_false_positives",
                        f"interval_start_date='{interval_start_date}';interval_stop_date='{interval_stop_date}'",
                        extract_dir)
    # ASSERT
    for coord in ["wasabi1"]:
        with open(os.path.join(extract_dir, "Scanner", coord, "no_remix_txs.json"), "r") as file:
            results = orjson.loads(file.read())
            assert len(results[
                           'inputs_noremix']) == 12, f"Expected {12} no inputs remix coinjoins, got {len(results['inputs_noremix'])}"
            assert len(results[
                           'outputs_noremix']) == 6, f"Expected {6} no outputs remix coinjoins, got {len(results['outputs_noremix'])}"
            assert len(results[
                           'both_noremix']) == 2, f"Expected {2} both no remix coinjoins, got {len(results['both_noremix'])}"
            assert len(results[
                           'specific_denoms_noremix_in']) == 0, f"Expected {0} specific denoms noinput in, got {len(results['specific_denoms_noremix_in'])}"
            assert len(results[
                           'specific_denoms_noremix_out']) == 0, f"Expected {0} specific denoms noinput out, got {len(results['specific_denoms_noremix_out'])}"
            assert len(results[
                           'specific_denoms_noremix_both']) == 0, f"Expected {0} specific denoms noinput both, got {len(results['specific_denoms_noremix_both'])}"
            assert len(results[
                           'inputs_address_reuse_0_70']) == 0, f"Expected {0} input address reuse, got {len(results['inputs_address_reuse_0_70'])}"
            assert len(results[
                           'outputs_address_reuse_0_70']) == 0, f"Expected {0} output address reuse, got {len(results['outputs_address_reuse_0_70'])}"

    #
    # Detect and split additional coordinators
    #
    # utils.run_parse_dumplings("sw", "detect_coordinators", f"interval_start_date={interval_start_date};interval_stop_date={interval_stop_date}", extract_dir)
    utils.run_parse_dumplings("sw", "split_coordinators",
                        f"interval_start_date='{interval_start_date}';interval_stop_date='{interval_stop_date}'",
                        extract_dir)
    # TODO: ASSERT 'txid_coord_discovered_renamed.json'

    # Remove all directories except two relevenat ones for this test
    for coord in ["wasabi1", "wasabi1_mystery", "wasabi1_others", "wasabi1_zksnacks"]:
        target_path = os.path.join(extract_dir, "Scanner", coord)
        intervals = [d for d in os.listdir(target_path) if os.path.isdir(os.path.join(target_path, d))]
        for interval in intervals:
            if (interval != "2021-08-01 00-00-00--2021-09-01 00-00-00_unknown-static-100-1utxo"
                    and interval != "2021-09-01 00-00-00--2021-10-01 00-00-00_unknown-static-100-1utxo"):
                shutil.rmtree(os.path.join(target_path, interval))

    for coord in ["wasabi1", "wasabi1_others", "wasabi1_zksnacks", "wasabi1_mystery"]:
        target_dir = os.path.join(extract_dir, "Scanner", coord)
        shutil.copy(os.path.join(DATA, "wasabi1", "false_cjtxs.json"), os.path.join(target_dir, "false_cjtxs.json"))
        shutil.copy(os.path.join(DATA, "wasabi1", "fee_rates.json"), os.path.join(target_dir, "fee_rates.json"))
        shutil.copy(os.path.join(DATA, "wasabi1", "txid_coord.json"), os.path.join(target_dir, "txid_coord.json"))
        shutil.copy(os.path.join(DATA, "wasabi1", "txid_coord_t.json"), os.path.join(target_dir, "txid_coord_t.json"))

    #
    # Analyze liquidity
    #
    utils.run_parse_dumplings("sw", None,
                        f"ANALYSIS_LIQUIDITY=True;interval_start_date='{interval_start_date}';interval_stop_date='{interval_stop_date}'",
                        extract_dir)
    # ASSERT

    expected_results = {
        "wasabi1_zksnacks": {"total_fresh_inputs_value": 934.54815701, "total_friends_inputs_value": 0.0,
                             "total_unmoved_outputs_value": 838.22085515, "total_leaving_outputs_value": 96.31540543,
                             "total_nonstandard_leaving_outputs_value": 39.21255878,
                             "total_fresh_inputs_without_nonstandard_outputs_value": 895.33559823},
        "wasabi1_others": {"total_fresh_inputs_value": 1800.87207463, "total_friends_inputs_value": 0.0,
                           "total_unmoved_outputs_value": 1118.58730176, "total_leaving_outputs_value": 489.54757815,
                           "total_nonstandard_leaving_outputs_value": 473.459393,
                           "total_fresh_inputs_without_nonstandard_outputs_value": 1327.41268163,
                           }}
    for coord in expected_results.keys():
        with open(os.path.join(extract_dir, "Scanner", f"liquidity_summary_{coord}.json"), "r") as file:
            results = orjson.loads(file.read())
            for key in expected_results[coord].keys():
                assert math.isclose(results[key], expected_results[coord][
                    key]), f"Expected {expected_results[coord][key]} for {key}, got {results[key]}"

    #
    # Plot some graphs
    #
    utils.run_parse_dumplings("sw", "plot_coinjoins",
                        f"PLOT_REMIXES_MULTIGRAPH=False;MIX_IDS=['wasabi1', 'wasabi1_others', 'wasabi1_zksnacks'];interval_start_date='{interval_start_date}';interval_stop_date='{interval_stop_date}'",
                        extract_dir)
    utils.run_parse_dumplings("sw", "plot_coinjoins",
                        f"PLOT_REMIXES_MULTIGRAPH=True;MIX_IDS=['wasabi1', 'wasabi1_others', 'wasabi1_zksnacks'];interval_start_date='{interval_start_date}';interval_stop_date='{interval_stop_date}'",
                        extract_dir)
    utils.run_parse_dumplings("sw", None,
                        f"VISUALIZE_ALL_COINJOINS_INTERVALS=True;MIX_IDS=['whirlpool', 'wasabi1_others', 'wasabi1_zksnacks'];interval_start_date='{interval_start_date}';interval_stop_date='{interval_stop_date}'",
                        extract_dir)

    # ASSERT
    file_check = check_coinjoin_files(os.path.join(extract_dir, 'Scanner'))
    assert len(file_check['results']['whirlpool']['mix_base_files'][
                   'missing_files']) == 0, f"Missing files: {file_check['results']['wasabi1']['mix_base_files']['missing_files']}"
    assert len(file_check['results']['wasabi1_zksnacks']['mix_base_files'][
                   'missing_files']) == 5, f"Missing files: {file_check['results']['wasabi1_zksnacks']['mix_base_files']['missing_files']}"
