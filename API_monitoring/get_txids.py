import sqlite3
import json
import requests
from time import sleep


MEMPOOL_API = "https://mempool.space/api"
def parse_outpoint(outpoint_hex: str):
    """
    Parse Bitcoin outpoint into txid (big endian) and vout index.

    :param outpoint_hex: Hex string of outpoint (e.g. from your JSON)
    :return: (txid, vout)
    """
    # Strip "0x" if it sneaks in
    outpoint_hex = outpoint_hex.lower().replace("0x", "")

    # txid is first 32 bytes (64 hex chars), little-endian -> reverse
    txid_le = outpoint_hex[:64]
    txid_be = "".join(reversed([txid_le[i:i+2] for i in range(0, 64, 2)]))

    # vout is last 4 bytes, little-endian
    vout_hex = outpoint_hex[64:]
    vout = int.from_bytes(bytes.fromhex(vout_hex), byteorder="little")

    return txid_be, vout

def get_spending_txid(outpoint_hex: str):
    """
    Given an outpoint, return the txid of the transaction that spends it (if any).
    """
    txid, vout = parse_outpoint(outpoint_hex)

    url = f"{MEMPOOL_API}/tx/{txid}/outspend/{vout}"
    #print(url)
    sleep(1)
    resp = requests.get(url)
    if resp.status_code != 200:
        raise Exception(f"Failed to fetch outspend info: {resp.text}")
    data = resp.json()
    #print(data)
    if data.get("spent"):
        return data["txid"]
    return None

def get_io_counts(txid: str):
    """
    Get number of inputs and outputs for a Bitcoin transaction.
    """
    url = f"{MEMPOOL_API}/tx/{txid}"
    resp = requests.get(url)
    if resp.status_code != 200:
        raise Exception(f"Failed to fetch transaction {txid}: {resp.text}")
    tx = resp.json()

    num_inputs = len(tx.get("vin", []))
    num_outputs = len(tx.get("vout", []))

    return num_inputs, num_outputs

def io_counts_from_events(events):
    num_inputs = 0
    num_outputs = 0
    for event in events:
        if event["type"] == "InputAdded":
            num_inputs += 1
        elif event["type"] == "OutputAdded":
            num_outputs += 1
    return num_inputs, num_outputs

def lower_keys(obj):
    if isinstance(obj, dict):
        return {k.lower(): lower_keys(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [lower_keys(i) for i in obj]
    else:
        return obj

coordinators = [
('opencoordinator', 'https://api.opencoordinator.org/'),
('coinjoin_nl', 'https://coinjoin.nl/'),
('kruw', 'https://coinjoin.kruw.io/'),
('gingerwallet', 'https://api.gingerwallet.io/')]

conn = sqlite3.connect("file:./LiquiSabi/status_db.sqlite?mode=ro", uri=True)

cursor = conn.cursor()

searched_round_ids = set()

for coord, url in coordinators:
    with open(f"./txids_{coord}.txt", "a") as file:
        for row in cursor.execute("SELECT * FROM Logs WHERE coordinator='https://coinjoin.kruw.io/';"):

            data = lower_keys(json.loads(row[2]))

            if data["phase"] != 4 or data["endroundstate"] != 4:
                continue

            for event in data["coinjoinstate"]["events"]:
                b = io_counts_from_events(data["coinjoinstate"]["events"])
                if b[1] == 0:
                    continue
                if event["type"] == "InputAdded":

                    outpoint = event["coin"]["outpoint"]

                    if data["id"] in searched_round_ids:
                        continue
                    txid = get_spending_txid(outpoint)
                    a = get_io_counts(txid)
                    b = io_counts_from_events(data["coinjoinstate"]["events"])
                    
                    if a == b:
                        file.write(txid)
                        file.write(";")
                        file.write(row[0])
                        file.write("\n")
                        file.flush()

                    searched_round_ids.add(data["id"])
                    
                    break
                else:
                    continue

conn.close()