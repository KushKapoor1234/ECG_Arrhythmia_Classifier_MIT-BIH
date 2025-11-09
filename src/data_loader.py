import wfdb
import os
import glob
from tqdm import tqdm

# These records are known to have issues or are not standard ECGs
# Paced records have different morphology and are often excluded.
RECORDS_TO_EXCLUDE = ["102", "104", "107", "217"]


def load_all_records(database_name):
    """
    Loads all valid records (signal, annotations, metadata) 
    from the specified local database directory.

    :param database_name: The path to the local data folder (e.g., 'data')
    """

    search_path = os.path.join(database_name, "*.hea")
    header_files = glob.glob(search_path)
    records_list = [os.path.basename(f).replace(".hea", "") for f in header_files]

    if not records_list:
        print(f"Error: No '.hea' files found in directory: {database_name}")
        print("Please ensure your downloaded database files are in the 'data' folder (e.g., 100.dat, 100.hea, 100.atr).")
        return []

    all_records = []

    print(f"Found {len(records_list)} local records. Loading and parsing from '{database_name}'...")

    for rec_name in tqdm(records_list):
        if rec_name in RECORDS_TO_EXCLUDE:
            continue

        try:
            record_path = os.path.join(database_name, rec_name)

            # Read the record and annotation locally
            record = wfdb.rdrecord(record_path)
            annotation = wfdb.rdann(record_path, "atr")

            all_records.append(
                {
                    "name": rec_name,
                    "signal": record.p_signal,
                    "fs": record.fs,
                    "ann_samples": annotation.sample,
                    "ann_symbols": annotation.symbol,
                }
            )

        except Exception as e:
            print(f"Error loading record {rec_name}: {e}")

    return all_records
