import pandas as pd
from itertools import combinations
from collections import defaultdict
from blocking_utils.blocking_utils import compute_similarity
import logging
import time
from tqdm import tqdm
import multiprocessing as mp
from math import comb

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

def generate_sorting_key(row, key_columns):
    """
    Generate a sorting key for a record based on specified columns.
    """
    combined_key = "".join(
        str(row[col]).strip().lower().replace(" ", "").replace(".", "")
        for col in key_columns
        if pd.notnull(row[col])
    )
    return combined_key

def compute_similarity_pair(args):
    """
    Computes the similarity score for a given pair of record IDs.
    """
    record_id1, record_id2 = args
    global df
    try:
        row1 = df[df['record_id'] == record_id1].squeeze()
        row2 = df[df['record_id'] == record_id2].squeeze()
        sim_score = compute_similarity(row1, row2)
        return (record_id1, record_id2, sim_score)
    except Exception as e:
        logger.error(f"Error computing similarity for pair ({record_id1}, {record_id2}): {e}")
        return (record_id1, record_id2, 0.0)

def init_worker(dataframe):
    """
    Initializer for worker processes to set the global DataFrame.
    """
    global df
    df = dataframe

def main():
    logger.info("Loading data...")
    start_time = time.time()
    df = pd.read_csv("data/processed/external_parties_test.csv")

    # Assign unique record IDs
    df["record_id"] = df.index

    # Specify columns for sorting key
    key_columns = [
        "parsed_name",
        "parsed_address_street_name",
        "parsed_address_postal_code",
    ]

    # Generate sorting keys
    logger.info("Generating sorting keys...")
    df["sorting_key"] = df.apply(lambda row: generate_sorting_key(row, key_columns), axis=1)

    # Sort the DataFrame based on the sorting key
    logger.info("Sorting records...")
    df_sorted = df.sort_values("sorting_key").reset_index(drop=True)

    # Define window size
    window_size = 4  # Adjust this size based on the dataset and desired performance
    logger.info(f"Using sliding window of size {window_size}")

    # Initialize Union-Find data structure
    parent = {record_id: record_id for record_id in df["record_id"]}

    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]  # Path compression
            u = parent[u]
        return u

    def union(u, v):
        pu, pv = find(u), find(v)
        if pu != pv:
            parent[pu] = pv

    # Process candidate pairs window by window
    logger.info("Processing candidate pairs window by window...")
    total_records = len(df_sorted)
    num_windows = total_records - window_size + 1
    similarity_threshold = 0.75  # Adjust based on your requirements

    # Initialize multiprocessing pool
    pool = mp.Pool(processes=mp.cpu_count(), initializer=init_worker, initargs=(df,))

    for i in tqdm(range(num_windows), desc="Processing Windows"):
        window_df = df_sorted.iloc[i:i+window_size]
        window_record_ids = window_df["record_id"].tolist()
        candidate_pairs = list(combinations(window_record_ids, 2))

        # Process candidate pairs in the window
        results = pool.map(compute_similarity_pair, candidate_pairs)

        for record_id1, record_id2, sim_score in results:
            if sim_score >= similarity_threshold:
                union(record_id1, record_id2)

    pool.close()
    pool.join()

    logger.info("Processing of candidate pairs completed.")

    # Add unions based on 'party_iban' and 'party_phone' without generating all combinations
    logger.info("Processing 'party_iban' and 'party_phone' groups for union...")
    for col in ['party_iban', 'party_phone']:
        groups = df.groupby(col)['record_id'].apply(list).to_dict()
        for record_ids in groups.values():
            if len(record_ids) > 1:
                first_id = record_ids[0]
                for other_id in record_ids[1:]:
                    union(first_id, other_id)

    # Assign predicted external IDs based on clusters
    logger.info("Assigning external IDs based on clusters...")
    df["external_id"] = df["record_id"].apply(lambda x: find(x))

    # Save results
    logger.info("Saving results to submission.csv...")
    df[["transaction_reference_id", "external_id"]].to_csv("submission.csv", index=False)

    logger.info(f"Script completed in {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()