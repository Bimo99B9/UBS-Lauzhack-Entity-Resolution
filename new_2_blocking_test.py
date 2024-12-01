import pandas as pd
from itertools import combinations
from concurrent.futures import ThreadPoolExecutor
from blocking_utils.blocking_utils import compute_similarity
import logging
import time
from tqdm import tqdm

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

def compute_similarity_pairs(pairs, df, similarity_threshold):
    """
    Compute similarity scores for a list of record ID pairs.
    """
    results = []
    for record_id1, record_id2 in pairs:
        try:
            row1 = df.loc[record_id1]
            row2 = df.loc[record_id2]
            sim_score = compute_similarity(row1, row2)
            if sim_score >= similarity_threshold:
                results.append((record_id1, record_id2))
        except Exception as e:
            logger.error(f"Error computing similarity for pair ({record_id1}, {record_id2}): {e}")
    return results

def main():
    logger.info("Loading data...")
    start_time = time.time()
    df = pd.read_csv("data/processed/external_parties_test.csv")

    # Assign unique record IDs
    df["record_id"] = df.index

    # columns for sorting key
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

    # Define window size and similarity threshold
    window_size = 4
    similarity_threshold = 0.75

    # Initialize Union-Find data structure
    parent = {record_id: record_id for record_id in df["record_id"]}

    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u

    def union(u, v):
        pu, pv = find(u), find(v)
        if pu != pv:
            parent[pu] = pv

    # Process candidate pairs window by window
    logger.info("Processing candidate pairs window by window...")
    total_records = len(df_sorted)
    df.set_index("record_id", inplace=True)

    with ThreadPoolExecutor(max_workers=22) as executor:
        for i in tqdm(range(total_records - window_size + 1), desc="Processing Windows"):
            window_df = df_sorted.iloc[i:i + window_size]
            window_record_ids = window_df["record_id"].tolist()
            candidate_pairs = list(combinations(window_record_ids, 2))

            # Process candidate pairs in parallel
            future = executor.submit(compute_similarity_pairs, candidate_pairs, df, similarity_threshold)
            results = future.result()

            for record_id1, record_id2 in results:
                union(record_id1, record_id2)

    logger.info("Processing of candidate pairs completed.")

    # Add unions based on 'party_iban' and 'party_phone' without generating all combinations
    logger.info("Processing 'party_iban' and 'party_phone' groups for union...")
    df.reset_index(inplace=True)  # Reset the index to restore 'record_id' as a column
    for col in ['party_iban', 'party_phone']:
        if col in df.columns:  # Ensure the column exists
            groups = df.groupby(col)['record_id'].apply(list).to_dict()
            for record_ids in groups.values():
                if len(record_ids) > 1:
                    first_id = record_ids[0]
                    for other_id in record_ids[1:]:
                        union(first_id, other_id)

    # Assign predicted external IDs based on clusters
    logger.info("Assigning external IDs based on clusters...")
    df["external_id"] = df.index.map(lambda x: find(x))

    # Save results
    logger.info("Saving results to submission.csv...")
    df[["transaction_reference_id", "external_id"]].to_csv("submission.csv", index=False)

    logger.info(f"Script completed in {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()
