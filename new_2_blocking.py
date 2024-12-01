import pandas as pd
from itertools import combinations
from collections import defaultdict
from blocking_utils.blocking_utils import compute_similarity
import logging
import time
from tqdm import tqdm
import multiprocessing as mp
import random

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

def compute_similarity_pair(pair):
    """
    Computes the similarity score for a given pair of record IDs.
    """
    record_id1, record_id2 = pair
    global df
    try:
        row1 = df.loc[df['record_id'] == record_id1].squeeze()
        row2 = df.loc[df['record_id'] == record_id2].squeeze()
        sim_score = compute_similarity(row1, row2)
        return (pair, sim_score)
    except Exception as e:
        logger.error(f"Error computing similarity for pair {pair}: {e}")
        return (pair, 0.0)

def init_worker(dataframe):
    """
    Initializer for worker processes to set the global DataFrame.
    """
    global df
    df = dataframe

def evaluate_clusters(df, predicted_col='predicted_external_id', true_col='external_id'):
    """
    Evaluate clustering performance based on ground truth.
    """
    logger.info("Starting evaluation of clusters...")
    start_time = time.time()
    
    # Ground truth clusters
    ground_truth = df.groupby(true_col)["record_id"].apply(set).to_dict()
    
    # Predicted clusters
    predicted_clusters = df.groupby(predicted_col)["record_id"].apply(set).to_dict()

    # Get true pairs and predicted pairs
    def get_all_pairs(cluster_dict):
        pairs = set()
        for records in cluster_dict.values():
            if len(records) > 1:
                pairs.update(combinations(sorted(records), 2))
        return pairs

    true_pairs = get_all_pairs(ground_truth)
    predicted_pairs = get_all_pairs(predicted_clusters)

    # Compute True Positives (TP), False Positives (FP), and False Negatives (FN)
    TP = len(true_pairs.intersection(predicted_pairs))
    FP = len(predicted_pairs.difference(true_pairs))
    FN = len(true_pairs.difference(predicted_pairs))

    # Precision, Recall, F1-Score
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1-Score: {f1_score:.4f}")
    logger.info(
        f"Evaluation completed in {time.time() - start_time:.2f} seconds."
    )

def main():
    logger.info("Loading data...")
    start_time = time.time()
    df = pd.read_csv("data/processed/external_parties_train.csv")

    # Assign unique record IDs
    df["record_id"] = df.index

    # Keep original external_id as ground truth
    df['true_external_id'] = df['external_id']
    
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

    # Generate candidate pairs using the sliding window
    logger.info("Generating candidate pairs using the Sorted Neighborhood Method...")
    candidate_pairs = set()
    total_records = len(df_sorted)
    for i in tqdm(range(total_records - window_size + 1), desc="Sliding Window"):
        window_records = df_sorted.iloc[i:i+window_size]["record_id"].tolist()
        pairs_in_window = combinations(window_records, 2)
        candidate_pairs.update(pairs_in_window)

    logger.info(f"Total candidate pairs generated: {len(candidate_pairs)}")

    # Step 3: Compute similarities
    logger.info("Computing similarities for candidate pairs...")
    similarity_threshold = 0.75 # Adjust based on your requirements
    matched_pairs = set()

    with mp.Pool(processes=mp.cpu_count(), initializer=init_worker, initargs=(df,)) as pool:
        results = pool.imap_unordered(compute_similarity_pair, candidate_pairs)
        for pair, sim_score in tqdm(results, total=len(candidate_pairs), desc="Computing Similarities"):
            if sim_score >= similarity_threshold:
                matched_pairs.add(pair)
    
    logger.info(f"Matched pairs after similarity threshold: {len(matched_pairs)}")

    # Step 3.5: Add pairs based on 'party_iban' and 'party_phone'
    logger.info("Adding pairs based on 'party_iban' and 'party_phone'...")
    party_iban_to_record_ids = (
        df.groupby("party_iban")["record_id"].apply(list).to_dict()
    )
    party_phone_to_record_ids = (
        df.groupby("party_phone")["record_id"].apply(list).to_dict()
    )

    iban_pairs = 0
    phone_pairs = 0

    for record_ids in party_iban_to_record_ids.values():
        if len(record_ids) > 1:
            new_pairs = set(combinations(sorted(record_ids), 2))
            matched_pairs.update(new_pairs)
            iban_pairs += len(new_pairs)
    for record_ids in party_phone_to_record_ids.values():
        if len(record_ids) > 1:
            new_pairs = set(combinations(sorted(record_ids), 2))
            matched_pairs.update(new_pairs)
            phone_pairs += len(new_pairs)

    logger.info(f"Added {iban_pairs} pairs based on 'party_iban'.")
    logger.info(f"Added {phone_pairs} pairs based on 'party_phone'.")
    logger.info(f"Total matched pairs after adding IBAN and phone: {len(matched_pairs)}")

    # Step 4: Union-Find for clustering
    logger.info("Clustering using Union-Find...")
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

    for u, v in matched_pairs:
        union(u, v)

    # Generate clusters
    clusters = defaultdict(set)
    for record_id in df["record_id"]:
        cluster_id = find(record_id)
        clusters[cluster_id].add(record_id)

    # Step 5: Assign predicted external IDs based on clusters
    df["predicted_external_id"] = df["record_id"].apply(lambda x: find(x))

    # Save results
    logger.info("Saving results to submission.csv...")
    df[["transaction_reference_id", "predicted_external_id"]].to_csv("submission.csv", index=False)

    # Step 6: Evaluate clustering
    evaluate_clusters(df, predicted_col='predicted_external_id', true_col='true_external_id')

    logger.info(f"Script completed in {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()