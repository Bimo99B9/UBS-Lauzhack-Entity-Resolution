import pandas as pd
from itertools import combinations
from collections import defaultdict
from blocking_utils.blocking_utils import compute_similarity
import multiprocessing as mp
import logging
import time
import numpy as np
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Global variable for DataFrame
df = None


def compute_similarity_pair(pair):
    """
    Computes the similarity score for a given pair of record IDs.
    Accesses the global DataFrame 'df'.
    """
    record_id1, record_id2 = pair
    global df
    try:
        row1 = df.loc[record_id1]
        row2 = df.loc[record_id2]
        sim_score = compute_similarity(row1, row2)
        return (pair, sim_score)
    except Exception as e:
        logger.error(f"Error computing similarity for pair {pair}: {e}")
        return (pair, 0.0)


def generate_adaptive_blocks(df, key_column, prefix_length=5, window_size=5):
    """
    Generates adaptive blocks by first grouping similar records and applying a sliding window.
    """
    logger.info("Generating adaptive blocks...")
    start_time = time.time()

    # Step 1: Group records by a hashed prefix of the key column
    df["prefix"] = df[key_column].str[:prefix_length].str.lower().fillna("").apply(
        lambda x: "".join(sorted(x.replace(" ", "").replace(".", "")))
    )
    groups = df.groupby("prefix")["record_id"].apply(list)

    # Step 2: Apply sliding window within each group
    blocks = defaultdict(set)
    for group_key, record_ids in groups.items():
        if len(record_ids) > 1:
            sorted_ids = sorted(record_ids)
            for i in range(len(sorted_ids) - window_size + 1):
                window = sorted_ids[i : i + window_size]
                block_key = f"{group_key}_{i}"
                blocks[block_key].update(window)

    block_sizes = [len(records) for records in blocks.values()]
    logger.info(f"Number of blocks created: {len(blocks)}")
    logger.info(f"Average block size: {np.mean(block_sizes):.2f}")
    logger.info(f"Max block size: {max(block_sizes)}")
    logger.info(f"Min block size: {min(block_sizes)}")
    logger.info(f"Blocks with size 1: {sum(1 for size in block_sizes if size == 1)}")
    logger.info(f"Adaptive block generation completed in {time.time() - start_time:.2f} seconds.")

    return blocks


def init_worker(dataframe):
    """
    Initializer for worker processes to set the global DataFrame.
    """
    global df
    df = dataframe


def evaluate_clusters(df, predicted_column, ground_truth_column):
    """
    Evaluate clustering performance based on ground truth.
    """
    logger.info("Starting evaluation of clusters...")
    start_time = time.time()

    # Group by the predicted and ground truth columns
    predicted_clusters = df.groupby(predicted_column)["record_id"].apply(set).to_dict()
    ground_truth_clusters = df.groupby(ground_truth_column)["record_id"].apply(set).to_dict()

    # Get true pairs and predicted pairs
    def get_all_pairs(cluster_dict):
        pairs = set()
        for records in cluster_dict.values():
            if len(records) > 1:
                pairs.update(combinations(sorted(records), 2))
        return pairs

    true_pairs = get_all_pairs(ground_truth_clusters)
    predicted_pairs = get_all_pairs(predicted_clusters)

    # Compute True Positives (TP), False Positives (FP), and False Negatives (FN)
    TP = len(true_pairs.intersection(predicted_pairs))
    FP = len(predicted_pairs.difference(true_pairs))
    FN = len(true_pairs.difference(predicted_pairs))

    # Precision, Recall, F1-Score
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = (
        2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    )

    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1-Score: {f1_score:.4f}")
    logger.info(f"Evaluation completed in {time.time() - start_time:.2f} seconds.")


def main():
    global df
    start_total = time.time()
    df = pd.read_csv("data/processed/external_parties_train.csv")

    logger.info("Initializing record IDs...")
    df["record_id"] = df.index

    # Adaptive blocking
    key_column = "parsed_name"
    blocks = generate_adaptive_blocks(df, key_column, prefix_length=5, window_size=10)

    # Generate candidate pairs from blocks
    logger.info("Generating candidate pairs from blocks...")
    candidate_pairs = set()
    for block_records in blocks.values():
        if len(block_records) > 1:
            candidate_pairs.update(combinations(sorted(block_records), 2))

    logger.info(f"Total candidate pairs generated: {len(candidate_pairs)}")

    # Compute similarity for candidate pairs
    similarity_threshold = 0.7
    matched_pairs = set()

    logger.info("Computing similarities for candidate pairs with multiprocessing...")
    start_similarity = time.time()

    with mp.Pool(processes=4, initializer=init_worker, initargs=(df,)) as pool:
        results = pool.imap_unordered(compute_similarity_pair, candidate_pairs)
        for pair, sim_score in tqdm(results, total=len(candidate_pairs), desc="Processing Similarities"):
            if sim_score >= similarity_threshold:
                matched_pairs.add(pair)

    logger.info(f"Similarity computation completed in {time.time() - start_similarity:.2f} seconds.")
    logger.info(f"Matched pairs after similarity threshold: {len(matched_pairs)}")

    # Include pairs based on IBAN and phone
    logger.info("Adding pairs based on 'party_iban' and 'party_phone'...")
    party_iban_to_record_ids = (
        df.groupby("party_iban")["record_id"].apply(list).to_dict()
    )
    party_phone_to_record_ids = (
        df.groupby("party_phone")["record_id"].apply(list).to_dict()
    )

    for record_ids in party_iban_to_record_ids.values():
        if len(record_ids) > 1:
            matched_pairs.update(combinations(sorted(record_ids), 2))
    for record_ids in party_phone_to_record_ids.values():
        if len(record_ids) > 1:
            matched_pairs.update(combinations(sorted(record_ids), 2))

    logger.info(f"Total matched pairs after adding IBAN and phone: {len(matched_pairs)}")

    # Union-Find implementation for clustering
    logger.info("Starting Union-Find for clustering...")
    start_union_find = time.time()
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

    # Assign predicted external IDs to records
    df["predicted_external_id"] = df["record_id"].apply(lambda x: find(x))

    # Save results
    logger.info("Saving results to submission.csv...")
    df[["transaction_reference_id", "predicted_external_id"]].to_csv("submission.csv", index=False)

    logger.info(f"Clustering completed in {time.time() - start_union_find:.2f} seconds.")

    # Evaluate clusters
    evaluate_clusters(df, "predicted_external_id", "external_id")

    logger.info(f"Total script completed in {time.time() - start_total:.2f} seconds.")


if __name__ == "__main__":
    main()
