import pandas as pd
from itertools import combinations
from collections import defaultdict
from blocking_utils.blocking_utils import compute_similarity
import logging
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

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
            logger.error(
                f"Error computing similarity for pair ({record_id1}, {record_id2}): {e}"
            )
    return results


def evaluate_clusters(
    df, predicted_col="predicted_external_id", true_col="external_id"
):
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
        2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    )

    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1-Score: {f1_score:.4f}")
    logger.info(f"Evaluation completed in {time.time() - start_time:.2f} seconds.")


def main():
    logger.info("Loading data...")
    start_time = time.time()
    df = pd.read_csv("data/processed/external_parties_train.csv")

    # Assign unique record IDs
    df["record_id"] = df.index

    # Keep original external_id as ground truth
    df["true_external_id"] = df["external_id"]

    # Specify columns for sorting key
    key_columns = [
        "parsed_name",
        "parsed_address_street_name",
        "parsed_address_postal_code",
    ]

    # Generate sorting keys
    logger.info("Generating sorting keys...")
    df["sorting_key"] = df.apply(
        lambda row: generate_sorting_key(row, key_columns), axis=1
    )

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
        for i in tqdm(
            range(total_records - window_size + 1), desc="Processing Windows"
        ):
            window_df = df_sorted.iloc[i : i + window_size]
            window_record_ids = window_df["record_id"].tolist()
            candidate_pairs = list(combinations(window_record_ids, 2))

            # Process candidate pairs in parallel
            future = executor.submit(
                compute_similarity_pairs, candidate_pairs, df, similarity_threshold
            )
            results = future.result()

            for record_id1, record_id2 in results:
                union(record_id1, record_id2)

    logger.info("Processing of candidate pairs completed.")

    # Add unions based on 'party_iban' and 'party_phone' without generating all combinations
    logger.info("Processing 'party_iban' and 'party_phone' groups for union...")
    df.reset_index(inplace=True)
    for col in ["party_iban", "party_phone"]:
        groups = df.groupby(col)["record_id"].apply(list).to_dict()
        for record_ids in groups.values():
            if len(record_ids) > 1:
                first_id = record_ids[0]
                for other_id in record_ids[1:]:
                    union(first_id, other_id)

    # Assign predicted external IDs based on clusters
    logger.info("Assigning external IDs based on clusters...")
    df["predicted_external_id"] = df.index.map(lambda x: find(x))

    # Save results
    logger.info("Saving results to submission.csv...")
    df[["transaction_reference_id", "predicted_external_id"]].to_csv(
        "submission.csv", index=False
    )

    # Step 6: Evaluate clustering
    evaluate_clusters(
        df, predicted_col="predicted_external_id", true_col="true_external_id"
    )

    logger.info(f"Script completed in {time.time() - start_time:.2f} seconds.")


if __name__ == "__main__":
    main()
