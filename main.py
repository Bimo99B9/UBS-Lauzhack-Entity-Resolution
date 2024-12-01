import pandas as pd
from itertools import combinations
from collections import defaultdict
from blocking_utils.blocking_utils import compute_similarity
from datasketch import MinHash, MinHashLSH
from nltk.util import ngrams
import multiprocessing as mp
import logging
import time
import numpy as np

# Global variable for DataFrame
df = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def analyze_record_block_membership(df_local, name_lsh, name_minhashes):
    logger.info("Analyzing record block membership...")
    start_time = time.time()

    # Track which blocks each record appears in
    record_to_blocks = {}

    for record_id, minhash in name_minhashes.items():
        neighbors = name_lsh.query(minhash)
        blocks = frozenset(neighbors)
        record_to_blocks[record_id] = blocks

    # Analyze membership
    membership_counts = [len(blocks) for blocks in record_to_blocks.values()]

    logger.info(
        f"Records appearing in multiple blocks: {sum(1 for x in membership_counts if x > 1)}"
    )
    logger.info(
        f"Average blocks per record: {sum(membership_counts)/len(membership_counts):.4f}"
    )
    logger.info(f"Max blocks per record: {max(membership_counts)}")

    # Show example of a record in multiple blocks
    for record_id, blocks in record_to_blocks.items():
        if len(blocks) > 1:
            logger.info(
                f"\nExample - Record {record_id} appears in {len(blocks)} blocks:"
            )
            logger.info(f"Name: {df_local.loc[record_id, 'parsed_name']}")
            break

    logger.info(
        f"Record block membership analysis completed in {time.time() - start_time:.2f} seconds."
    )
    return record_to_blocks


def evaluate_lsh_groups(df_local, name_lsh, name_minhashes):
    logger.info("Evaluating LSH groups...")
    start_time = time.time()

    # Get LSH groups
    lsh_groups = {}
    for idx, row in df_local.iterrows():
        record_id = row["record_id"]
        if record_id not in name_minhashes:
            continue
        neighbors = name_lsh.query(name_minhashes[record_id])
        sorted_neighbors = tuple(sorted(neighbors))
        lsh_groups[record_id] = sorted_neighbors

    # Evaluate against true external_ids
    correct_pairs = 0
    total_predicted_pairs = 0
    total_actual_pairs = 0

    # Count actual pairs
    external_id_groups = (
        df_local.groupby("external_id")["record_id"].apply(list).to_dict()
    )
    for group in external_id_groups.values():
        total_actual_pairs += len(group) * (len(group) - 1) // 2

    # Count correct and predicted pairs
    for record_id, group in lsh_groups.items():
        group_size = len(group)
        total_predicted_pairs += group_size * (group_size - 1) // 2

        true_external_id = df_local.at[record_id, "external_id"]

        for other_id in group:
            if other_id != record_id:
                other_external_id = df_local.at[other_id, "external_id"]
                if true_external_id == other_external_id:
                    correct_pairs += 1

    correct_pairs = correct_pairs // 2  # Each pair was counted twice

    precision = (
        correct_pairs / total_predicted_pairs if total_predicted_pairs > 0 else 0
    )
    recall = correct_pairs / total_actual_pairs if total_actual_pairs > 0 else 0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    metrics = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "correct_pairs": correct_pairs,
        "predicted_pairs": total_predicted_pairs,
        "actual_pairs": total_actual_pairs,
    }

    logger.info(
        f"LSH groups evaluation completed in {time.time() - start_time:.2f} seconds."
    )
    logger.info(f"Metrics: {metrics}")
    return metrics


def count_lsh_blocks(name_lsh, name_minhashes):
    logger.info("Counting LSH blocks...")
    start_time = time.time()

    blocks = set()
    for record_id, minhash in name_minhashes.items():
        neighbors = name_lsh.query(minhash)
        block = frozenset(neighbors)
        blocks.add(block)

    # Print statistics
    block_sizes = [len(block) for block in blocks]
    logger.info(f"Number of blocks: {len(blocks)}")
    logger.info(f"Average block size: {sum(block_sizes)/len(block_sizes):.4f}")
    logger.info(f"Max block size: {max(block_sizes)}")
    logger.info(f"Min block size: {min(block_sizes)}")
    logger.info(f"Blocks of size 1: {sum(1 for x in block_sizes if x == 1)}")

    logger.info(
        f"LSH block counting completed in {time.time() - start_time:.2f} seconds."
    )
    return blocks


def get_all_pairs(cluster_dict):
    pairs = set()
    for records in cluster_dict.values():
        if len(records) > 1:
            pairs.update(combinations(sorted(records), 2))
    return pairs


def compute_similarity_pair(pair):
    """
    Computes the similarity score for a given pair of record IDs.
    Accesses the global DataFrame 'df'.
    """
    record_id1, record_id2 = pair
    # Access the global DataFrame
    global df
    try:
        row1 = df.loc[record_id1]
        row2 = df.loc[record_id2]
        sim_score = compute_similarity(row1, row2)
        return (pair, sim_score)
    except Exception as e:
        logger.error(f"Error computing similarity for pair {pair}: {e}")
        return (pair, 0.0)


def create_ngram_lsh_parallel(
    df_local, colname, n=3, threshold=0.5, num_perm=128, num_processes=4
):
    logger.info(f"Creating parallel LSH for column: {colname}")
    start_time = time.time()

    # Initialize LSH
    ngram_lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    ngram_minhashes = {}

    # Split DataFrame into chunks
    num_chunks = num_processes
    chunks = np.array_split(df_local, num_chunks)

    # Prepare arguments for each chunk
    args = [(chunk, colname, n, num_perm) for chunk in chunks]

    # Use Pool to compute MinHashes in parallel
    with mp.Pool(processes=num_processes) as pool:
        results = pool.map(compute_minhash_chunk, args)

    # Collect MinHashes and insert into LSH
    for partial_minhashes in results:
        for record_id, m in partial_minhashes.items():
            ngram_minhashes[record_id] = m
            ngram_lsh.insert(record_id, m)

    logger.info(
        f"Parallel LSH creation for column {colname} completed in {time.time() - start_time:.2f} seconds."
    )

    # Log the number of blocks and size distribution
    count_lsh_blocks(ngram_lsh, ngram_minhashes)

    return ngram_lsh, ngram_minhashes


def compute_minhash_chunk(args):
    """
    Computes MinHash signatures for a chunk of the DataFrame.
    """
    chunk, colname, n, num_perm = args
    from nltk.util import ngrams
    from datasketch import MinHash
    import pandas as pd

    def text_to_ngrams(text, n):
        text = text.lower().replace(" ", "")
        return set("".join(ng) for ng in ngrams(text, n))

    partial_minhashes = {}
    for idx, row in chunk.iterrows():
        col = row[colname]
        record_id = row["record_id"]

        if pd.isnull(col):
            continue

        col_ngrams = text_to_ngrams(col, n)

        m = MinHash(num_perm=num_perm)
        for ngram in col_ngrams:
            m.update(ngram.encode("utf8"))

        partial_minhashes[record_id] = m
    return partial_minhashes


def init_worker(dataframe):
    """
    Initializer for worker processes to set the global DataFrame.
    """
    global df
    df = dataframe


def main():
    global df
    start_total = time.time()
    df = pd.read_csv("data/processed/external_parties_train.csv")
    # cols = ["parsed_name", "parsed_address_street_name", "parsed_address_city"]
    cols = ["parsed_name"]
    thres = [0.6, 0.8, 0.8]
    ngram = [3, 3, 3]
    num_perm = 32
    num_processes = mp.cpu_count()

    logger.info("Initializing record IDs...")
    df["record_id"] = df.index

    # Initialize dictionaries to hold LSH and MinHashes for each column
    lsh_dict = {}
    minhash_dict = {}

    # Create LSH for each column, processing rows in parallel
    for col, th, n in zip(cols, thres, ngram):
        logger.info(f"Processing column: {col} with threshold: {th} and n-gram size: {n}")
        lsh, minhash = create_ngram_lsh_parallel(
            df, col, n=n, threshold=th, num_perm=num_perm, num_processes=num_processes
        )
        lsh_dict[col] = lsh
        minhash_dict[col] = minhash
        logger.info(f"Completed LSH creation for column: {col}")

    # Evaluate LSH groups
    for col in cols:
        logger.info(f"Evaluating LSH groups for column: {col}")
        metrics = evaluate_lsh_groups(df, lsh_dict[col], minhash_dict[col])
        logger.info(f"{col}: {metrics}")

    ################ IMPROVED PAIRING STRATEGY ################
    MAX_BLOCK_SIZE = 1000  # Limit for block size to prevent explosion
    SECONDS_TO_WAIT = 10  # Time to wait between attempts to split a block further

    seen_pairs = set()  # Track already-seen pairs

    logger.info("Starting improved pairing strategy...")
    start_pairing = time.time()

    composite_key_to_records = defaultdict(set)

    # Group records into composite blocks
    for record_id in df["record_id"]:
        for col in cols:
            if record_id in minhash_dict[col]:
                minhash = minhash_dict[col][record_id]
                neighbors = lsh_dict[col].query(minhash)
                key = frozenset(neighbors)
                composite_key_to_records[key].add(record_id)

    candidate_pairs = set()

    # Function to split a large block using a secondary blocking key
    def split_block(records, secondary_col):
        sub_blocks = defaultdict(set)
        for record_id in records:
            secondary_value = df.at[record_id, secondary_col]
            sub_blocks[secondary_value].add(record_id)
        return sub_blocks.values()

    # Generate candidate pairs with constraints
    for records in composite_key_to_records.values():
        block_size = len(records)
        if block_size <= 1:
            continue
        if block_size <= MAX_BLOCK_SIZE:
            # Generate pairs directly
            for pair in combinations(sorted(records), 2):
                if pair not in seen_pairs:
                    candidate_pairs.add(pair)
                    seen_pairs.add(pair)
        else:
            # Split the block using a secondary blocking key
            logger.info(f"Block size {block_size} exceeds limit. Splitting the block.")
            # Choose a secondary column for splitting; for example, 'party_phone' or 'party_iban'
            secondary_cols = ["party_phone", "party_iban"]
            split_success = False
            for secondary_col in secondary_cols:
                if secondary_col in df.columns:
                    sub_blocks = split_block(records, secondary_col)
                    for sub_block in sub_blocks:
                        sub_block_size = len(sub_block)
                        if sub_block_size <= MAX_BLOCK_SIZE and sub_block_size > 1:
                            for pair in combinations(sorted(sub_block), 2):
                                if pair not in seen_pairs:
                                    candidate_pairs.add(pair)
                                    seen_pairs.add(pair)
                    split_success = True
                    break  # Stop after successful split
            if not split_success:
                # If no suitable secondary column is found, apply further n-gram blocking or skip
                logger.warning(
                    f"Unable to split block of size {block_size} using secondary columns. Skipping block."
                )
                continue

    logger.info(f"Generated {len(candidate_pairs)} unique candidate pairs.")
    logger.info(f"Pairing strategy completed in {time.time() - start_pairing:.2f} seconds.")
    #############################################################

    # Set a similarity threshold
    similarity_threshold = 0.6

    # Lists to store matched pairs and their similarity scores
    matched_pairs = set()
    unmatched_candidate_pairs = []

    logger.info("Computing similarities for candidate pairs...")
    start_similarity = time.time()

    # Use multiprocessing to compute similarities
    with mp.Pool(
        processes=num_processes, initializer=init_worker, initargs=(df,)
    ) as pool:
        results = pool.map(compute_similarity_pair, candidate_pairs)

    candidate_pairs_similarity = {}
    for pair, sim_score in results:
        candidate_pairs_similarity[pair] = sim_score
        if sim_score >= similarity_threshold:
            matched_pairs.add(pair)
        else:
            unmatched_candidate_pairs.append(pair)

    logger.info(
        f"Similarity computation completed in {time.time() - start_similarity:.2f} seconds."
    )
    logger.info(f"Matched pairs after similarity threshold: {len(matched_pairs)}")

    # Additional pairing based on 'party_iban' and 'party_phone'
    logger.info("Adding pairs based on 'party_iban' and 'party_phone'...")
    party_iban_to_record_ids = (
        df.groupby("party_iban")["record_id"].apply(list).to_dict()
    )
    party_phone_to_record_ids = (
        df.groupby("party_phone")["record_id"].apply(list).to_dict()
    )

    for record_ids in party_iban_to_record_ids.values():
        if len(record_ids) > 1:
            for pair in combinations(sorted(record_ids), 2):
                if pair not in matched_pairs:
                    matched_pairs.add(pair)
    for record_ids in party_phone_to_record_ids.values():
        if len(record_ids) > 1:
            for pair in combinations(sorted(record_ids), 2):
                if pair not in matched_pairs:
                    matched_pairs.add(pair)

    logger.info(
        f"Total matched pairs after adding IBAN and phone: {len(matched_pairs)}"
    )
    logger.info(
        f"Pairing strategy completed in {time.time() - start_pairing:.2f} seconds."
    )
    ###############################################################

    # Union-Find implementation
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

    # Create predicted_external_id column
    df["predicted_external_id"] = df["record_id"].apply(lambda x: find(x))
    logger.info(
        f"Clustering completed in {time.time() - start_union_find:.2f} seconds."
    )

    ################ EVALUATION ################
    logger.info("Starting evaluation of clusters...")
    start_evaluation = time.time()

    # Ground truth clusters based on 'external_id'
    ground_truth = df.groupby("external_id")["record_id"].apply(set).to_dict()

    # Get true pairs and predicted pairs
    true_pairs = get_all_pairs(ground_truth)
    predicted_pairs = get_all_pairs(clusters)

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

    metrics = {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "true_positives": TP,
        "false_positives": FP,
        "false_negatives": FN,
    }

    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1-Score: {f1_score:.4f}")
    logger.info(
        f"Evaluation completed in {time.time() - start_evaluation:.2f} seconds."
    )

    ################ ANALYZE MISSED PAIRS ################
    logger.info("Analyzing missed pairs (False Negatives)...")
    start_missed = time.time()

    # Identify False Negative pairs (missed pairs)
    fn_pairs = true_pairs.difference(predicted_pairs)

    missed_pairs_info = []

    for pair in fn_pairs:
        record_id1, record_id2 = pair
        try:
            record1 = df.loc[df["record_id"] == record_id1].iloc[0]
            record2 = df.loc[df["record_id"] == record_id2].iloc[0]
        except IndexError:
            logger.error(f"Record ID not found for pair: {pair}")
            continue

        # Check if the pair was a candidate pair
        if pair in candidate_pairs or (pair[::-1] in candidate_pairs):
            # They were compared but similarity score was below threshold
            sim_score = candidate_pairs_similarity.get(
                pair
            ) or candidate_pairs_similarity.get(pair[::-1])
            reason = (
                f"Low similarity score ({sim_score:.4f})"
                if sim_score is not None
                else "Low similarity score"
            )
        else:
            # They were not compared; likely in different blocks
            sim_score = None
            reason = "Different blocks (not compared)"

        # Collect information for exporting
        missed_pairs_info.append(
            {
                "record_id_1": record_id1,
                "parsed_name_1": record1["parsed_name"],
                "external_id_1": record1["external_id"],
                "record_id_2": record_id2,
                "parsed_name_2": record2["parsed_name"],
                "external_id_2": record2["external_id"],
                "similarity_score": sim_score,
                "reason": reason,
            }
        )

    # Create a DataFrame from the missed pairs information
    missed_pairs_df = pd.DataFrame(missed_pairs_info)

    # Save to CSV file
    missed_pairs_df.to_csv("missed_pairs_analysis.csv", index=False)

    logger.info(f"Number of missed pairs: {len(missed_pairs_df)}")
    logger.info(
        "Missed pairs with explanations have been saved to 'missed_pairs_analysis.csv'"
    )
    logger.info(
        f"Missed pairs analysis completed in {time.time() - start_missed:.2f} seconds."
    )

    ################ OPTIONAL: ANALYSIS ################
    logger.info("Analyzing reasons for missing pairs...")
    reasons_counts = missed_pairs_df["reason"].value_counts()
    # logger.info("\nReasons for missing pairs:")
    # logger.info(reasons_counts.to_string())

    logger.info(f"Total script completed in {time.time() - start_total:.2f} seconds.")


if __name__ == "__main__":
    main()
