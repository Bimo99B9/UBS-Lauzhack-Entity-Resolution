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
from tqdm import tqdm
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

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

    # Calculate block sizes
    block_sizes = [len(block) for block in blocks]

    # Print statistics
    logger.info(f"Number of blocks: {len(blocks)}")
    logger.info(f"Average block size: {sum(block_sizes) / len(block_sizes):.4f}")
    logger.info(f"Max block size: {max(block_sizes)}")
    logger.info(f"Min block size: {min(block_sizes)}")
    logger.info(f"Blocks of size 1: {sum(1 for x in block_sizes if x == 1)}")
    logger.info(f"Median block size: {np.median(block_sizes):.2f}")
    logger.info(f"90th Percentile block size: {np.percentile(block_sizes, 90):.2f}")

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
        results = list(tqdm(pool.imap(compute_minhash_chunk, args), total=len(args), desc="Processing Chunks"))

        results = pool.map(compute_minhash_chunk, args)
    with ngram_lsh.insertion_session() as session:
    # Collect MinHashes and insert into LSH
        for partial_minhashes in results:
            for record_id, m in partial_minhashes.items():
                ngram_minhashes[record_id] = m
                session.insert(record_id, m)

    logger.info(
        f"Parallel LSH creation for column {colname} completed in {time.time() - start_time:.2f} seconds."
    )
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
    for idx, row in tqdm(chunk.iterrows()):
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


def create_minhash_text_based(text, n, num_perm=128):
    text = text.lower().replace(" ", "")
    col_ngrams = set("".join(ng) for ng in ngrams(text, n))
    
    # col_ngrams = text_to_ngrams(col, n)

    m = MinHash(num_perm=num_perm)
    for ngram in col_ngrams:
        m.update(ngram.encode("utf8"))
    return m


def init_worker(dataframe):
    """
    Initializer for worker processes to set the global DataFrame.
    """
    global df
    df = dataframe

def process_record(record_id, col, lsh_dict, ngram, num_perm):
    """
    Process a single record for LSH and returns composite key and record ID.
    """
    global df
    val = df[col].iloc[record_id]
    if pd.isnull(val):
        return None, None
    try:
        minhash = create_minhash_text_based(val, ngram, num_perm)
        query = lsh_dict[col].query(minhash)
        key = frozenset(query)
        lsh_dict[col].insert(record_id, minhash)
        return key, record_id
    except Exception as e:
        logger.error(f"Error processing record {record_id}: {e}")
        return None, None

def main():
    global df
    start_total = time.time()
    df = pd.read_csv("data/processed/external_parties_train.csv")
    # cols = ["parsed_name", "parsed_address_street_name", "parsed_address_city", "name_soundex", "party_info_unstructured"]
    # thres = [0.4, 0.4, 0.4, 0.4, 0.3]
    # ngram = [3, 4, 4, 3, 3]
    # num_perm = 32
    cols = ["parsed_name", "parsed_address_street_name", "parsed_address_city"]
    thres = [0.6, 0.6, 0.6]
    ngram = [2, 3, 3]
    num_perm = 128
    
    num_processes = 8
    logger.info("Initializing record IDs...")
    df["record_id"] = df.index

    lsh_dict = {col: MinHashLSH(threshold=thres[0], num_perm=num_perm) for col in cols}

    logger.info("Starting pairing strategy with multithreading...")
    start_pairing = time.time()

    composite_key_to_records = defaultdict(set)

    with ThreadPoolExecutor(max_workers=num_processes) as executor:
        futures = [
            executor.submit(process_record, record_id, cols[0], lsh_dict, ngram[0], num_perm)
            for record_id in tqdm(df["record_id"], desc="Submitting Tasks")
        ]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Tasks"):
            key, record_id = future.result()
            if key is not None and record_id is not None:
                composite_key_to_records[key].add(record_id)


    candidate_pairs = set()
    for records in composite_key_to_records.values():
        if 100 > len(records) > 1:
            candidate_pairs.update(combinations(sorted(records), 2))
        elif len(records) > 100:
            random_records = random.sample(sorted(records), 100)
            candidate_pairs.update(combinations(random_records, 2))

    logger.info(f"Generated {len(candidate_pairs)} candidate pairs.")
    similarity_threshold = 0.5
    matched_pairs = set()
    unmatched_candidate_pairs = []

    logger.info("Computing similarities for candidate pairs with multiprocessing...")
    start_similarity = time.time()

    with mp.Pool(processes=num_processes, initializer=init_worker, initargs=(df,)) as pool:
        results = pool.map(compute_similarity_pair, candidate_pairs)

    for pair, sim_score in tqdm(results, desc="Processing Similarities"):
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
            matched_pairs.update(combinations(sorted(record_ids), 2))
    for record_ids in party_phone_to_record_ids.values():
        if len(record_ids) > 1:
            matched_pairs.update(combinations(sorted(record_ids), 2))

    logger.info(
        f"Total matched pairs after adding IBAN and phone: {len(matched_pairs)}"
    )
    logger.info(
        f"Pairing strategy completed in {time.time() - start_pairing:.2f} seconds."
    )
    ###############################################

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
    # df["external_id"] = df["record_id"].apply(lambda x: find(x))
    
    # Save with only "transaction_reference_id" and "external_id" columns
    df[["transaction_reference_id", "external_id"]].to_csv("submission.csv", index=False)
    
    logger.info(
        f"Clustering completed in {time.time() - start_union_find:.2f} seconds."
    )

    logger.info(f"Total script completed in {time.time() - start_total:.2f} seconds.")

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

    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1-Score: {f1_score:.4f}")
    logger.info(
        f"Evaluation completed in {time.time() - start_evaluation:.2f} seconds."
    )

    # ################ ANALYZE MISSED PAIRS ################
    # logger.info("Analyzing missed pairs (False Negatives)...")
    # start_missed = time.time()

    # # Identify False Negative pairs (missed pairs)
    # fn_pairs = true_pairs.difference(predicted_pairs)

    # missed_pairs_info = []

    # for pair in fn_pairs:
    #     record_id1, record_id2 = pair
    #     try:
    #         record1 = df.loc[df["record_id"] == record_id1].iloc[0]
    #         record2 = df.loc[df["record_id"] == record_id2].iloc[0]
    #     except IndexError:
    #         logger.error(f"Record ID not found for pair: {pair}")
    #         continue

    #     # Check if the pair was a candidate pair
    #     if pair in candidate_pairs or (pair[::-1] in candidate_pairs):
    #         # They were compared but similarity score was below threshold
    #         sim_score = candidate_pairs_similarity.get(
    #             pair
    #         ) or candidate_pairs_similarity.get(pair[::-1])
    #         reason = (
    #             f"Low similarity score ({sim_score:.4f})"
    #             if sim_score is not None
    #             else "Low similarity score"
    #         )
    #     else:
    #         # They were not compared; likely in different blocks
    #         sim_score = None
    #         reason = "Different blocks (not compared)"

    #     # Collect information for exporting
    #     missed_pairs_info.append(
    #         {
    #             "record_id_1": record_id1,
    #             "parsed_name_1": record1["parsed_name"],
    #             "external_id_1": record1["external_id"],
    #             "record_id_2": record_id2,
    #             "parsed_name_2": record2["parsed_name"],
    #             "external_id_2": record2["external_id"],
    #             "similarity_score": sim_score,
    #             "reason": reason,
    #         }
    #     )

    # # Create a DataFrame from the missed pairs information
    # missed_pairs_df = pd.DataFrame(missed_pairs_info)

    # # Save to CSV file
    # missed_pairs_df.to_csv("missed_pairs_analysis.csv", index=False)

    # logger.info(f"Number of missed pairs: {len(missed_pairs_df)}")
    # logger.info(
    #     "Missed pairs with explanations have been saved to 'missed_pairs_analysis.csv'"
    # )
    # logger.info(
    #     f"Missed pairs analysis completed in {time.time() - start_missed:.2f} seconds."
    # )

    # ################ OPTIONAL: ANALYSIS ################
    # logger.info("Analyzing reasons for missing pairs...")
    # reasons_counts = missed_pairs_df["reason"].value_counts()
    # # logger.info("\nReasons for missing pairs:")
    # # logger.info(reasons_counts.to_string())

    logger.info(f"Total script completed in {time.time() - start_total:.2f} seconds.")


if __name__ == "__main__":
    main()
