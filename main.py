import pandas as pd
from itertools import combinations
from collections import defaultdict
from blocking_utils.blocking_utils import compute_similarity, create_ngram_lsh


def main():
    df = pd.read_csv("data/processed/external_parties_train.csv")
    df["record_id"] = df.index

    # MinHash for name similarity
    name_lsh, name_minhashes = create_ngram_lsh(df, 'parsed_name', n=2, threshold=0.25, num_perm=128)

    def evaluate_lsh_groups(df, name_lsh, name_minhashes):
        # Get LSH groups
        lsh_groups = {}
        for idx, row in df.iterrows():
            record_id = row["record_id"]
            neighbors = name_lsh.query(name_minhashes[record_id])
            # Sort to ensure consistent group identification
            sorted_neighbors = tuple(sorted(neighbors))
            lsh_groups[record_id] = sorted_neighbors

        # Evaluate against true external_ids
        correct_pairs = 0
        total_predicted_pairs = 0
        total_actual_pairs = 0

        # Count actual pairs
        external_id_groups = df.groupby("external_id").record_id.agg(list).to_dict()
        for group in external_id_groups.values():
            total_actual_pairs += len(group) * (len(group) - 1) // 2

        # Count correct and predicted pairs
        for record_id, group in lsh_groups.items():
            group_size = len(group)
            total_predicted_pairs += group_size * (group_size - 1) // 2

            # Get actual external_id for this record
            true_external_id = df.loc[record_id, "external_id"]

            # Count correct pairs in this group
            for other_id in group:
                if other_id != record_id:
                    other_external_id = df.loc[other_id, "external_id"]
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

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "correct_pairs": correct_pairs,
            "predicted_pairs": total_predicted_pairs,
            "actual_pairs": total_actual_pairs,
        }

    # Use it like this:
    metrics = evaluate_lsh_groups(df, name_lsh, name_minhashes)
    print(metrics)

    def count_lsh_blocks(name_lsh, name_minhashes):
        # Get all unique groups
        blocks = set()
        for record_id, minhash in name_minhashes.items():
            neighbors = name_lsh.query(minhash)
            # Convert to frozen set for hashability
            block = frozenset(neighbors)
            blocks.add(block)

        # Print statistics
        block_sizes = [len(block) for block in blocks]
        print(f"Number of blocks: {len(blocks)}")
        print(f"Average block size: {sum(block_sizes)/len(blocks):.2f}")
        print(f"Max block size: {max(block_sizes)}")
        print(f"Min block size: {min(block_sizes)}")
        print(f"Blocks of size 1: {sum(1 for x in block_sizes if x == 1)}")

        return blocks

    # Use after creating LSH:
    blocks = count_lsh_blocks(name_lsh, name_minhashes)

    def analyze_record_block_membership(name_lsh, name_minhashes):
        # Track which blocks each record appears in
        record_to_blocks = {}

        for record_id, minhash in name_minhashes.items():
            neighbors = name_lsh.query(minhash)
            # Convert each group to frozenset for hashability
            blocks = frozenset(neighbors)
            record_to_blocks[record_id] = blocks

        # Analyze membership
        membership_counts = [len(blocks) for blocks in record_to_blocks.values()]

        print(
            f"Records appearing in multiple blocks: {sum(1 for x in membership_counts if x > 1)}"
        )
        print(
            f"Average blocks per record: {sum(membership_counts)/len(membership_counts):.2f}"
        )
        print(f"Max blocks per record: {max(membership_counts)}")

        # Show example of a record in multiple blocks
        for record_id, blocks in record_to_blocks.items():
            if len(blocks) > 1:
                print(
                    f"\nExample - Record {record_id} appears in {len(blocks)} blocks:"
                )
                print(f"Name: {df.loc[record_id, 'parsed_name']}")
                break

        return record_to_blocks

    # Use after creating LSH:
    record_memberships = analyze_record_block_membership(name_lsh, name_minhashes)

    # df["`address_simhash`"] = df["full_address"].apply(get_simhash)

    ################ PAIRING STRATEGY ################

    # Create a mapping from composite keys to record IDs:
    # composite_key = (minhash_bucket, simhash_value)
    # This might be too strict, consider finding neighbors over different LSHashes independently
    composite_key_to_records = defaultdict(set)

    # for record_id in df['record_id']:
    #     key = create_composite_key(record_id, name_lsh, name_minhashes, df)
    #     composite_key_to_records[key].add(record_id)

    # Just use the minhash for now
    for record_id in df["record_id"]:
        minhash = name_minhashes[record_id]
        key = frozenset(name_lsh.query(minhash))
        composite_key_to_records[key].add(record_id)

    # Generate candidate pairs within each composite bucket
    candidate_pairs = set()

    for records in composite_key_to_records.values():
        if len(records) > 1:
            for pair in combinations(records, 2):
                candidate_pairs.add(tuple(sorted(pair)))
    print(f"Candidate pairs: {len(candidate_pairs)}")

    # Set a similarity threshold
    similarity_threshold = 0.4

    # List to store matched pairs
    matched_pairs = []

    for pair in candidate_pairs:
        first = df.loc[df["record_id"] == pair[0]].iloc[0]
        second = df.loc[df["record_id"] == pair[1]].iloc[0]
        sim_score = compute_similarity(first, second)
        if sim_score >= similarity_threshold:
            matched_pairs.append(pair)
    print(f"Matched pairs: {len(matched_pairs)}")

    ###############################################

    # Union-Find implementation
    parent = {}

    def find(u):
        if parent[u] != u:
            parent[u] = find(parent[u])
        return parent[u]

    def union(u, v):
        pu, pv = find(u), find(v)
        if pu != pv:
            parent[pu] = pv

    # Initialize parent pointers
    for record_id in df["record_id"]:
        parent[record_id] = record_id

    # Union matched pairs
    for u, v in matched_pairs:
        union(u, v)

    # Generate clusters
    clusters = defaultdict(set)
    for record_id in df["record_id"]:
        cluster_id = find(record_id)
        clusters[cluster_id].add(record_id)

    ################ EVALUATION ################

    # Ground truth clusters based on 'external_id'
    ground_truth = defaultdict(set)
    for idx, row in df.iterrows():
        external_id = row["external_id"]
        record_id = row["record_id"]
        ground_truth[external_id].add(record_id)

    # Predicted clusters are stored in 'clusters'
    def get_all_pairs(cluster_dict):
        pairs = set()
        for records in cluster_dict.values():
            if len(records) > 1:
                for pair in combinations(records, 2):
                    pairs.add(tuple(sorted(pair)))
        return pairs

    # Get true pairs and predicted pairs
    true_pairs = get_all_pairs(ground_truth)
    predicted_pairs = get_all_pairs(clusters)

    # Compute True Positives (TP), False Positives (FP), and False Negatives (FN)
    TP = len(
        true_pairs.intersection(predicted_pairs)
    )  # (A, B) match and we predicted (A, B)
    FP = len(
        predicted_pairs.difference(true_pairs)
    )  # (A, C) dont match but we predicted (A, C)
    FN = len(
        true_pairs.difference(predicted_pairs)
    )  # (A, B) match but we did not predict (A, B)

    # Precision, Recall, F1-Score
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = (
        2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    )

    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-Score: {f1_score:.2f}")

    """
    # jaccard sim, cosine or distance metrics that take into account missing dimensions, GPS
    1.	MinHash for Jaccard Similarity:
        •	Use Case: Suitable for sets or binary vectors (e.g., name tokens, address tokens).
        •	Implementation:
        •	Represent names and addresses as sets of tokens or n-grams.
        •	Apply MinHash to generate hash signatures.
        •	Use multiple hash functions to create a composite signature.
    2.	SimHash for Cosine Similarity:
        •	Use Case: Works well with high-dimensional vectors from textual data.
        •	Implementation:
        •	Vectorize textual features using TF-IDF.
        •	Compute weighted sums of random hyperplanes.
        •	Generate binary hash codes based on the sign of the projection.
	3.	LSH for Euclidean Distance (P-Stable Distributions):
        •	Use Case: Suitable for numerical features and embeddings.
        •	Implementation:
        •	Use numerical features like IBAN and phone numbers.
    	•	Project data onto random vectors and hash based on quantized projections.
	4.	N-Gram LSH:
        •	Use Case: Effective for strings with minor differences.
        •	Implementation:
        •	Generate character n-grams from names and addresses.
        •	Apply LSH to n-gram frequency vectors.
    """


if __name__ == "__main__":
    main()
