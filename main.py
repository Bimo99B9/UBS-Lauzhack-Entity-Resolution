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

    # Create a mapping from composite keys to record IDs
    composite_key_to_records = defaultdict(set)

    # Just use the minhash for now
    for record_id in df["record_id"]:
        minhash = name_minhashes[record_id]
        key = frozenset(name_lsh.query(minhash))
        composite_key_to_records[key].add(record_id)

    # Generate candidate pairs within each composite bucket
    candidate_pairs = set()
    candidate_pairs_similarity = {}

    for records in composite_key_to_records.values():
        if len(records) > 1:
            for pair in combinations(records, 2):
                candidate_pairs.add(tuple(sorted(pair)))

    # Set a similarity threshold
    similarity_threshold = 0.4

    # Lists to store matched pairs and their similarity scores
    matched_pairs = []
    unmatched_candidate_pairs = []

    for pair in candidate_pairs:
        first = df.loc[df["record_id"] == pair[0]].iloc[0]
        second = df.loc[df["record_id"] == pair[1]].iloc[0]
        sim_score = compute_similarity(first, second)
        # Store the similarity score for the candidate pair
        candidate_pairs_similarity[pair] = sim_score
        if sim_score >= similarity_threshold:
            matched_pairs.append(pair)
        else:
            unmatched_candidate_pairs.append(pair)

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
            records = list(records)
            if len(records) > 1:
                for pair in combinations(records, 2):
                    pairs.add(tuple(sorted(pair)))
        return pairs

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

    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-Score: {f1_score:.2f}")

    ################ ANALYZE MISSED PAIRS ################

    # Identify False Negative pairs (missed pairs)
    fn_pairs = true_pairs.difference(predicted_pairs)

    missed_pairs_info = []

    for pair in fn_pairs:
        record_id1, record_id2 = pair
        record1 = df.loc[df["record_id"] == record_id1].iloc[0]
        record2 = df.loc[df["record_id"] == record_id2].iloc[0]

        # Check if the pair was a candidate pair
        if pair in candidate_pairs or (record_id2, record_id1) in candidate_pairs:
            # They were compared but similarity score was below threshold
            sim_score = candidate_pairs_similarity.get(
                pair
            ) or candidate_pairs_similarity.get((record_id2, record_id1))
            reason = f"Low similarity score ({sim_score:.2f})"
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

    print(f"Number of missed pairs: {len(missed_pairs_df)}")
    print(
        "Missed pairs with explanations have been saved to 'missed_pairs_analysis.csv'"
    )

    ################ OPTIONAL: ANALYSIS ################

    # Analyze reasons for missing pairs
    reasons_counts = missed_pairs_df["reason"].value_counts()
    print("\nReasons for missing pairs:")
    print(reasons_counts)

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
