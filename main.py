import jellyfish
import pandas as pd
import numpy as np
from datasketch import MinHash, MinHashLSH
from simhash import Simhash
from itertools import combinations
from collections import defaultdict
from rapidfuzz.distance import JaroWinkler


def get_minhash(text, num_perm=128, ngram_size=3):
    if not text or len(text) < ngram_size:
        return MinHash(num_perm=num_perm)

    word_tokens = set(text.split())

    char_tokens = set(
        text[i : i + ngram_size] for i in range(len(text) - ngram_size + 1)
    )

    all_tokens = word_tokens.union(char_tokens)
    m = MinHash(num_perm=num_perm)  # default seed=1
    for token in all_tokens:
        m.update(token.encode("utf8"))
    return m


def get_simhash(text):
    return Simhash(text)


# Function to compute detailed similarity between two records
# def compute_similarity(df, record_id1, record_id2):
#     row1 = df.loc[df["record_id"] == record_id1].iloc[0]
#     row2 = df.loc[df["record_id"] == record_id2].iloc[0]

#     # Name similarity using Jaro-Winkler
#     name_similarity = JaroWinkler.distance(row1["parsed_name"], row2["parsed_name"])

#     # Address similarity using Jaro-Winkler
#     # address_similarity = JaroWinkler.distance(
#     #     row1["full_address"], row2["full_address"]
#     # )

#     # IBAN similarity (exact match)
#     iban_similarity = 1 if row1["party_iban"] == row2["party_iban"] else 0

#     # Phone similarity (exact match)
#     phone_similarity = 1 if row1["party_phone"] == row2["party_phone"] else 0

#     # Weighted average of similarities
#     # similarities = np.array(
#     #     [name_similarity, address_similarity, iban_similarity, phone_similarity]
#     # )
#     similarities = np.array(
#         [name_similarity, iban_similarity, phone_similarity]
#     )
#     weights = np.array([0.4, 0.3, 0.3])

#     # Adjust weights based on available features
#     available = ~np.isnan(similarities)
#     similarities = similarities[available]
#     weights = weights[available]
#     weights /= weights.sum()

#     overall_similarity = np.dot(similarities, weights)
#     return overall_similarity

def compute_similarity(row1, row2):
    # Initialize similarity score
    similarity_score = 0.0
    total_weight = 0.0

    # Define weights for each feature
    weights = {
        'is_company': 2.0,
        'parsed_name': 3.0,
        'name_phonetic': 2.0,
        'surname': 3.0,
        'surname_phonetic': 2.0,
        'given_name': 1.0,
        'surname_length': 0.5,
        'party_iban': 5.0,
        'party_phone': 1.0
    }

    # 1. Compare 'is_company'
    if not pd.isnull(row1['is_company']) and not pd.isnull(row2['is_company']):
        if row1['is_company'] == row2['is_company']:
            similarity_score += weights['is_company']
        total_weight += weights['is_company']

    # 2. Compare 'parsed_name' using Jaro-Winkler similarity
    if not pd.isnull(row1['parsed_name']) and not pd.isnull(row2['parsed_name']):
        name_similarity = jellyfish.jaro_winkler_similarity(row1['parsed_name'], row2['parsed_name'])
        similarity_score += name_similarity * weights['parsed_name']
        total_weight += weights['parsed_name']

    # 3. Compare name phonetic encodings
    phonetic_matches = 0
    phonetic_total = 0

    for encoding in ['name_soundex', 'name_metaphone', 'name_nysiis']:
        if encoding in row1 and encoding in row2:
            if not pd.isnull(row1[encoding]) and not pd.isnull(row2[encoding]):
                phonetic_total += 1
                if row1[encoding] == row2[encoding]:
                    phonetic_matches += 1

    if phonetic_total > 0:
        phonetic_similarity = (phonetic_matches / phonetic_total)
        similarity_score += phonetic_similarity * weights['name_phonetic']
        total_weight += weights['name_phonetic']

    # 4. Compare 'surname' using Jaro-Winkler similarity
    if not pd.isnull(row1['surname']) and not pd.isnull(row2['surname']):
        surname_similarity = jellyfish.jaro_winkler_similarity(row1['surname'], row2['surname'])
        similarity_score += surname_similarity * weights['surname']
        total_weight += weights['surname']

    # 5. Compare surname phonetic encodings
    surname_phonetic_matches = 0
    surname_phonetic_total = 0

    for encoding in ['surname_soundex', 'surname_metaphone', 'surname_nysiis']:
        if encoding in row1 and encoding in row2:
            if not pd.isnull(row1[encoding]) and not pd.isnull(row2[encoding]):
                surname_phonetic_total += 1
                if row1[encoding] == row2[encoding]:
                    surname_phonetic_matches += 1

    if surname_phonetic_total > 0:
        surname_phonetic_similarity = (surname_phonetic_matches / surname_phonetic_total)
        similarity_score += surname_phonetic_similarity * weights['surname_phonetic']
        total_weight += weights['surname_phonetic']

    # 6. Compare 'given_name' using Jaro-Winkler similarity
    if not pd.isnull(row1['given_name']) and not pd.isnull(row2['given_name']):
        given_name_similarity = jellyfish.jaro_winkler_similarity(row1['given_name'], row2['given_name'])
        similarity_score += given_name_similarity * weights['given_name']
        total_weight += weights['given_name']

    # 7. Compare 'surname_length'
    if not pd.isnull(row1['surname_length']) and not pd.isnull(row2['surname_length']):
        length_difference = abs(row1['surname_length'] - row2['surname_length'])
        max_length = max(row1['surname_length'], row2['surname_length'])
        if max_length > 0:
            length_similarity = 1 - (length_difference / max_length)
            similarity_score += length_similarity * weights['surname_length']
            total_weight += weights['surname_length']

    # 8. Compare 'party_iban' if available
    if 'party_iban' in row1 and 'party_iban' in row2:
        if not pd.isnull(row1['party_iban']) and not pd.isnull(row2['party_iban']):
            if row1['party_iban'] == row2['party_iban']:
                similarity_score += weights['party_iban']
            total_weight += weights['party_iban']

    # 9. Compare 'party_phone' if available
    if 'party_phone' in row1 and 'party_phone' in row2:
        if not pd.isnull(row1['party_phone']) and not pd.isnull(row2['party_phone']):
            phone_similarity = jellyfish.jaro_winkler_similarity(row1['party_phone'], row2['party_phone'])
            similarity_score += phone_similarity * weights['party_phone']
            total_weight += weights['party_phone']

    # Handle case where total_weight is zero to avoid division by zero
    if total_weight == 0:
        return 0.0

    # Calculate final similarity score as a percentage
    final_similarity = (similarity_score / total_weight) * 100

    return final_similarity

def create_composite_key(record_id, name_lsh, name_minhashes, df):
    # Get MinHash signature bucket for name
    name_buckets = name_lsh.query(name_minhashes[record_id])
    name_bucket = frozenset(name_buckets)

    # Get SimHash value for address
    address_simhash = df.loc[df["record_id"] == record_id, "address_simhash"].values[0]
    address_hash = address_simhash.value

    # Combine name bucket and address hash to create composite key
    composite_key = (name_bucket, address_hash)
    return composite_key


def main():
    df = pd.read_csv("data/processed/external_parties_train.csv")

    df["record_id"] = df.index

    # MinHash for name similarity
    name_lsh = MinHashLSH(threshold=0.25, num_perm=128)
    name_minhashes = {}
    for idx, row in df.iterrows():
        record_id = row["record_id"]
        name = row["parsed_name"]
        minhash = get_minhash(name)
        name_minhashes[record_id] = minhash
        name_lsh.insert(record_id, minhash)

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

    # df["address_simhash"] = df["full_address"].apply(get_simhash)

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
    # jaccard distance, cosine or distance metrics that take into account missing dimensions, GPS
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
