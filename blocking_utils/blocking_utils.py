import jellyfish
from datasketch import MinHash, MinHashLSH
import pandas as pd
from nltk.util import ngrams

def create_ngram_lsh(df, colname, n=3, threshold=0.5, num_perm=128):
    # removes spaces and creates ngrams
    # Initialize LSH
    ngram_lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    ngram_minhashes = {}

    def text_to_ngrams(text, n):
        # Convert text to character n-grams
        text = text.lower().replace(' ', '')
        return set(''.join(ng) for ng in ngrams(text, n))

    # Create MinHash for each address using n-grams
    for idx, row in df.iterrows():
        col = row[colname]
        record_id = row['record_id']
        
        # Skip rows with missing values in the specified column
        if pd.isnull(col):
            continue
        
        # Generate n-grams
        col_ngrams = text_to_ngrams(col, n)
        
        # Create MinHash from n-grams
        m = MinHash(num_perm=num_perm)
        for ngram in col_ngrams:
            m.update(ngram.encode('utf8'))
            
        ngram_minhashes[record_id] = m
        ngram_lsh.insert(record_id, m)

    return ngram_lsh, ngram_minhashes

def create_ngram_words_lsh(df, colname, n=3, threshold=0.5, num_perm=128):
    # splits by words then does ngrams on words
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

    name_lsh = MinHashLSH(threshold=0.25, num_perm=128)
    name_minhashes = {}
    for idx, row in df.iterrows():
        record_id = row["record_id"]
        name = row[colname]
        minhash = get_minhash(name)
        name_minhashes[record_id] = minhash
        name_lsh.insert(record_id, minhash)

def compute_similarity(row1, row2):
    # Initialize similarity score
    similarity_score = 0.0
    total_weight = 0.0

    # Define weights for each feature
    weights = {
        "is_company": 2.0,
        "parsed_name": 3.0,
        "name_phonetic": 2.0,
        "surname": 3.0,
        "surname_phonetic": 2.0,
        "given_name": 1.0,
        "surname_length": 0.5,
        "party_iban": 5.0,
        "party_phone": 1.0,
    }

    # 1. Compare 'is_company'
    if not pd.isnull(row1["is_company"]) and not pd.isnull(row2["is_company"]):
        if row1["is_company"] == row2["is_company"]:
            similarity_score += weights["is_company"]
        total_weight += weights["is_company"]

    # 2. Compare 'parsed_name' using Jaro-Winkler similarity
    if not pd.isnull(row1["parsed_name"]) and not pd.isnull(row2["parsed_name"]):
        name_similarity = jellyfish.jaro_winkler_similarity(
            row1["parsed_name"], row2["parsed_name"]
        )
        similarity_score += name_similarity * weights["parsed_name"]
        total_weight += weights["parsed_name"]

    # 3. Compare name phonetic encodings
    phonetic_matches = 0
    phonetic_total = 0

    for encoding in ["name_soundex", "name_metaphone", "name_nysiis"]:
        if encoding in row1 and encoding in row2:
            if not pd.isnull(row1[encoding]) and not pd.isnull(row2[encoding]):
                phonetic_total += 1
                if row1[encoding] == row2[encoding]:
                    phonetic_matches += 1

    if phonetic_total > 0:
        phonetic_similarity = phonetic_matches / phonetic_total
        similarity_score += phonetic_similarity * weights["name_phonetic"]
        total_weight += weights["name_phonetic"]

    # 4. Compare 'surname' using Jaro-Winkler similarity
    if not pd.isnull(row1["surname"]) and not pd.isnull(row2["surname"]):
        surname_similarity = jellyfish.jaro_winkler_similarity(
            row1["surname"], row2["surname"]
        )
        similarity_score += surname_similarity * weights["surname"]
        total_weight += weights["surname"]

    # 5. Compare surname phonetic encodings
    surname_phonetic_matches = 0
    surname_phonetic_total = 0

    for encoding in ["surname_soundex", "surname_metaphone", "surname_nysiis"]:
        if encoding in row1 and encoding in row2:
            if not pd.isnull(row1[encoding]) and not pd.isnull(row2[encoding]):
                surname_phonetic_total += 1
                if row1[encoding] == row2[encoding]:
                    surname_phonetic_matches += 1

    if surname_phonetic_total > 0:
        surname_phonetic_similarity = surname_phonetic_matches / surname_phonetic_total
        similarity_score += surname_phonetic_similarity * weights["surname_phonetic"]
        total_weight += weights["surname_phonetic"]

    # 6. Compare 'given_name' using Jaro-Winkler similarity
    if not pd.isnull(row1["given_name"]) and not pd.isnull(row2["given_name"]):
        given_name_similarity = jellyfish.jaro_winkler_similarity(
            row1["given_name"], row2["given_name"]
        )
        similarity_score += given_name_similarity * weights["given_name"]
        total_weight += weights["given_name"]

    # 7. Compare 'surname_length'
    if not pd.isnull(row1["surname_length"]) and not pd.isnull(row2["surname_length"]):
        length_difference = abs(row1["surname_length"] - row2["surname_length"])
        max_length = max(row1["surname_length"], row2["surname_length"])
        if max_length > 0:
            length_similarity = 1 - (length_difference / max_length)
            similarity_score += length_similarity * weights["surname_length"]
            total_weight += weights["surname_length"]

    # 8. Compare 'party_iban' if available
    if "party_iban" in row1 and "party_iban" in row2:
        if not pd.isnull(row1["party_iban"]) and not pd.isnull(row2["party_iban"]):
            if row1["party_iban"] == row2["party_iban"]:
                similarity_score += weights["party_iban"]
            total_weight += weights["party_iban"]

    # 9. Compare 'party_phone' if available
    if "party_phone" in row1 and "party_phone" in row2:
        if not pd.isnull(row1["party_phone"]) and not pd.isnull(row2["party_phone"]):
            phone_similarity = jellyfish.jaro_winkler_similarity(
                row1["party_phone"], row2["party_phone"]
            )
            similarity_score += phone_similarity * weights["party_phone"]
            total_weight += weights["party_phone"]

    # Handle case where total_weight is zero to avoid division by zero
    if total_weight == 0:
        return 0.0

    # Calculate final similarity score as a percentage
    final_similarity = (similarity_score / total_weight)

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
