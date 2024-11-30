import pandas as pd
import re
import string


def clean_text(text: str, name: bool) -> str:
    if not isinstance(text, str):
        return text

    # Convert to lowercase
    text = text.lower()
    # Remove punctuation, treating periods specially
    text = re.sub(
        r"[{}]".format(re.escape(string.punctuation)),
        lambda x: "" if x.group() == "." else " ",
        text,
    )
    # Remove special characters except spaces, letters, and periods
    text = re.sub(r"[^ \w\.]", "", text)
    # Handle numbers for names
    if name:
        text = re.sub(r"\w*\d\w*", "", text)  # Remove words with numbers

    # Remove multiple spaces
    text = re.sub(" +", " ", text)

    # Strip leading and trailing spaces
    text = text.strip()

    return text


def clean_text_dataset(df: pd.DataFrame, col_name: str, name: bool) -> pd.DataFrame:
    df[col_name] = df[col_name].apply(
        lambda x: clean_text(x, name) if pd.notna(x) else x
    )
    return df


def remove_words(df: pd.DataFrame, col_name: str, words: list) -> pd.DataFrame:
    for word in words:
        # Remove entire word matches only
        df[col_name] = df[col_name].apply(
            lambda x: re.sub(rf"\b{word}\b", "", x).strip() if isinstance(x, str) else x
        )
    # Remove any extra spaces introduced after processing
    df[col_name] = df[col_name].apply(
        lambda x: re.sub(" +", " ", x).strip() if isinstance(x, str) else x
    )
    return df

# Phone number normalization function
def normalize_phone(phone):
    if pd.isnull(phone):
        return ''
    phone = re.sub(r'[^\d]', '', phone)  # Keep digits only
    return str(phone)


if __name__ == "__main__":
    # Load dataset
    df = pd.read_csv("data/external_parties_train.csv")

    # Clean names
    df = clean_text_dataset(df, "parsed_name", name=True)
    # Clean addresses
    df = clean_text_dataset(df, "parsed_address_street_name", name=False)
    
    # Apply phone number normalization
    df["party_phone"] = df["party_phone"].apply(normalize_phone)

    # Define common words to remove
    remove = [
        "mrs",
        "sr",
        "jr",
        "dr",
        "mr",
        "iii",
        "ii",
        "and",
        "phd",
        "iv",
        "v",
        "md",
        "dds",
        "dvm",
    ]
    df = remove_words(df, "parsed_name", remove)

    # Save cleaned data
    df.to_csv("cleaned.csv", index=False)
