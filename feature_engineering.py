import pandas as pd
import jellyfish
import re


def soundex(name):
    """
    Computes the Soundex encoding for a given name.
    Returns an empty string if the name is null or empty.
    """
    if pd.isnull(name) or name == "":
        return ""
    return jellyfish.soundex(name)


def metaphone(name):
    """
    Computes the Metaphone encoding for a given name.
    Returns an empty string if the name is null or empty.
    """
    if pd.isnull(name) or name == "":
        return ""
    return jellyfish.metaphone(name)


def nysiis(name):
    """
    Computes the NYSIIS encoding for a given name.
    Returns an empty string if the name is null or empty.
    """
    if pd.isnull(name) or name == "":
        return ""
    return jellyfish.nysiis(name)


def is_company(name):
    """
    Determines if a given name corresponds to a company based on predefined keywords.
    Returns 1 if it's a company, else 0.
    """
    if pd.isnull(name) or name == "":
        return 0  # Assume individual if name is missing
    company_keywords = [
        "ltd",
        "inc",
        "co",
        "corp",
        "llc",
        "plc",
        "limited",
        "incorporated",
        "company",
        "corporation",
        "gmbh",
        "kg",
        "llp",
        "pte",
        "pty",
        "sa",
        "sarl",
        "bv",
        "nv",
        "ag",
        "oy",
        "oyj",
        "ab",
        "spa",
        "srl",
        "sas",
        "kft",
        "ks",
        "sp",
        "group",
        "holdings",
        "partners",
        "associates",
        "international",
        "global",
    ]
    name = " ".join(name.split())
    for keyword in company_keywords:
        # Check if the keyword is a whole word in the name (case-insensitive)
        if re.search(rf"\b{re.escape(keyword)}\b", name, re.IGNORECASE):
            return 1  # It's a company
    return 0  # It's an individual


def split_name(name):
    """
    Splits a full name into given names and surname.
    Returns a Series with 'given_name' and 'surname'.
    """
    if pd.isnull(name) or name.strip() == "":
        return pd.Series({"given_name": "", "surname": ""})
    name = name.strip()
    name_parts = name.split()
    surname = name_parts[-1]
    given_name = " ".join(name_parts[:-1])
    return pd.Series({"given_name": given_name.strip(), "surname": surname.strip()})


def perform_feature_engineering(df):
    
    # Encoding 'parsed_name' using Soundex, Metaphone, and NYSIIS
    print("Encoding 'parsed_name' using Soundex, Metaphone, and NYSIIS")
    df["name_soundex"] = df["parsed_name"].apply(soundex)
    df["name_metaphone"] = df["parsed_name"].apply(metaphone)
    df["name_nysiis"] = df["parsed_name"].apply(nysiis)

    # Creating 'is_company' feature
    print("Creating 'is_company' feature")
    df["is_company"] = df["parsed_name"].apply(is_company)

    # Splitting 'parsed_name' into 'given_name' and 'surname'
    print("Splitting 'parsed_name' into 'given_name' and 'surname'")
    name_split = df["parsed_name"].apply(split_name)
    df = pd.concat([df, name_split], axis=1)

    # Encoding 'surname' using Soundex, Metaphone, and NYSIIS
    print("Encoding 'surname' using Soundex, Metaphone, and NYSIIS")
    df["surname_soundex"] = df["surname"].apply(soundex)
    df["surname_metaphone"] = df["surname"].apply(metaphone)
    df["surname_nysiis"] = df["surname"].apply(nysiis)

    # Creating 'surname_length' feature
    df["surname_length"] = df["surname"].apply(len)

    return df
