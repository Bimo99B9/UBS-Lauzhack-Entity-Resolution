import pandas as pd
from normalize_text import clean_text_dataset, remove_words


def data_preprocessing(
    dir_account_booking, dir_external_parties
):

    #### Account Booking ####

    # Load the data
    print(f"Loading data")
    account_booking = pd.read_csv(f"data/{dir_account_booking}")
    account_booking = account_booking.drop(columns=["transaction_currency"])

    # change debit credit to 0 and 1
    print(f"Changing debit credit to 0 and 1")
    account_booking["debit_credit_indicator"] = account_booking[
        "debit_credit_indicator"
    ].map({"CREDIT": 0, "DEBIT": 1})
    # change transaction date to datetime
    account_booking["transaction_date"] = pd.to_datetime(
        account_booking["transaction_date"]
    )

    ##### External parties #####
    print(f"Loading external parties")

    external_parties = pd.read_csv(f"data/{dir_external_parties}")
    # change BENE to 0 and ORG to 1
    print(f"Changing BENE to 0 and ORG to 1")
    external_parties["party_role"] = external_parties["party_role"].map(
        {"BENE": 0, "ORG": 1}
    )
    # external_parties = external_parties.merge(
    #     account_booking, on="transaction_reference_id", how="left"
    # )
    # print(external_parties.head())
    # print("Done preprocessing data, save to:", save_name)
    # external_parties.to_csv(save_name)

    print(f"Cleaning text data")
    external_parties = clean_text_dataset(external_parties, "parsed_name", name=True)
    # Clean addresses
    external_parties = clean_text_dataset(
        external_parties, "parsed_address_street_name", name=False
    )

    # Delete common words for names
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

    print(f"Removing common words")
    external_parties = remove_words(external_parties, "parsed_name", remove)

    return account_booking, external_parties


if __name__ == "__main__":
    account_booking_train, external_parties_train = data_preprocessing(
        "account_booking_train.csv", "external_parties_train.csv"
    )
    account_booking_test, external_parties_test = data_preprocessing(
        "account_booking_test.csv", "external_parties_test.csv"
    )
