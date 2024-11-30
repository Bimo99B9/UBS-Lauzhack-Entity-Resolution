# Import necessary modules
import preprocessing_utils.feature_engineering as feature_engineering
from preprocessing_utils.data_preprocessing import data_preprocessing


# Section 1: Preprocessing
def preprocess_data():
    """
    Preprocess the training and testing datasets.
    """
    # Load and preprocess training data
    account_booking_train, external_parties_train = data_preprocessing(
        "account_booking_train.csv", "external_parties_train.csv"
    )

    # Load and preprocess testing data
    account_booking_test, external_parties_test = data_preprocessing(
        "account_booking_test.csv", "external_parties_test.csv"
    )

    return (
        account_booking_train,
        external_parties_train,
        account_booking_test,
        external_parties_test,
    )


# Section 2: Feature Engineering
def feature_engineering_data(external_parties_train, external_parties_test):
    """
    Apply feature engineering to the training and testing datasets.
    """
    # Apply feature engineering to training data
    external_parties_train = feature_engineering.perform_feature_engineering(
        external_parties_train
    )

    # Apply feature engineering to testing data
    external_parties_test = feature_engineering.perform_feature_engineering(
        external_parties_test
    )

    return external_parties_train, external_parties_test


# Section 3: Save Processed Data
def save_data(external_parties_train, external_parties_test):
    """
    Save the processed training and testing datasets to CSV files.
    """
    # Save a sample of training data for inspection
    external_parties_train.head(15).to_csv("example.csv", index=False)

    # Save the full processed training and testing data
    external_parties_train.to_csv(
        "data/processed/external_parties_train.csv", index=False
    )
    external_parties_test.to_csv(
        "data/processed/external_parties_test.csv", index=False
    )


if __name__ == "__main__":
    # Preprocess the data
    (
        account_booking_train,
        external_parties_train,
        account_booking_test,
        external_parties_test,
    ) = preprocess_data()

    # Perform feature engineering
    external_parties_train, external_parties_test = feature_engineering_data(
        external_parties_train, external_parties_test
    )

    # Save the processed data
    save_data(external_parties_train, external_parties_test)
