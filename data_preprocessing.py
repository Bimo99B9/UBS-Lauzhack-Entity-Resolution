import pandas as pd  

def data_preprocessing(dir_account_booking, dir_external_parties, save_name='', save=False):
    # Load the data
    account_booking = pd.read_csv(dir_account_booking)
    account_booking = account_booking.drop(columns=['transaction_currency'])

    # change debit credit to 0 and 1
    account_booking['debit_credit_indicator'] = account_booking['debit_credit_indicator'].map({'CREDIT': 0, 'DEBIT': 1})
    # change transaction date to datetime 
    account_booking['transaction_date'] = pd.to_datetime(account_booking['transaction_date'])
    
    external_parties = pd.read_csv(dir_external_parties)
    # change BENE to 0 and ORG to 1
    external_parties['party_role'] = external_parties['party_role'].map({'BENE': 0, 'ORG': 1})
    external_parties = external_parties.merge(account_booking, on='transaction_reference_id', how='left')
    print(external_parties.head())
    print('Done preprocessing data, save to:', save_name)
    external_parties.to_csv(save_name)
    return external_parties

if __name__ == '__main__':
    data_preprocessing('account_booking_train.csv', 'external_parties_train.csv', 'train.csv')
    data_preprocessing('account_booking_test.csv', 'external_parties_test.csv', 'test.csv')

