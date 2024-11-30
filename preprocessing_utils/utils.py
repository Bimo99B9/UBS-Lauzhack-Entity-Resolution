from rapidfuzz.distance import JaroWinkler

def custom_distance(x1, x2):
    """
    Custom distance function for dataframe attribute
    assumption: x1 and x2 are pandas dataframe
    containing columns:
    - parsed_name: string
    - party_info_unstructured: string
    - debit_credit_indicator: categorical
    - account_id: categorical
    - party_role: categorical
    - transaction_amount: numerical
    - transaction_date: datetime
    """
    distance = []
    string_col = ['parsed_name', 'party_info_unstructured']
    categorical_col = ['debit_credit_indicator', 'account_id', 'party_role']    
    val_col = ['transaction_amount', 'transaction_date']
    for col in string_col:
        distance.append(JaroWinkler.distance(x1[col], x2[col]))
    for col in categorical_col:
        distance.append(x1[col] != x2[col])
    for col in val_col:
        dis = abs(x1[col] - x2[col])
        if col == 'transaction_date':
            dis = int(dis.days)
        distance.append(dis)
    # print(distance)
    return distance

if __name__ == '__main__':
    x = 'kitten'
    y = 'sitting'
    print(f'unit test Levenshtein Distance between "{x}" and "{y}":', JaroWinkler.distance(x, y))  # Output: 3