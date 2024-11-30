def levenshtein_distance(str1, str2):
    # Create a matrix to store distances
    len_str1 = len(str1) + 1
    len_str2 = len(str2) + 1
    matrix = [[0] * len_str2 for _ in range(len_str1)]

    # Initialize the matrix
    for i in range(len_str1):
        matrix[i][0] = i
    for j in range(len_str2):
        matrix[0][j] = j

    # Fill the matrix
    for i in range(1, len_str1):
        for j in range(1, len_str2):
            cost = 0 if str1[i - 1] == str2[j - 1] else 1
            matrix[i][j] = min(matrix[i - 1][j] + 1,       # Deletion
                               matrix[i][j - 1] + 1,       # Insertion
                               matrix[i - 1][j - 1] + cost) # Substitution

    return matrix[-1][-1]

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
        distance.append(levenshtein_distance(x1[col], x2[col]))
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
    print(f'unit test Levenshtein Distance between "{x}" and "{y}":', levenshtein_distance(x, y))  # Output: 3