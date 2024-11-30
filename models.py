from utils import *
from sklearn.preprocessing import MinMaxScaler
from collections import Counter
import numpy as np
from sklearn.metrics import accuracy_score
import pandas as pd
from data_preprocessing import data_preprocessing

class BaseModel:
    def __init__(self):
        pass

    def fit(self, X_train, y_train):
        pass

    def predict(self, X_test):
        predictions = X_test.apply(lambda x: self._predict(x), axis=1)
        return predictions
    
    def _predict(self, x):
        pass

class KNN(BaseModel):
    def __init__(self, k=3, distance_function=custom_distance):
        self.k = k
        self.distance_function = distance_function
        self.scaler = MinMaxScaler()
        
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def _predict(self, x):
        # Calculate distances between x and all points in the training set
        distances = self.X_train.apply(lambda x_train: self.distance_function(x, x_train), axis=1).tolist()# return vector n*features
        distances = np.array(distances)
        
        distances = self.scaler.fit_transform(distances)
        distances = np.linalg.norm(distances, axis=1)
        # Sort the distances and get the indices of the k-nearest neighbors
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        # Return the most common label among the k-nearest neighbors
        most_common = Counter(k_nearest_labels).most_common(1)
        # print(most_common)
        return most_common[0][0]
    

def train(model, X_train, y_train, **kwargs):
    if model.__class__.__name__ == 'KNN':
        if 'k' in kwargs:
            model.k = kwargs['k']
        if 'distance_function' in kwargs:
            model.distance_function = kwargs['distance_function']
        model.fit(X_train, y_train)
    else:
        'Model not supported'

# Example Usage
if __name__ == "__main__":
    # Toy dataset (features and labels)
    # train_df = data_preprocessing('account_booking_train.csv', 'external_parties_train.csv', 'train.csv')
    # test_df = data_preprocessing('account_booking_test.csv', 'external_parties_test.csv', 'test.csv')
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    train_df['transaction_date'] = pd.to_datetime(train_df['transaction_date'])
    test_df['transaction_date'] = pd.to_datetime(test_df['transaction_date'])
    
    X_train = train_df.drop(columns='external_id')
    y_train = train_df.external_id
    X_test = test_df

    # Initialize the KNN classifier
    model = KNN(k=3)
    train(model, X_train, y_train, k=3, distance_function=custom_distance)
    # Predict on the test data
    print('Unit Test')
    predictions = model.predict(X_train.iloc[:2])
    
    # Save the predictions to a CSV file to submit
    predictions.to_csv('predictions.csv')   
    accuracy = accuracy_score(y_train[:2], predictions)
    print("Accuracy of prediction:", accuracy)