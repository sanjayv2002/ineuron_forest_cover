import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def load_data(file_path):
    """
    Load data from a CSV file.
    
    Args:
        file_path (str): Path to the CSV file.
    
    Returns:
        pandas.DataFrame: Loaded data.
    """
    return pd.read_csv(file_path)

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split the data into training and testing sets.
    
    Args:
        X (pandas.DataFrame): Features.
        y (pandas.Series): Target variable.
        test_size (float): Proportion of the data to be used as the test set.
        random_state (int): Random seed for reproducibility.
    
    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def scale_data(X_train, X_test):
    """
    Scale the features using StandardScaler.
    
    Args:
        X_train (pandas.DataFrame): Training features.
        X_test (pandas.DataFrame): Testing features.
    
    Returns:
        tuple: Scaled X_train, X_test
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

def train_model(X_train, y_train, n_estimators=100, random_state=42):
    """
    Train a Random Forest classifier.
    
    Args:
        X_train (numpy.ndarray): Scaled training features.
        y_train (pandas.Series): Training target variable.
        n_estimators (int): Number of trees in the forest.
        random_state (int): Random seed for reproducibility.
    
    Returns:
        RandomForestClassifier: Trained model.
    """
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model on the test set.
    
    Args:
        model (RandomForestClassifier): Trained model.
        X_test (numpy.ndarray): Scaled testing features.
        y_test (pandas.Series): Testing target variable.
    """
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    print("Accuracy:", acc)
