from model import *

# Main code
data = load_data("data/train.csv")
X = data.drop(['Cover_Type'], axis=1)
y = data['Cover_Type']

X_train, X_test, y_train, y_test = split_data(X, y)
X_train_scaled, X_test_scaled = scale_data(X_train, X_test)
model = train_model(X_train_scaled, y_train)
evaluate_model(model, X_test_scaled, y_test)
