import numpy as np
import pandas as pd
import learning_rate
from SVM import SVM

np.random.seed(350)
print("\nReading data ...")
DATA = 'project_data/data/'

# Load train, test, and eval data using pandas

train_df = pd.read_csv(DATA + 'tfidf/tfidf.train.csv')
test_df = pd.read_csv(DATA + 'tfidf/tfidf.test.csv')
eval_df = pd.read_csv(DATA + 'tfidf/tfidf.eval.anon.csv')

# Print available columns
print(eval_df.columns)

# Assume 'label' is the column name for the labels
train_data = train_df.drop('label', axis=1).values
train_label = train_df['label'].values
test_data = test_df.drop('label', axis=1).values
test_label = test_df['label'].values
eval_data = eval_df.drop('label', axis=1).values

# Create an example_id for eval data if it doesn't exist
if 'example_id' not in eval_df.columns:
    eval_df['example_id'] = np.arange(len(eval_df))

eval_example_ids = eval_df['example_id'].values

# Simple feature scaling (standardization)
mean = train_data.mean(axis=0)
std = train_data.std(axis=0)
train_data = (train_data - mean) / std
test_data = (test_data - mean) / std
eval_data = (eval_data - mean) / std

# SVM parameters
init_learning_rate = 1
C = 0.01
epochs = 20
max_features = train_data.shape[1]

# Initialize SVM
svm = SVM(max_features, C)

# Training
for epoch in range(epochs):
    for i in range(len(train_data)):
        x = train_data[i]
        y = train_label[i]
        lr_t = learning_rate.decay_lr(init_learning_rate, epoch)
        svm.update(lr_t, x, y)

# Function to predict and adjust output
def predict_and_adjust(model, data):
    predictions = []
    for i in range(len(data)):
        prediction = model.predict(data[i])
        # Convert -1 to 0
        prediction = 0 if prediction == -1 else 1
        predictions.append(prediction)
    return predictions

# Predict on eval dataset
eval_predictions = predict_and_adjust(svm, eval_data)

# Save eval predictions to a CSV file
result_df = pd.DataFrame({
    'example_id': eval_example_ids,
    'label': eval_predictions
})
result_df.to_csv('proc.csv', index=False)

print("Evaluation predictions saved to proc.csv.")
