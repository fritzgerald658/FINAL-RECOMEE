import random
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB  # Import MultinomialNB
from sklearn.metrics import accuracy_score

# Read the CSV file
df = pd.read_csv('FINAL-DSET-Macintosh.csv')

# Handle missing values
df = df.fillna('')

# Convert all columns to string type
df = df.astype(str)

# Select features (X) and target variable (y)
X = df[['Skills', 'Interests', 'Industry']]
y = df['Career']

# Define the range for test_size and random_state values
test_size_range = [0.2, 0.3, 0.4, 0.5]  # Adjust as needed
random_state_range = range(200)  # Adjust as needed

for _ in range(100):  # Change the range as needed
    # Randomly select test_size and random_state values
    test_size = round(random.choice(test_size_range), 3)
    random_state = random.choice(random_state_range)

    # Split the dataset using the selected values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Combine text features into a single string for each instance
    X_train_combined = X_train.apply(lambda x: ' '.join(x), axis=1)
    X_test_combined = X_test.apply(lambda x: ' '.join(x), axis=1)

    # Use CountVectorizer to convert text data into numerical format
    vectorizer = CountVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train_combined)
    X_test_vec = vectorizer.transform(X_test_combined)

    # Create and train the Multinomial Naive Bayes model
    nb_model = MultinomialNB()  # Create MultinomialNB object
    nb_model.fit(X_train_vec, y_train)

    # Make predictions on the training set
    y_train_pred = nb_model.predict(X_train_vec)

    # Calculate accuracy on the training set
    training_accuracy = accuracy_score(y_train, y_train_pred)
    print(f'Model Accuracy on Training Set (test_size={test_size:.3f}, random_state={random_state}): {training_accuracy:.3f}')

    # Make predictions on the test set
    y_pred = nb_model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Model Accuracy on Test Set (test_size={test_size:.3f}, random_state={random_state}): {accuracy:.3f}')

    print()  # Print an empty line for spacing
