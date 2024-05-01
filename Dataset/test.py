import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, f1_score, confusion_matrix
from joblib import dump, load
import matplotlib.pyplot as plt
import seaborn as sns

# Read the CSV file
df = pd.read_csv('FINAL-DSET-Macintosh.csv')

# Handle missing values
df = df.fillna('')

# Convert all columns to string type
df = df.astype(str)

# Select features (X) and target variable (y)
X = df[['Skills', 'Interests', 'Industry']]
y = df['Career']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=24)

# Combine text features into a single string for each instance
X_train_combined = X_train.apply(lambda x: ' '.join(x), axis=1)
X_test_combined = X_test.apply(lambda x: ' '.join(x), axis=1)

# Use CountVectorizer to convert text data into numerical format
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train_combined)
X_test_vec = vectorizer.transform(X_test_combined)

# Create and train the Multinomial Naive Bayes model
nb_model = MultinomialNB()
nb_model.fit(X_train_vec, y_train)

# Make predictions on the training set
y_train_pred = nb_model.predict(X_train_vec)

# Calculate accuracy on the training set
training_accuracy = accuracy_score(y_train, y_train_pred)
print(f'Model Accuracy on Training Set: {training_accuracy:.3f}')

# Make predictions on the test set
y_pred = nb_model.predict(X_test_vec)

# Calculate accuracy on the test set
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy on Test Set: {accuracy:.3f}')

# Calculate precision
precision = precision_score(y_test, y_pred, average='weighted')
print(f'Precision: {precision:.3f}')

#Calculate F1 score
f1 = f1_score(y_test, y_pred, average='weighted')
print(f'F1 Score: {f1:.3f}')

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(conf_matrix)

# Create a heatmap of the confusion matrix
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')

#Add labels and title
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix')

# Show the plot
plt.show()

# fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# # Count the number of occurrences of each category in the 'Skills' column
# skills_counts = df['Skills'].value_counts()
# axs[0].bar(skills_counts.index, skills_counts.values)
# axs[0].set_xlabel('Skills')
# axs[0].set_ylabel('Count')
# axs[0].set_title('Distribution of Skills')

# # Count the number of occurrences of each category in the 'Interests' column
# interests_counts = df['Interests'].value_counts()
# axs[1].bar(interests_counts.index, interests_counts.values)
# axs[1].set_xlabel('Interests')
# axs[1].set_ylabel('Count')
# axs[1].set_title('Distribution of Interests')

# # Count the number of occurrences of each category in the 'Industry' column
# industry_counts = df['Industry'].value_counts()
# axs[2].bar(industry_counts.index, industry_counts.values)
# axs[2].set_xlabel('Industry')
# axs[2].set_ylabel('Count')
# axs[2].set_title('Distribution of Industry')

# plt.tight_layout()
# plt.show()
# Save the model and vectorizer
dump(nb_model, 'ml_model.joblib')
dump(vectorizer, 'ml_vectorizer.joblib')
