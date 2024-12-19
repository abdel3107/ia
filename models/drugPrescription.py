import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import chain

# Load datasets
description_df = pd.read_csv('dataset/description.csv')
medications_df = pd.read_csv('dataset/medications.csv')
symptoms_df = pd.read_csv('dataset/symptoms_df.csv')

# %%
description_df.describe()

# %%
medications_df.describe()
# %%
symptoms_df.describe()
# %%

# %%
medications_df.head()
# %%
symptoms_df.head()
# %%
# Standardize disease names
description_df['Disease'] = description_df['Disease'].str.lower().str.strip()
medications_df['Disease'] = medications_df['Disease'].str.lower().str.strip()
symptoms_df['Disease'] = symptoms_df['Disease'].str.lower().str.strip()

# %%
description_df.isnull().sum()

# %%
medications_df.isnull().sum()

# %%
symptoms_df.isnull().sum()
# %%

# %%
# Merge datasets
merged_df = symptoms_df.merge(medications_df, on='Disease', how='inner')
merged_df = merged_df.merge(description_df, on='Disease', how='inner')

# %%

# %%
merged_df.head(10)
# %%
# Save cleaned dataset
merged_df.to_csv('processed_dataset.csv', index=False)

# %%
# Check the shape of the dataset

merged_df.describe()

# %%
# Column-wise non-null counts and data types
merged_df.info()

# %%

# View the first few rows
merged_df.head()
# %%
print(merged_df['Medication'].head())
print(type(merged_df['Medication'].iloc[0]))  # Check the type of an individual cell

# %%
# Frequency of diseases
disease_counts = merged_df['Disease'].value_counts()
disease_counts
# %%
plt.figure(figsize=(10, 6))
sns.barplot(x=disease_counts.index, y=disease_counts.values)
plt.title("Distribution of Diseases")
plt.xlabel("Disease")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.show()
# %%
import ast

# Convert strings to lists
merged_df['Medication'] = merged_df['Medication'].apply(
    lambda x: ast.literal_eval(x) if isinstance(x, str) else x
)
# %%
# Frequency of medications (flatten if they are lists)

all_medications = list(chain.from_iterable(merged_df['Medication']))
medication_counts = pd.Series(all_medications).value_counts()

medication_counts
# %%
# Plot top medications
plt.figure(figsize=(10, 6))
sns.barplot(x=medication_counts[:10].index, y=medication_counts[:10].values)
plt.title("Top 10 Medications")
plt.xlabel("Medication")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.show()
# %%
# Aggregate symptoms into a single column
symptoms_columns = ['Symptom_1', 'Symptom_2', 'Symptom_3', 'Symptom_4']
all_symptoms = merged_df[symptoms_columns].apply(lambda x: ', '.join(x.dropna()), axis=1)

# Count occurrences of each symptom
from collections import Counter

symptom_counts = Counter(', '.join(all_symptoms).split(', '))

# Plot top symptoms
symptom_counts_df = pd.DataFrame(symptom_counts.items(), columns=['Symptom', 'Count']).sort_values(by='Count',
                                                                                                   ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=symptom_counts_df[:10]['Symptom'], y=symptom_counts_df[:10]['Count'])
plt.title("Top 10 Symptoms")
plt.xlabel("Symptom")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.show()

# %%
# Heatmap of diseases and symptoms
disease_symptom_matrix = pd.crosstab(merged_df['Disease'], all_symptoms)
sns.heatmap(disease_symptom_matrix, cmap='coolwarm', annot=False)
plt.title("Disease-Symptom Relationship")
plt.show()

# %%
from sklearn.preprocessing import MultiLabelBinarizer

# Combine symptoms into a list for each disease
merged_df['All_Symptoms'] = merged_df[['Symptom_1', 'Symptom_2', 'Symptom_3', 'Symptom_4']].values.tolist()

# %%
# Remove missing symptoms
merged_df['All_Symptoms'] = merged_df['All_Symptoms'].apply(
    lambda x: [symptom for symptom in x if pd.notna(symptom)]
)

# One-hot encode symptoms
mlb = MultiLabelBinarizer()
symptoms_encoded = mlb.fit_transform(merged_df['All_Symptoms'])

# Create a DataFrame for the encoded symptoms
symptoms_df = pd.DataFrame(symptoms_encoded, columns=mlb.classes_)

# %%
# Merge encoded symptoms with disease column
final_df = pd.concat([merged_df['Disease'], symptoms_df], axis=1)

# Features (X) and target (y)
X = final_df.drop(columns=['Disease'])
y = final_df['Disease']

# %%
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Example: Using RandomForest for feature importance
rf = RandomForestClassifier(random_state=42)
rf.fit(X, y)
feature_importances = pd.Series(rf.feature_importances_, index=X.columns)
top_features = feature_importances.nlargest(20).index
X = X[top_features]

# %%
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# %%
model.fit(X_train, y_train)

# %%
from sklearn.metrics import classification_report, accuracy_score

# Make predictions
y_pred = model.predict(X_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# %%
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
}
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

# %%
import joblib

joblib.dump(model, 'disease_prediction_model.pkl')

# %%
# Make predictions using the test set (X_test)
y_pred = model.predict(X_test)

# %%
# Example symptoms input
test_symptoms = [' high_fever', ' vomiting', ' chills', ' fatigue']

# Encode the symptoms (ensure it matches the format used in training)
test_encoded = pd.DataFrame(mlb.transform([test_symptoms]), columns=mlb.classes_)

# Align the columns of the test data with the training data
# Reindex the columns to match the training data (X_train.columns)
test_encoded = test_encoded.reindex(columns=X_train.columns, fill_value=0)

# Predict the disease
predicted_disease = model.predict(test_encoded)
print("Predicted Disease:", predicted_disease[0])

# %%

# %%
# Predicted disease
predicted_disease = model.predict(test_encoded)[0]

# Look up medications for the predicted disease
prescribed_medications = medications_df.loc[medications_df['Disease'] == predicted_disease, 'Medication']

# If multiple medications are listed, extract them
if not prescribed_medications.empty:
    medications_list = prescribed_medications.iloc[0]
    print(f"Prescribed Medications for {predicted_disease}: {medications_list}")
else:
    print(f"No medications found for the disease: {predicted_disease}")


# %%
def recommend_medication(symptoms):
    # Encode symptoms
    encoded_symptoms = pd.DataFrame(mlb.transform([symptoms]), columns=mlb.classes_)
    encoded_symptoms = encoded_symptoms.reindex(columns=X_train.columns, fill_value=0)

    # Predict disease
    predicted_disease = model.predict(encoded_symptoms)[0]

    # Look up medications
    prescribed_medications = medications_df.loc[medications_df['Disease'] == predicted_disease, 'Medication']

    if not prescribed_medications.empty:
        medications_list = prescribed_medications.iloc[0]
        return predicted_disease, medications_list
    else:
        return predicted_disease, "No medications found"


# Example usage
symptoms = [' skin_rash', ' stomach_pain', ' vomiting', ' cough']
disease, medications = recommend_medication(symptoms)
print(f"Predicted Disease: {disease}")
print(f"Recommended Medications: {medications}")

# %%
mlb = MultiLabelBinarizer()

# Fit the MultiLabelBinarizer on the symptoms column (AllSymptoms)
mlb.fit(merged_df['All_Symptoms'].tolist())

# Check the unique symptoms it has learned
print(mlb.classes_)  # assuming the symptoms are in a list format

# %% md

# %%
# Transform the symptoms into binary format (Disease x Symptom matrix)
encoded_symptoms = mlb.transform(merged_df['All_Symptoms'].tolist())

# Convert it into a DataFrame for easier inspection
encoded_symptoms_df = pd.DataFrame(encoded_symptoms, columns=mlb.classes_)

# Add the disease column to the DataFrame to indicate which disease the symptoms belong to
# encoded_symptoms_df['Disease'] = merged_df['Disease']

# Now we have a Disease x Symptom matrix with 1's and 0's indicating symptom presence
print(encoded_symptoms_df.head())

# %%
# Save the MLB to a file
joblib.dump(mlb, 'mlb.pkl')