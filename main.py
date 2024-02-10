import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, classification_report, accuracy_score, confusion_matrix
from daal4py.sklearn import patch_sklearn as d4p_patch_sklearn
from intel_tensorflow import IpuContext
from intel_neural_compressor import Compressor
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming you have a CSV file for both students and teachers
students_data = pd.read_csv('C:\\Users\\jnyanadeep\\Desktop\\student_prediction.csv')
teachers_data = pd.read_csv('C:\\Users\\jnyanadeep\\Desktop\\Teacher_survey.csv')

# Assuming 'Course ID' is a common column in both datasets
merged_data = pd.merge(students_data, teachers_data, on='Course ID', how='inner')

# Select relevant features for mapping and prediction
mapping_features = ['STUDY_HRS', 'READ_FREQ', 'READ_FREQ_SCI', 'ATTEND_DEPT', 'IMPACT', 'ATTEND', 'PREP_EXAM', 'LISTENS', 'LIKES_DISCUSS']
teacher_features = ['I prefer to plan my tasks well in advance rather than dealing with them spontaneously.',
                    'I enjoy engaging in lively discussions and debates with colleagues and students.',
                    'I find satisfaction in guiding and mentoring students towards their academic and professional goals.',
                    'I tend to keep a professional distance from my students, maintaining a strictly academic relationship.',
                    'I enjoy exploring innovative teaching techniques and see mistakes and setbacks as opportunities for growth.',
                    'I prioritize academic excellence above all else, even if it means pushing students to their limits.',
                    'I am passionate about my subject area and enjoy sharing my enthusiasm with others.',
                    'I am patient and understanding when working with students who are struggling academically.',
                    'I believe in the importance of ethics and integrity in academic research and teaching.',
                    'I value creativity and encourage students to think outside the box in their assignments and projects']

# Create feature and target datasets
X = merged_data[mapping_features + teacher_features]
y = merged_data['GRADE']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Get the predicted grades for each student
predictions_intel = model.predict(preprocessor.transform(X_test))
predictions_tf = model_tf.predict(X_test)

# Create a table with mapping information
mapping_table = pd.DataFrame({
    'Student_ID': X_test.index,
    'Teacher_ID_Intel': np.argmax(predictions_intel, axis=1),
    'Teacher_ID_TensorFlow': np.argmax(predictions_tf, axis=1),
    'Predicted_Grade_Intel': np.round(predictions_intel).astype(int),
    'Predicted_Grade_TensorFlow': np.round(predictions_tf).astype(int),
    'True_Grade': y_test
})

# Display the mapping table
print(mapping_table)
# Preprocessing for numerical data
numeric_features = X.select_dtypes(include=['float64', 'int64']).columns
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Preprocessing for categorical data
categorical_features = X.select_dtypes(include=['object']).columns
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Initialize Intel oneAPI context
d4p_patch_sklearn()
IpuContext()

# Intel oneDNN Compressor
with Compressor():
    # Define the model using Intel oneDAL
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(preprocessor.fit_transform(X_train), y_train, epochs=50, batch_size=32, verbose=0)

    # Make predictions
    predictions_intel = model.predict(preprocessor.transform(X_test))

# Calculate MAE and Accuracy for Intel oneAPI
mae_intel = mean_absolute_error(y_test, predictions_intel)
accuracy_intel = accuracy_score(y_test, np.round(predictions_intel))

print(f'Mean Absolute Error (Intel): {mae_intel}')
print(f'Accuracy (Intel): {accuracy_intel}')

# TensorFlow Model
model_tf = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model_tf.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the TensorFlow model
model_tf.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)

# Make predictions
predictions_tf = model_tf.predict(X_test)

# Calculate MAE and Accuracy for TensorFlow
mae_tf = mean_absolute_error(y_test, predictions_tf)
accuracy_tf = accuracy_score(y_test, np.round(predictions_tf))

print(f'Mean Absolute Error (TensorFlow): {mae_tf}')
print(f'Accuracy (TensorFlow): {accuracy_tf}')

# Confusion Matrix
conf_matrix_intel = confusion_matrix(y_test, np.round(predictions_intel), labels=range(8))
conf_matrix_tf = confusion_matrix(y_test, np.round(predictions_tf), labels=range(8))

# Plot heatmaps
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
sns.heatmap(conf_matrix_intel, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0], cbar=False)
axes[0, 0].set_title('Confusion Matrix (Intel)')
axes[0, 0].set_xlabel('Predicted Grade')
axes[0, 0].set_ylabel('True Grade')

sns.heatmap(conf_matrix_tf, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1], cbar=False)
axes[0, 1].set_title('Confusion Matrix (TensorFlow)')
axes[0, 1].set_xlabel('Predicted Grade')
axes[0, 1].set_ylabel('True Grade')

sns.heatmap(conf_matrix_intel / np.sum(conf_matrix_intel), annot=True, fmt='.2%', cmap='Blues', ax=axes[1, 0], cbar=False)
axes[1, 0].set_title('Normalized Confusion Matrix (Intel)')
axes[1, 0].set_xlabel('Predicted Grade')
axes[1, 0].set_ylabel('True Grade')

sns.heatmap(conf_matrix_tf / np.sum(conf_matrix_tf), annot=True, fmt='.2%', cmap='Blues', ax=axes[1, 1], cbar=False)
axes[1, 1].set_title('Normalized Confusion Matrix (TensorFlow)')
axes[1, 1].set_xlabel('Predicted Grade')
axes[1, 1].set_ylabel('True Grade')

plt.tight_layout()
plt.show()