import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Load your dataset
df = pd.read_csv("ayurvedic_dosha_dataset.csv")

# Clean
df_clean = df.drop_duplicates()

# Split features and label
X_raw = df_clean.drop(columns=["Dosha"])
y_raw = df_clean["Dosha"]

# One-hot encode features
X_encoded = pd.get_dummies(X_raw)

# Encode label (Vata, Pitta, Kapha → 0, 1, 2)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_raw)
y_categorical = to_categorical(y_encoded)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y_categorical, test_size=0.2, random_state=42
)

# Build Deep Neural Network
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(3, activation='softmax'))  # 3 output classes

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)

# Evaluate on test set
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy:.2%}")

import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Convert predictions and true labels from one-hot to class indices
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)

# Confusion matrix & report
cm = confusion_matrix(y_true, y_pred)
report = classification_report(y_true, y_pred, target_names=label_encoder.classes_)

# Print text results
print("Confusion Matrix:\n", cm)
print("\nClassification Report:\n", report)

# Optional: Plot it
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix (AyurAI - Deep Learning)')
plt.tight_layout()
plt.show()

model.save("ayurai_model.h5")

# Save label encoder
import joblib
joblib.dump(label_encoder, "label_encoder.pkl")

# Save model input columns
X_encoded.columns.to_frame().T.to_csv("sample_model_input_columns.csv", index=False, header=False)