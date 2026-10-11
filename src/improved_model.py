import numpy as np
import pandas as pd
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report


# ============================================================
# 1. Dataset load karo
# ============================================================

data = pd.read_csv(
    "data/SMSSpamCollection",
    sep="\t",
    header=None,
    names=["label", "message"]
)


# ============================================================
# 2. Smart text cleaning
# ============================================================

def smart_clean(text):
    # Lowercase
    text = text.lower()

    # URLs ko ek meaningful token mein convert karo
    text = re.sub(r"https?://\S+|www\.\S+", " URL ", text)

    # Extra spaces remove karo
    text = re.sub(r"\s+", " ", text).strip()

    return text


data["clean_message"] = data["message"].apply(smart_clean)


# ============================================================
# 3. Labels
# ============================================================

y = data["label"].map({
    "ham": 0,
    "spam": 1
})


# ============================================================
# 4. Train/Test Split
# ============================================================

X_train_text, X_test_text, y_train, y_test = train_test_split(
    data["clean_message"],
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


print("Training samples:", len(X_train_text))
print("Testing samples:", len(X_test_text))


# ============================================================
# 5. Word + Character TF-IDF
# ============================================================

features = FeatureUnion([
    
    (
        "word_tfidf",
        TfidfVectorizer(
            max_features=5000,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2
        )
    ),

    (
        "char_tfidf",
        TfidfVectorizer(
            analyzer="char",
            ngram_range=(3, 5),
            min_df=2,
            max_features=10000
        )
    )
])


# Training data se features learn karo
X_train = features.fit_transform(X_train_text)

# Test data ko same features mein convert karo
X_test = features.transform(X_test_text)


print("\nFeature Matrix:")
print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)


# ============================================================
# 6. Improved Linear SVM
# ============================================================

svm_model = LinearSVC() 

svm_model.fit(X_train, y_train)


# ============================================================
# 7. Predictions
# ============================================================

predictions = svm_model.predict(X_test)


# ============================================================
# 8. Results
# ============================================================

accuracy = accuracy_score(y_test, predictions)

print("\n" + "=" * 60)
print("IMPROVED LINEAR SVM RESULTS")
print("=" * 60)

print("Accuracy:", accuracy)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predictions,
        target_names=["Ham", "Spam"]
    )
)

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# Improved model confusion matrix
cm = confusion_matrix(y_test, predictions)

print("\nConfusion Matrix:")
print(cm)


plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Ham", "Spam"],
    yticklabels=["Ham", "Spam"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Improved Linear SVM - Confusion Matrix")

plt.tight_layout()
plt.show() 

# ============================================================
# 9. Spam Decision Threshold Tuning
# ============================================================

from sklearn.metrics import precision_score, recall_score, f1_score


# SVM decision scores
decision_scores = svm_model.decision_function(X_test)


print("\n" + "=" * 70)
print("THRESHOLD TUNING")
print("=" * 70)

print(
    f"{'Threshold':<12}"
    f"{'Accuracy':<12}"
    f"{'Spam Precision':<16}"
    f"{'Spam Recall':<14}"
    f"{'Spam F1':<10}"
    f"{'Missed Spam':<13}"
    f"{'False Positives'}"
)

print("-" * 90)


# Different thresholds test karo
thresholds = [
    0.20,
    0.10,
    0.05,
    0.00,
    -0.05,
    -0.10,
    -0.15,
    -0.20,
    -0.25,
    -0.30,
    -0.32,
-0.34,
-0.36,
-0.38,
    -0.40,
    -0.50,
-0.60,
-0.70,
-0.80,
-1.00
]


for threshold in thresholds:

    # Threshold ke basis par prediction
    threshold_predictions = (
        decision_scores >= threshold
    ).astype(int)

    accuracy = accuracy_score(
        y_test,
        threshold_predictions
    )

    precision = precision_score(
        y_test,
        threshold_predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        threshold_predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        threshold_predictions,
        zero_division=0
    )

    cm_threshold = confusion_matrix(
        y_test,
        threshold_predictions
    )

    false_positives = cm_threshold[0][1]
    missed_spam = cm_threshold[1][0]

    print(
        f"{threshold:<12.2f}"
        f"{accuracy:<12.4f}"
        f"{precision:<16.4f}"
        f"{recall:<14.4f}"
        f"{f1:<10.4f}"
        f"{missed_spam:<13}"
        f"{false_positives}"
    ) 

    # ============================================================
# 10. SAVE TRAINED MODEL
# ============================================================

import joblib
import os

# Final threshold
FINAL_THRESHOLD = -0.30
# Save actual test labels and predictions for confusion matrix
test_predictions = (decision_scores >= FINAL_THRESHOLD).astype(int)

joblib.dump(
    np.asarray(y_test),
    "models/y_test.joblib"
)

joblib.dump(
    test_predictions,
    "models/y_test_predictions.joblib"
)

print("Test labels and predictions saved successfully!")
# Models folder ensure karo
os.makedirs("models", exist_ok=True)

# Save trained SVM model
joblib.dump(
    svm_model,
    "models/spam_svm_model.joblib"
)

# Save feature extractor
joblib.dump(
    features,
    "models/tfidf_features.joblib"
)

# Save threshold
joblib.dump(
    FINAL_THRESHOLD,
    "models/spam_threshold.joblib"
)

print("\n" + "=" * 70)
print("MODEL SAVED SUCCESSFULLY")
print("=" * 70)
print("Model     : models/spam_svm_model.joblib")
print("Features  : models/tfidf_features.joblib")
print("Threshold : models/spam_threshold.joblib")