import pandas as pd
import re


# Dataset load karo
data = pd.read_csv(
    "data/SMSSpamCollection",
    sep="\t",
    header=None,
    names=["label", "message"]
)


# Text cleaning function
def clean_text(text):
    # Lowercase
    text = text.lower()

    # URLs remove karo
    text = re.sub(r"http\S+|www\S+", "", text)

    # Special characters remove karo
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # Extra spaces remove karo
    text = re.sub(r"\s+", " ", text).strip()

    return text


# Clean messages
data["clean_message"] = data["message"].apply(clean_text)


# Result dekho
print("Original vs Cleaned messages:\n")
print(data[["message", "clean_message"]].head(10)) 

from sklearn.feature_extraction.text import TfidfVectorizer


# TF-IDF vectorizer
vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words="english"
)


# Text ko numerical features mein convert karo
X = vectorizer.fit_transform(data["clean_message"])


print("\nTF-IDF Shape:")
print(X.shape)   


from sklearn.model_selection import train_test_split


# Labels ko numbers mein convert karo
y = data["label"].map({
    "ham": 0,
    "spam": 1
})


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


print("\nTrain/Test Split:")
print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0]) 


from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report


# Naive Bayes model
nb_model = MultinomialNB()

# Model train karo
nb_model.fit(X_train, y_train)

# Test data par prediction
nb_predictions = nb_model.predict(X_test)


# Accuracy
nb_accuracy = accuracy_score(y_test, nb_predictions)

print("\nNaive Bayes Results:")
print("Accuracy:", nb_accuracy)

print("\nClassification Report:")
print(classification_report(
    y_test,
    nb_predictions,
    target_names=["Ham", "Spam"]
)) 

from sklearn.linear_model import LogisticRegression


# Logistic Regression model
lr_model = LogisticRegression(
    max_iter=1000
)

# Model train karo
lr_model.fit(X_train, y_train)

# Test data par prediction
lr_predictions = lr_model.predict(X_test)


# Accuracy
lr_accuracy = accuracy_score(y_test, lr_predictions)

print("\nLogistic Regression Results:")
print("Accuracy:", lr_accuracy)

print("\nClassification Report:")
print(classification_report(
    y_test,
    lr_predictions,
    target_names=["Ham", "Spam"]
)) 

from sklearn.svm import LinearSVC


# Linear SVM model
svm_model = LinearSVC()

# Model train karo
svm_model.fit(X_train, y_train)

# Test data par prediction
svm_predictions = svm_model.predict(X_test)


# Accuracy
svm_accuracy = accuracy_score(y_test, svm_predictions)

print("\nLinear SVM Results:")
print("Accuracy:", svm_accuracy)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        svm_predictions,
        target_names=["Ham", "Spam"]
    )
)


from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# Confusion Matrix
cm = confusion_matrix(y_test, svm_predictions)

print("\nConfusion Matrix:")
print(cm)


# Confusion Matrix plot
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
plt.title("Linear SVM - Confusion Matrix")

plt.tight_layout()
# Error Analysis
# Actual spam tha, lekin model ne ham predict kiya
missed_spam = data.loc[
    y_test.index[
        (y_test == 1) & (svm_predictions == 0)
    ]
]


print("\n" + "=" * 60)
print("MISSED SPAM MESSAGES")
print("=" * 60)

print("Total missed spam:", len(missed_spam))

print("\nExamples:")

for i, row in missed_spam.head(10).iterrows():
    print("\nOriginal message:")
    print(row["message"])
plt.show()