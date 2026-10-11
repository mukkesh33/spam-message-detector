import joblib
import numpy as np
import seaborn as sns
import streamlit as st

st.set_page_config(
    page_title="AI Spam Message Detector",
    page_icon="🛡️",
    layout="centered"
)
import streamlit as st
import joblib
import re


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Spam Message Detector",
    page_icon="🛡️",
    layout="centered"
)


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

svm_model = joblib.load(
    "models/spam_svm_model.joblib"
)

features = joblib.load(
    "models/tfidf_features.joblib"
)

threshold = joblib.load(
    "models/spam_threshold.joblib"
)


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):

    text = text.lower()

    # Remove URLs
    text = re.sub(
        r"http\S+|www\S+",
        "",
        text
    )

    # Remove special characters
    text = re.sub(
        r"[^a-zA-Z\s]",
        "",
        text
    )

    # Remove extra spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def get_spam_indicators(message):
    text = message.lower()
    indicators = []

    prize_words = [
        "won", "winner", "prize", "reward",
        "cash", "free", "congratulations"
    ]

    if any(word in text for word in prize_words):
        indicators.append("🎁 Prize / reward related language")

    money_words = [
        "money", "cash", "₹", "rs",
        "rupees", "pounds", "dollars"
    ]

    if any(word in text for word in money_words):
        indicators.append("💰 Money-related language")

    urgency_words = [
    "urgent",
    "urgently",
    "now",
    "immediately",
    "hurry",
    "act fast",
    "act now",
    "limited time",
    "expires",
    "expire",
    "today only",
    "claim now",
    "last chance"
]

    if any(word in text for word in urgency_words):
        indicators.append("⚠️ Urgency or pressure language")

    if re.search(r"http\S+|www\S+", text):
        indicators.append("🔗 Contains a web link")

    if re.search(r"\b\d{7,}\b", text):
        indicators.append("📱 Contains a phone number")

    contact_words = [
        "call", "reply", "click", "contact"
    ]

    if any(word in text for word in contact_words):
        indicators.append("📞 Call / click / contact request")

    return indicators
def predict_message(message):

    cleaned_message = clean_text(message)

    # Convert text into TF-IDF features
    message_features = features.transform(
        [cleaned_message]
    )

    # SVM decision score
    score = svm_model.decision_function(
        message_features
    )[0]

    # Apply our selected threshold
    prediction = int(
        score >= threshold
    )

    return prediction, score
def get_risk_level(score, threshold):
    distance = score - threshold

    if distance >= 1.0:
        return "🔴 HIGH RISK", "This message is strongly classified as Spam."
    elif distance >= 0:
        return "🟠 MEDIUM RISK", "This message is classified as Spam."
    elif distance >= -1.0:
        return "🟡 LOW RISK", "This message is classified as Ham, but shows some similarity to Spam."
    else:
        return "🟢 VERY LOW RISK", "This message is strongly classified as Ham."


# ============================================================
# USER INTERFACE
# ============================================================

st.title("🛡️ AI-Powered Spam Message Detector")

st.markdown(
    """
    ### 🔐 Intelligent SMS Security System
    Detect suspicious messages using **NLP + Machine Learning**.

    **TF-IDF + Linear SVM** | Explainable Prediction | **99%+ Test Accuracy**
    """
)

st.divider()
st.write(
    "Enter an SMS or message below and the AI model "
    "will predict whether it is **Ham (Safe)** or **Spam**."
)

st.divider()


# Message input
st.subheader("💬 Enter Your Message")
st.caption(
    "Paste an SMS or message below and let the AI model check whether it is Spam or Ham."
)
message = st.text_area(
    "📩 Enter your message:",
    height=150,
    placeholder="Example: Congratulations! You have won a prize. Click here..."
)
st.markdown("**🧪 Quick Test Examples**")

col1, col2 = st.columns(2)

with col1:
    st.info(
        "🚨 Spam Example\n\n"
        "Congratulations! You have won a cash prize. "
        "Click here to claim your reward."
    )

with col2:
    st.success(
        "✅ Ham Example\n\n"
        "Hey, are we still meeting at 5 PM today?"
    )

# Predict button
if st.button(
    "🔍 Check Message",
    use_container_width=True
):

    if not message.strip():

        st.warning(
            "⚠️ Please enter a message first."
        )

    else:

        prediction, score = predict_message(
            message
        )
        risk_label, risk_description = get_risk_level(score, threshold)

        st.info(f"**{risk_label}** — {risk_description}")
        

        # ====================================================
        # SPAM RESULT
        # ====================================================

        

        st.divider()
        st.subheader("🤖 AI Prediction Result")
        
        if prediction == 1:

            st.error(
                "🚨 SPAM MESSAGE DETECTED"
            )
            # Model Decision Score
            st.subheader("📈 Model Decision Score")

            st.write(f"**Decision Score:** `{score:.2f}`")
            st.write(f"**Classification Threshold:** `{threshold:.2f}`")

            st.caption(
    "A score above the threshold indicates a Spam prediction."
)
            st.write(
                "This message shows characteristics "
                "commonly associated with spam."
            )

            # Show possible reasons
            indicators = get_spam_indicators(message)

            if indicators:

                st.subheader("🔎 Possible Spam Indicators")

                for indicator in indicators:
                    st.write(f"- {indicator}")

            else:

                st.info(
                    "The model detected spam based on "
                    "learned language patterns."
                )


        # ====================================================
        # HAM RESULT
        # ====================================================

        else:

            st.success(
                "✅ MESSAGE LOOKS SAFE"
            )
            # Model Decision Score
              
            st.subheader("📈 Model Decision Score")

            st.write(f"**Decision Score:** `{score:.2f}`")
            st.write(f"**Classification Threshold:** `{threshold:.2f}`")

            st.caption(
    "A score below the threshold indicates a Ham prediction."
)
            st.write(
                "This message appears to be a normal "
                "legitimate message."
            )

        st.info(
            "The model detected spam based on learned "
            "language patterns."
        )    

                # ====================================================
        # SPAM / HAM RESULT
        # ====================================================

        

        

# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "AI-Powered Spam Message Detector | "
    "TF-IDF + Linear SVM"
) 
# ============================================================
# SPAM INDICATORS
# ============================================================

def get_spam_indicators(message):

    text = message.lower()

    indicators = []

    # Prize / reward
    prize_words = [
        "won", "winner", "prize", "reward",
        "cash", "free", "congratulations"
    ]

    if any(word in text for word in prize_words):
        indicators.append("🎁 Prize / reward related language")

    # Money
    money_words = [
        "money", "cash", "₹", "rs", "rupees",
        "pounds", "dollars"
    ]

    if any(word in text for word in money_words):
        indicators.append("💰 Money-related language")
        
    # Urgency
    urgency_words = [
    "urgent",
    "urgently",
    "hurry",
    "immediately",
    "act now",
    "limited time",
    "expires",
    "expire",
    "today only",
    "claim now",
    "last chance"
    ]
    

    if any(word in text for word in urgency_words):
        indicators.append("⚠️ Urgency or pressure language")

    # Links
    if re.search(r"http\S+|www\S+", text):
        indicators.append("🔗 Contains a web link")

    # Contact / phone number
    if re.search(r"\b\d{7,}\b", text):
        indicators.append("📱 Contains a phone number")

    # Call / reply
    contact_words = [
        "call", "reply", "click", "contact"
    ]

    if any(word in text for word in contact_words):
        indicators.append("📞 Call / click / contact request")

    return indicators

# ============================================================
# MODEL COMPARISON
# ============================================================

st.divider()

st.subheader("📊 Model Performance Comparison")

model_results = {
    "Logistic Regression": 95.96,
    "Naive Bayes": 96.95,
    "Linear SVM": 98.12,
    "Improved Linear SVM": 99.19,
    "Improved SVM (Tuned)": 99.10
}

st.write(
    "Different machine learning models were evaluated "
    "on the SMS spam dataset."
)

for model_name, accuracy in model_results.items():

    st.write(
        f"**{model_name}** — {accuracy:.2f}%"
    )

    st.progress(
        accuracy / 100
    )
st.bar_chart(model_results)
st.success(
    "🏆 Best Model: Improved Linear SVM — 99.19% Accuracy"
)

# ============================================================
# CONFUSION MATRIX - ACTUAL TEST PREDICTIONS
# ============================================================

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

st.divider()
st.subheader("📉 Confusion Matrix")

st.write(
    "This heatmap shows the actual test performance "
    "of the tuned Linear SVM model."
)

# Load actual test labels and saved predictions
y_test = joblib.load("models/y_test.joblib")
test_predictions = joblib.load("models/y_test_predictions.joblib")

# Calculate confusion matrix from saved predictions
cm = confusion_matrix(
    y_test,
    test_predictions,
    labels=[0, 1]
)

# Display heatmap
fig, ax = plt.subplots(figsize=(7, 4.5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    linewidths=2,
    linecolor="white",
    xticklabels=["Ham", "Spam"],
    yticklabels=["Ham", "Spam"],
    annot_kws={"size": 16, "weight": "bold"},
    ax=ax
)

ax.set_xlabel("Predicted Label", fontsize=12, fontweight="bold")
ax.set_ylabel("Actual Label", fontsize=12, fontweight="bold")
ax.set_title("Spam Detection Confusion Matrix", fontsize=14, fontweight="bold")

plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

# Calculate performance metrics safely
tn, fp, fn, tp = cm.ravel()

total = cm.sum()
accuracy = (tn + tp) / total * 100 if total else 0
spam_precision = tp / (tp + fp) * 100 if (tp + fp) else 0
spam_recall = tp / (tp + fn) * 100 if (tp + fn) else 0

col1, col2, col3 = st.columns(3)

col1.metric("Accuracy", f"{accuracy:.2f}%")
col2.metric("Spam Precision", f"{spam_precision:.2f}%")
col3.metric("Spam Recall", f"{spam_recall:.2f}%")

st.markdown("### 📖 Understanding the Results")

col1, col2 = st.columns(2)

with col1:
    st.success(
        f"✅ Correct Ham: {tn}\n\n"
        f"✅ Correct Spam: {tp}"
    )

with col2:
    st.warning(
        f"⚠️ Ham incorrectly flagged as Spam: {fp}\n\n"
        f"⚠️ Spam missed by the model: {fn}"
    )

st.caption(
    "These metrics are calculated from saved test labels "
    "and predictions using the tuned model threshold."
)