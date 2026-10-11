import pandas as pd

# Dataset load karo
data = pd.read_csv(
    "data/SMSSpamCollection",
    sep="\t",
    header=None,
    names=["label", "message"]
)

# Total messages
print("Total messages:", len(data))

# First 5 messages
print("\nFirst 5 messages:")
print(data.head())

# Spam aur Ham count
print("\nClass distribution:")
print(data["label"].value_counts())