# AI-Powered Spam Message Detector

## 1. Project Overview
This project detects whether a text message is Spam or Ham
using Natural Language Processing (NLP) and Machine Learning.

## 2. Objectives
- Identify spam and legitimate messages.
- Help users recognize unwanted or potentially fraudulent messages.
- Provide a simple web interface for message classification.

## 3. Technologies Used
- Python
- Natural Language Processing (NLP)
- TF-IDF Vectorization
- Linear Support Vector Machine (SVM)
- Scikit-learn
- Streamlit
- Pandas and NumPy
- Joblib

## 4. Dataset
The project uses the SMS Spam Collection dataset.

- Total messages: 5,572
- Ham messages: 4,825
- Spam messages: 747

## 5. Model Performance
The trained model achieved approximately 99.10% accuracy
on the test dataset.

- Spam Precision: 98.60%
- Spam Recall: 94.63%
- True Negatives: 964
- False Positives: 2
- False Negatives: 8
- True Positives: 141

## 6. Project Structure

- app.py: Streamlit web application
- data/: Dataset files
- models/: Saved machine learning model files
- src/: Model training and preprocessing scripts
- notebooks/: Data analysis and experiments
- requirements.txt: Required Python libraries
- README.md: Project documentation

## 7. How to Run the Project

1. Install the required libraries:

   pip install -r requirements.txt

2. Run the Streamlit application:

   streamlit run app.py

3. Open the local URL shown in the terminal.

## 8. Limitations
- Predictions may not always be correct.
- Performance depends on the dataset used for training.
- New spam patterns may require updated training data.

## 9. Future Scope
- Support for multiple languages.
- Training with newer message datasets.
- Integration with email or messaging applications.

## 10. Conclusion
This project demonstrates how NLP and Machine Learning
can be used to classify text messages as Spam or Ham.