## Health Insight Bot

### Overview
The Health Insight Bot is a comprehensive diagnostic tool that leverages machine learning to assess symptoms reported by users and provide disease predictions along with actionable recommendations. Designed for interactive use, it combines data-driven insights with user feedback to offer valuable health advice.

### Objective
The primary aim of the Health Insight Bot is to assist users in identifying potential health conditions based on their symptoms. The tool is designed to:
- **Facilitate Early Diagnosis**: Help users recognize potential diseases early based on symptom patterns.
- **Assess Symptom Severity**: Evaluate how severe symptoms are and suggest whether medical consultation is necessary.
- **Provide Preventive Measures**: Offer practical advice to manage symptoms and reduce health risks.

### Goal
1. **Interactive Symptom Input**:
   - Allow users to enter symptoms they are experiencing.
   - Guide users through a series of questions to clarify and detail their symptoms.
   
2. **Predictive Modeling**:
   - Utilize machine learning algorithms to predict potential diseases based on the input symptoms.
   - Assess symptom severity to provide tailored recommendations.

3. **Actionable Recommendations**:
   - Suggest appropriate precautionary measures based on the predicted disease.
   - Offer practical advice to mitigate symptoms and enhance overall well-being.

### Modeling
1. **Data Collection and Preparation**:
   - **Symptom Data**: Loaded from CSV files, including descriptions, severity scores, and precautionary measures.
   - **Feature Engineering**: Transformed symptom data into feature vectors suitable for machine learning models.

2. **Model Training**:
   - **Algorithm Selection**: Used Random Forest Classifier for its robustness in handling classification tasks.
   - **Training**: Trained the model on a dataset containing symptoms and corresponding diseases.
   - **Validation**: Split data into training and test sets to evaluate model performance.

3. **Decision Tree Visualization**:
   - Implemented visualization techniques to interpret the decision-making process of the Random Forest model.
   - Provided insights into how symptoms lead to specific disease predictions.

### EDA (Exploratory Data Analysis)
1. **Data Inspection**:
   - Explored the dataset to understand the distribution and frequency of symptoms and diseases.
   - Identified patterns and correlations between symptoms and diagnoses.

2. **Feature Analysis**:
   - Analyzed the importance of different features (symptoms) in predicting diseases.
   - Evaluated feature relevance to ensure the model focuses on significant symptoms.

3. **Data Cleaning**:
   - Addressed inconsistencies and missing values in the dataset.
   - Ensured data quality and reliability for accurate predictions.

### Result
- **Disease Prediction**:
   - Successfully predicts potential diseases based on user-reported symptoms.
   - Provides a high level of accuracy in disease classification using machine learning.

- **Severity Calculation**:
   - Assesses the severity of symptoms to determine whether professional medical advice is needed.
   - Offers recommendations based on severity levels, guiding users on when to seek further evaluation.

- **Precaution Recommendations**:
   - Lists precautionary measures tailored to the predicted condition.
   - Provides actionable steps to manage symptoms and prevent complications.
![result](https://github.com/user-attachments/assets/dd6acc50-95c1-4090-96fa-9d2640bb151f)
![result2](https://github.com/user-attachments/assets/fae21ab4-2bc8-4452-b474-7dfaa1bafe9a)

---
