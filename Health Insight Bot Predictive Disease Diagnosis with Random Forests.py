#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import csv


# In[2]:


train_df=pd.read_csv('Train.csv')
test_df=pd.read_csv('Test.csv')


# In[3]:


train_df

Exploratory Data Analysis
# In[4]:


train_df.shape #Number of rows and columns


# In[5]:


train_df.describe()  # Description about dataset


# In[6]:


train_df.info() # Information about Dataset

Checking Null values present in the Train dataset or not.
# In[7]:


train_df.isnull().sum()

Take the Target Variable.
# In[8]:


cols= train_df.columns
cols= cols[:-1]

# x stores every column data except the last one
x = train_df[cols]

# y stores the target variable for disease prediction
y = train_df['prognosis']

Distribution of Diseases
# In[9]:


import matplotlib.pyplot as plt
import seaborn as sns

# Check the count of each class
class_counts = train_df['prognosis'].value_counts()
print("Class distribution:\n", class_counts)

# Distribution of the target variable (prognosis)
plt.figure(figsize=(10, 20))
sns.countplot(y='prognosis', data=train_df, order=train_df['prognosis'].value_counts().index)
plt.title('Distribution of Prognosis')
plt.xlabel('Count')
plt.ylabel('Prognosis')
plt.show()


# In[ ]:




Correlation Analysis : correlation heatmap will show which symptoms are correlated
# In[10]:


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Calculate the correlation matrix for the symptoms
correlation_matrix = train_df.iloc[:, :-1].corr()
correlation_matrix


# In[11]:


# Plot the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=False,cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap of Symptoms')
plt.show()


# In[ ]:




Common Symptoms Across All Diseases :visualization will show the most common symptoms in the dataset, indicating their importance.
# In[12]:


# Calculate the frequency of each symptom across the dataset
symptom_frequency = train_df.iloc[:, :-1].mean()

# Plot the top 20 most frequent symptoms
plt.figure(figsize=(10, 6))
symptom_frequency.sort_values(ascending=False).head(20).plot(kind='bar', color='steelblue')
plt.title('Top 20 Most Frequent Symptoms')
plt.xlabel('Symptoms')
plt.ylabel('Frequency (Proportion of Cases)')
plt.xticks(rotation=90)
plt.show()


# In[ ]:




Symptom Presence for Each Diagnosis: We can also explore the relationship between a specific diagnosis and the symptoms associated with it. This heatmap will show the average presence of each symptom for each diagnosis.
# In[13]:


# Grouping by prognosis and calculating the mean presence of each symptom
reduced_data = train_df.groupby(train_df['prognosis']).max()
reduced_data


# In[14]:


# Heatmap showing the mean symptom presence for each diagnosis
plt.figure(figsize=(20, 15))
sns.heatmap(reduced_data, cmap='coolwarm', linewidths=0.5)
plt.title('Mean Symptom Presence for Each Diagnosis')
plt.show()


# In[ ]:




Pairplot for Selected Symptoms and Prognosis: Visualizing relationships between individual symptoms and prognosis using pairplots (though limited due to binary variables) can still reveal trends.
# In[15]:


# Selecting a few symptoms and the target for pairplot
selected_symptoms = ['itching', 'skin_rash', 'nodal_skin_eruptions', 'continuous_sneezing']
sns.pairplot(train_df[selected_symptoms + ['prognosis']], hue='prognosis', diag_kind='kde')
plt.show()


# In[ ]:




Feature Selection and Correlation Analysis
# In[16]:


from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder

# Encode the prognosis column
le = preprocessing.LabelEncoder()

# Fit the label encoder to the target variable 'y' and transform it
le.fit(y)
y = le.transform(y)


# In[17]:


from sklearn.feature_selection import chi2

X = train_df.drop(columns=['prognosis'])
y = train_df['prognosis']

chi_scores = chi2(X, y)

# Create a dataframe for better visualization
chi2_df = pd.DataFrame({'Symptom': X.columns, 'Chi-Square Score': chi_scores[0]})
chi2_df = chi2_df.sort_values(by='Chi-Square Score', ascending=False)

chi2_df.head(10)  # Display top 10 symptoms by chi-square score

Features such as receiving_blood_transfusion, increased_appetite, and polyuria have the highest Chi-Square scores, indicating a strong association with the target variable. These features are more important for prediction and should be given priority in modeling
# In[ ]:




Dimensionality Reduction with PCA:

Principal Component Analysis (PCA) is used to reduce the dimensionality of your dataset while retaining as much variance as possible.
# In[18]:


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialize PCA
pca = PCA(n_components=4)  # Reduce to 4 dimensions
X_pca = pca.fit_transform(X_scaled)

# Create a DataFrame for the PCA results
# Adjust column names to match the number of components
pca_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2', 'PC3', 'PC4'])
pca_df['Prognosis'] = y.reset_index(drop=True)  # Adding target variable for color coding

# Plotting first 2 components
plt.figure(figsize=(10, 7))
sns.scatterplot(x='PC1', y='PC2', hue='Prognosis', data=pca_df, palette='viridis')
plt.title('PCA of Symptoms')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()


# In[19]:


print(pca_df)

Model : Random ForestFeature Importance (Using Random Forest) : Train a simple random forest model to calculate feature importance and see which symptoms are most predictive of the prognosis.
# In[20]:


from sklearn.model_selection import train_test_split

# Define features (X) and target (y)
X = train_df.drop(columns=['prognosis'])
y = train_df['prognosis']

# Split the dataset into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# In[21]:


from sklearn.ensemble import RandomForestClassifier

# Initialize the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Predict on the test set
y_pred = rf_model.predict(X_test)


# In[22]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Random Forest Accuracy: {accuracy * 100:.2f}%")

# Display classification report
print("Classification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
import seaborn as sns
import matplotlib.pyplot as plt

conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.show()


# In[23]:


# Feature Importance
feature_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

# Display top 10 important features
print(feature_importances.head(10))


# In[24]:


# Plot the feature importances
plt.figure(figsize=(12, 8))
plt.barh(feature_importances['Feature'], feature_importances['Importance'])
plt.xlabel('Feature Importance')
plt.title('Feature Importances from Random Forest')
plt.gca().invert_yaxis()  # Invert y-axis to display the most important features at the top
plt.show()


# In[25]:


from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

# Fit grid search
grid_search.fit(X_train, y_train)

# Best parameters
print("Best Parameters: ", grid_search.best_params_)


# In[26]:


# Re-evaluate the model with the best found parameters (Optional)
best_rf_model = grid_search.best_estimator_
y_pred_best = best_rf_model.predict(X_test)

# Calculate accuracy with tuned model
best_accuracy = accuracy_score(y_test, y_pred_best)
print(f"Optimized Random Forest Accuracy: {best_accuracy * 100:.2f}%")

Evaluate with Cross-Validation: Check how the model performs on cross-validation.
# In[27]:


# Perform cross-validation on the optimized model
cv_scores = cross_val_score(best_rf_model, X, y, cv=5, scoring='accuracy')
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean cross-validation accuracy: {cv_scores.mean():.4f}")


# In[28]:


import joblib

# Save the model
joblib.dump(best_rf_model, 'optimized_random_forest_model.pkl')

Feature Importance Analysis: Since we have 100% accuracy, analyzing which features (or symptoms) are contributing most to the predictions can be very insightful.
# In[29]:


X.shape


# In[30]:


train_df.shape


# In[31]:


# Display feature importances
feature_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': best_rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("Top Important Features:\n", feature_importances.head(50))

Feature Importance Visualization: We can visualize the feature importance to get a better sense of how different symptoms influence the predictions.
# In[32]:


# Plot feature importances
plt.figure(figsize=(10, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importances.head(20))
plt.title('Top 20 Important Features')
plt.show()

Evaluate Model on Unseen Data:
# In[33]:


y_pred_test = best_rf_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

Create a pipeline that includes PCA and Random Forest
# In[34]:


from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline that includes PCA and Random Forest
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=4)),  # Adjust number of components as needed
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train the pipeline
pipeline.fit(X_train, y_train)

# Make predictions
y_pred = pipeline.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print(classification_report(y_test, y_pred))


# In[ ]:




ChatBot Model for Prediction of Diseases based on symptoms
# In[35]:


# Function to convert text to speech
def text_to_speech(text):
    # Set properties (optional)
    engine.setProperty('rate', 100)    # Speed percent (can go over 100)
    engine.setProperty('volume', 0.9)  # Volume 0-1

    # Convert text to speech
    engine.say(text)
    engine.runAndWait()


# In[36]:


import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import _tree
import pyttsx3
import re
from sklearn.preprocessing import LabelEncoder
from collections import Counter

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# Initialize dictionaries to store symptom severity, description, and precautions
severityDictionary=dict()
description_list = dict()
precautionDictionary=dict()

# Dictionary to map symptoms to their indices
symptoms_dict = {}

# Populate symptoms dictionary with indices
for index, symptom in enumerate(x):
       symptoms_dict[symptom] = index

# Function to calculate the overall severity of the symptom
def calc_condition(exp, days):
    total_severity = sum(severityDictionary[item] for item in exp)
    if (total_severity * days) / (len(exp) + 1) > 13:
        print("It's advisable to consult a doctor for further evaluation.")
    else:
        print("While it may not be severe, taking preventive measures is recommended.")

# Function to read and store symptom descriptions from a CSV file
def getDescription():
    global description_list
    description_list={}
    with open('symptomDescription.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if len(row) >= 2:  # Check if the row has at least 2 columns
                description_list[row[0]] = row[1]
            else:
                print(f"Skipping row due to insufficient columns: {row}")

def getSeverityDict():
    global severityDictionary
    severityDictionary = {}
    with open('sympseverity.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if len(row) >= 2:
                if len(row) >= 2:  # Check if the row has at least 2 columns
                    try:
                        severityDictionary[row[0]] = int(row[1])
                    except ValueError:
                        print(f"Skipping row with invalid severity value: {row}")
                else:
                    print(f"Skipping row due to insufficient columns: {row}")
    
        
# Function to read and store symptom precaution information from a CSV file
def getprecautionDict():
    global precautionDictionary
    precautionDictionary = {}
    with open('precaution.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if len(row) >= 5:  # Check if the row has at least 5 columns
                precautionDictionary[row[0]]=[row[1],row[2],row[3],row[4],row[5],row[6]]
            else:
                print(f"Skipping row due to insufficient columns: {row}")


# In[37]:


def getInfo():
    print("~~~~~~~~~~~~~~~~~~~~~~ Welcome to Health Insight Bot ~~~~~~~~~~~~~~~~~~~~~~~")
    print("\nWhat is Your Name? \t\t\t\t",end="->")
    name=input("")
    print("Hello", name)
    
def check_pattern(dis_list,inp):
    pred_list=[]
    inp=inp.replace(' ','_')
    patt = f"{inp}"
    regexp = re.compile(patt)
    pred_list=[item for item in dis_list if regexp.search(item)]
    return (1, pred_list) if len(pred_list) > 0 else (0, [])


# In[38]:


# Grouping Data by Prognosis and Finding Maximum Values
reduced_data = train_df.groupby(train_df['prognosis']).max()

# Extract column names from the aggregated DataFrame
red_cols = reduced_data.columns


# In[39]:


def print_disease(node):
    node = node[0]
    val  = node.nonzero()
    disease = le.inverse_transform(val[0])
    return list(map(lambda x: x.strip(), list(disease)))


# In[40]:


def sec_predict(symptoms_exp):
    df = pd.read_csv('Train.csv')
    X = df.iloc[:, :-1]
    y = df['prognosis']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)
    rf_clf = RandomForestClassifier(random_state=20)
    rf_clf.fit(X, y)

    symptoms_dict = {symptom: index for index, symptom in enumerate(X)}
    input_vector = np.zeros(len(symptoms_dict))
    for item in symptoms_exp:
        input_vector[[symptoms_dict[item]]] = 1

    return rf_clf.predict([input_vector])


# In[45]:


from sklearn.tree import export_text
import pyttsx3
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import _tree

# Initialize the text-to-speech engine
engine = pyttsx3.init()

def print_tree(tree_model, feature_names):
    tree_ = tree_model.estimators_[0].tree_
    if len(feature_names) != tree_.n_features:
        raise ValueError(f"Feature names list has {len(feature_names)} elements, but the tree expects {tree_.n_features} features.")
    
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    
    tree_text = export_text(tree_model.estimators_[0], feature_names=feature_name)
    print("Decision Tree:")
    print(tree_text)


def tree_to_code(tree_model, feature_names):
    tree_ = tree_model.estimators_[0].tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    chk_dis = ",".join(feature_names).split(",")
    symptoms_present = []
    
    def recurse(node, depth):
        indent = " " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            if name == disease_input:
                val = 1
            else:
                val = 0
            if val <= threshold:
                recurse(tree_.children_left[node], depth + 1)
            else:
                symptoms_present.append(name)
                recurse(tree_.children_right[node], depth + 1)
        else:
            present_disease = print_disease(tree_.value[node])

            red_cols = reduced_data.columns
            symptoms_given = red_cols[reduced_data.loc[present_disease].values[0].nonzero()]
            
            engine.say("Are you currently experiencing any of these symptoms?")
            engine.runAndWait()
            print("Are you currently experiencing any of these symptoms?")
            experienced_symptoms = []
            for syms in list(symptoms_given):
                inp = ""
                engine.say(f"{syms}, are you experiencing it?")
                engine.runAndWait()
                print(syms, "? : ", end='')
                while True:
                    inp = input("")
                    if inp == "yes" or inp == "no":
                        break
                    else:
                        print("Provide proper answers i.e. (yes/no): ", end="")
                if inp == "yes":
                    experienced_symptoms.append(syms)
                    
            second_prediction = sec_predict(experienced_symptoms)
            calc_condition(experienced_symptoms, num_days)
            if present_disease[0] == second_prediction[0]:
                engine.say(f"You may have {present_disease[0]}")
                engine.runAndWait()
                print(f"You may have {present_disease[0]}")
                print(description_list[present_disease[0]])
            else:
                engine.say(f"You may have {present_disease[0]} or {second_prediction[0]}.")
                engine.runAndWait()
                print(f"You may have {present_disease[0]} or {second_prediction[0]}")
                print(description_list[present_disease[0]])
                print(description_list[second_prediction[0]])
            
            precaution_list = precautionDictionary[present_disease[0]]
            print("Take the following measures: ")
            for i, j in enumerate(precaution_list):
                print(i + 1, ")", j)

    while True:
        engine.say("\nEnter the symptom you are experiencing \t\t\t")
        engine.runAndWait()
        print("\nEnter the symptom you are experiencing  \t\t", end="->")
        disease_input = input("")
        
        conf, cnf_dis = check_pattern(chk_dis, disease_input)
        if conf == 1:
            print("searches related to input: ")
            for num, it in enumerate(cnf_dis):
                print(num, ")", it)
            conf_inp = int(input(f"Select the one you meant (0 - {num}):  ")) if num != 0 else 0
            disease_input = cnf_dis[conf_inp]
            break
        else:
            print("Enter valid symptom.")
            
    
    while True:
        try:
            num_days = int(input("Okay. From how many days? : "))
            break
        except:
            print("Enter valid input.")
    
    recurse(0, 1)


# Initialize the RandomForest model
getSeverityDict()
getDescription()
getprecautionDict()
getInfo()

# Train Random Forest Classifier and use it for tree_to_code
df = pd.read_csv('Train.csv')
X = df.iloc[:, :-1]
y = df['prognosis']
rf_clf = RandomForestClassifier(random_state=20)
rf_clf.fit(X, y)

# Use the first tree in the Random Forest for visualization and interpretation
cols = X.columns.tolist()  # Print the decision tree text
tree_to_code(rf_clf, cols)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




