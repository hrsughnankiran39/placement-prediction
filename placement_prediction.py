#%%
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import LabelEncoder
from tabulate import tabulate
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score
#%%
df=pd.read_csv('dataset.csv')
df
#%%
df.info()
#%%
#removing un-necessary column
df=df.drop(columns='StudentID')
df
#%%
df.groupby('PlacementStatus').size().plot(kind='pie')
plt.title('Placement Distribution')
#%%
#replacing yes/no with boolean value 0/1
from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()
object_cols = df.select_dtypes(include=['object']).columns
for column in object_cols:
    df[column]=labelencoder.fit_transform(df[column])
df
#%%
df.isnull().sum() # for checking nulls
#%%
df.describe()
#%%
# Visualizating effect of internships on placement
Internships_effect = df.pivot_table(index = 'PlacementStatus',values="CGPA", columns='Internships', aggfunc='count')
Internships_effect
#%%
X = Internships_effect.columns
X_axis = np.arange(len(X))

plt.bar(X_axis - 0.2, Internships_effect.iloc[0,:], 0.4, label = 'NotPlaced')
plt.bar(X_axis + 0.2, Internships_effect.iloc[1,:], 0.4, label = 'Placed')

plt.xticks(X_axis, X)
plt.xlabel("Number of internships")
plt.ylabel("Number of placements")
plt.title("internships effect")
plt.legend()
plt.show()
#%%
# Effect of CGPA and softskills
CGPA_effect = df.pivot_table(index = 'PlacementStatus', values= ['CGPA', 'SoftSkillsRating'])
CGPA_effect
# Shows that higher cgpa and softskillSSLCore are more probable to land student a placement
#%%
plt.figure(figsize=(15,5))
sns.heatmap(df.corr(), annot=True)
#%%
sns.set(style='whitegrid')
sns.histplot(data=df, x='CGPA', hue='PlacementStatus', multiple='stack')
plt.legend(title='Placement Status', labels=['Placed', 'Not Placed'])
#%%
sns.countplot(data=df,x='CGPA')
plt.xticks(rotation=90)
plt.title('CGPA Analysis')
#%%
sns.countplot(data=df.loc[(df.PlacementStatus==1)],x='CGPA')
plt.xticks(rotation=90)
plt.title('CGPA wise Placement')
#%%
sns.displot(df['PUC_Marks'])
plt.title('High School Marks')
#%%
sns.countplot(data=df.loc[(df.PlacementStatus==1)],x='PUC_Marks')
plt.xticks(rotation=90)
plt.title('High School Marks wise Placement')
#%%
sns.countplot(data=df.loc[(df.PlacementStatus==1)],x='ExtracurricularActivities')
plt.xticks(rotation=90)
plt.title('ExtracurricularActivities wise Placement')
#%%
sns.countplot(data=df.loc[(df.PlacementStatus==1)],x='SoftSkillsRating')
plt.xticks(rotation=90)
plt.title('Softskills wise Placement')
#%%
sns.countplot(data=df.loc[(df.PlacementStatus==1)],x='Projects')
plt.xticks(rotation=90)
plt.title('Projects wise Placement')
#%%
sns.countplot(data=df.loc[(df.PlacementStatus==1)],x='AptitudeTestScore')
plt.xticks(rotation=90)
plt.title('AptitudeTestScore wise Placement')
#%%
X=df.drop(columns='PlacementStatus')   #all columns except placement status
y=df.PlacementStatus          #placement status
#%%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)
#%% md
# # XGBoost model
#%%
# Create a list to store accuracies
accuracies = []

# Create XGBoost model
xgb_model = XGBClassifier(n_estimators=1000, learning_rate=0.05, objective="binary:logistic", eval_metric="logloss", random_state=1)

# Fit the model
xgb_model.fit(X_train, y_train)

# Make predictions
y_pred_xgb = xgb_model.predict(X_test)

# Evaluate the model
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
print(f"XGBoost Accuracy: {accuracy_xgb}")
accuracies.append({'Model': 'XGBoost', 'Accuracy': accuracy_xgb})

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_xgb, normalize='true')
target_names = ['NotPlaced', 'Placed']

# Display the confusion matrix
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names).plot(cmap='Blues')
plt.title('XGBoost Model') 
#%% md
# # Logistic Regression
#%%
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,ConfusionMatrixDisplay
lr=LogisticRegression(max_iter=50000,penalty=None)
lr.fit(X_train,y_train)
prediction=lr.predict(X_test)
accuracy = accuracy_score(prediction, y_test)
print(f"Logistic Regression Accuracy: {accuracy}")
accuracies.append({'Model': 'Logistic Regression', 'Accuracy': accuracy})

target_names=['NotPlaced','Placed']
print(classification_report(y_test,prediction,target_names=target_names))
cm=confusion_matrix(y_test,prediction,normalize='true')
ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=target_names).plot(cmap='Reds')
plt.title('Logistic Regression')
#%% md
# # Decision Tree model
#%%
# Create Decision Tree model
dt_model = DecisionTreeClassifier(random_state=1)

# Fit the model
dt_model.fit(X_train, y_train)

# Make predictions
y_pred_dt = dt_model.predict(X_test)

# Evaluate the model
accuracy_dt = accuracy_score(y_test, y_pred_dt)
print(f"Decision Tree Accuracy: {accuracy_dt}")
accuracies.append({'Model': 'Decision Tree', 'Accuracy': accuracy_dt})


# Confusion Matrix
cm_dt = confusion_matrix(y_test, y_pred_dt, normalize='true')

# Display the confusion matrix
ConfusionMatrixDisplay(confusion_matrix=cm_dt, display_labels=target_names).plot(cmap='Greens')
plt.title('Decision Tree')
#%% md
# # Naive Bayes model
#%%
# Create Naive Bayes model (Gaussian Naive Bayes)
nb_model = GaussianNB()

# Fit the model
nb_model.fit(X_train, y_train)

# Make predictions
y_pred_nb = nb_model.predict(X_test)

# Evaluate the model
accuracy_nb = accuracy_score(y_test, y_pred_nb)
print(f"Naive Bayes Accuracy: {accuracy_nb}")
accuracies.append({'Model': 'Naive Bayes', 'Accuracy': accuracy_nb})

# Confusion Matrix
cm_nb = confusion_matrix(y_test, y_pred_nb, normalize='true')

# Display the confusion matrix

ConfusionMatrixDisplay(confusion_matrix=cm_nb, display_labels=target_names).plot(cmap='Blues')
plt.title('Naive Bayes')
#%% md
# # Support vector machines (SVM)
#%%
from sklearn.svm import SVC

svm_model = SVC(kernel='linear', random_state=1)

# Fit the model
svm_model.fit(X_train, y_train)

# Make predictions
y_pred_svm = svm_model.predict(X_test)

# Evaluate the model
#svm best in case of overfitting model
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print(f"SVM Accuracy: {accuracy_svm}")
accuracies.append({'Model': 'Support Vector Machine', 'Accuracy': accuracy_svm})
# Confusion Matrix
cm_svm = confusion_matrix(y_test, y_pred_svm, normalize='true')

# Display the confusion matrix
ConfusionMatrixDisplay(confusion_matrix=cm_svm, display_labels=target_names).plot(cmap='Blues')
plt.title('SVM')
#%%
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# Assuming you have models named cgb_model, lr_model, dt_model, nb_model, svm_model, lgb_model
models = {'XG-Boost': xgb_model, 'Logistic Regression': lr, 'Decision Trees': dt_model, 'Naive Bayes': nb_model, 'SVM': svm_model}

# Create a list to store accuracies
accuracies = []

# Iterate over models
for model_name, model in models.items():
    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    
      # Precision, Recall, F1 Score, Specificity
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Calculate specificity
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp)

    # Append metrics to the accuracies list
    accuracies.append({
        'Model': model_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'Specificity': specificity
    })
#%%
from termcolor import colored  # Add this import statement

accuracy_df = pd.DataFrame(accuracies)

# Remove duplicate rows based on the 'Model' column
accuracy_df = accuracy_df.drop_duplicates(subset='Model', keep='first')

# Display the accuracies in a beautified and colored tabular form
headers = accuracy_df.columns
data = accuracy_df.values  # Make sure 'data' is defined here

colored_table = [[colored(cell, 'blue', attrs=['bold']) if i == 0 else cell for i, cell in enumerate(row)] for row in data]
print(tabulate(colored_table, headers=headers, tablefmt='fancy_grid', showindex=False))

# Define the input features in the same order as the model was trained
feature_names = ['CGPA', 'Internships', 'Projects', 'Workshops/Certifications', 'AptitudeTestScore',
                 'SoftSkillsRating', 'ExtracurricularActivities', 'PlacementTraining', 'SSLC_Marks', 'PUC_Marks']

# Function to get input from the user
def get_user_input():
    user_input = []
    for feature in feature_names:
        value = float(input(f"Enter {feature}: "))
        user_input.append(value)
    return np.array(user_input).reshape(1, -1)

# Get user input
user_data = get_user_input()

# Make prediction
placement_prediction = svm_model.predict(user_data)

# Print the prediction
if placement_prediction[0] == 1:
    print("The model predicts that the student will be placed.")
else:
    print("The model predicts that the student will not be placed.")
#%%
import pickle
# output = open('data.pkl', 'wb')
# Pickle dictionary using protocol 0.
filename = 'savedmodel.sav'
pickle.dump(svm_model, open(filename,'wb'))
#%%
load_model = pickle.load(open(filename,'rb'))
load_model.predict([[8.9,0,3,2,90,4.0,1,1,78,82]])
#%%
import lightgbm as lgb
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Load your dataset
df=pd.read_csv('C:/Users/Pavan T S/Downloads/College_Placement_Prediction/dataset.csv')

# Preprocess your data: Convert categorical variables to one-hot encoding
categorical_cols = ['ExtracurricularActivities', 'PlacementTraining']
df_encoded = pd.get_dummies(df, columns=categorical_cols)

# Define features (X) and target variable (y)
X = df_encoded[['CGPA', 'Internships', 'Projects', 'Workshops/Certifications', 'AptitudeTestScore', 'SoftSkillsRating', 'SSLC_Marks', 'PUC_Marks',
                'ExtracurricularActivities_No', 'ExtracurricularActivities_Yes', 'PlacementTraining_No', 'PlacementTraining_Yes']]
y = df_encoded['PlacementStatus']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Create and train the LightGBM model
clf = lgb.LGBMClassifier()
clf.fit(X_train, y_train)

# Access feature importance
feature_importance = clf.feature_importances_

# Map feature importance to feature names
feature_names = X.columns
feature_importance_dict = dict(zip(feature_names, feature_importance))

# Sort and visualize feature importance
sorted_feature_importance = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)

# Plotting
features, importance = zip(*sorted_feature_importance)
plt.barh(features, importance)
plt.xlabel('Feature Importance')
plt.ylabel('Feature Names')
plt.title('Feature Importance Plot')
plt.show()

#%%
