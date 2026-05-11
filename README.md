# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Start 2.Import required libraries. 3.Create or load the student dataset (CGPA, IQ, Placement Status). 4.Separate input features (CGPA, IQ) and output label (Placement). 5.Split the dataset into training data and testing data. 6.Create the Logistic Regression model. 7.Train the model using training data. 8.Test the model using testing data. 9.Calculate accuracy of the model. 10.Get CGPA and IQ from the user. 11.Predict whether the student is placed or not placed. 12.Display the prediction result. 13.Stop. 

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
## Developed by: MAHALAKSHMI P
## RegisterNumber: 212225100024
*/
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv(r"Placement_Data.csv")

df = df.drop(["sl_no", "salary"], axis=1)

df = df.dropna()
df = df.drop_duplicates()


le = LabelEncoder()
categorical_columns = [
    "gender", "ssc_b", "hsc_b", "hsc_s",
    "degree_t", "workex", "specialisation", "status"
]

for col in categorical_columns:
    df[col] = le.fit_transform(df[col])

X = df.drop("status", axis=1)
y = df["status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

model = LogisticRegression(solver="liblinear")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

cm = confusion_matrix(y_test, y_pred)

report = classification_report(y_test, y_pred)

print("        LOGISTIC REGRESSION CLASSIFICATION REPORT")
print(report)
print(f"Accuracy of the Model : {accuracy:.4f}")

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Placement Prediction")
plt.show()

sample_input = pd.DataFrame(
    [[1, 80, 1, 90, 1, 1, 90, 1, 0, 85, 1, 85]],
    columns=X.columns
)

sample_prediction = model.predict(sample_input)
print("\nSample Prediction:", sample_prediction)
```

## Output:
<img width="654" height="622" alt="{36244723-86F9-4401-BEAB-FEEB7BDFCA39}" src="https://github.com/user-attachments/assets/b483336c-a2be-4298-90eb-e1a7da3488d2" />



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
