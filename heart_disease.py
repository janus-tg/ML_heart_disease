import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('cleveland.csv')

columnTitle = ['age', 'sex', 'cPain', 'restBp', 'chol',
              'fbs', 'restEcg', 'maxHr', 'exang', 
              'stElev', 'slope', 'vessel', 'thal', 'target']

df.columns = columnTitle

# age: age in years
# sex: (1 = male; 0 = female)
# cPain: chest pain type
# restBp: resting blood pressure (in mm Hg on admission to the hospital)
# chol: serum cholesterol in mg/dl
# fbs: (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)
# restEcg: resting electrocardiographic results
# maxHr: maximum heart rate achieved
# exang: exercise induced angina (1 = yes; 0 = no)
# stElev: ST depression induced by exercise relative to rest
# slope: the slope of the peak exercise ST segment
# vessel: number of major vessels (0–3) colored by fluoroscopy
# thal: thalassemia (3 = normal; 6 = fixed defect; 7 = reversible defect)
# target: (1= heart disease or 0= no heart disease)

print(df.isnull().sum()) 
#checking for null values in the DataFrame.
#4 NA values in vessel and 2 in thal column

#using imputation and filling the NA cells with the mean of the data frame
thal = df["thal"]
vessel = df["vessel"]
thalMean = thal.mean()
vesselMean = vessel.mean()
thal.fillna(thalMean, inplace = True)
vessel.fillna(vesselMean, inplace = True)

#plotting graphs between relevant parameters to observe correlations

#plotting a correlation matrix
corrMatrix = df.corr()
plt.matshow(corrMatrix)
plt.xticks(range(len(corrMatrix.columns)), corrMatrix.columns, fontsize = 11,  rotation = 45);
plt.yticks(range(len(corrMatrix.columns)), corrMatrix.columns, fontsize = 11);
plt.colorbar()
plt.suptitle("Correlation Matrix", fontsize = '12')
plt.show()

#plotting a relationship between age and people affected by heart disease
df['target'] = df.target.map({0: 0, 1: 1, 2: 1, 3: 1, 4: 1})
sns.catplot(x = "age", hue = "target", data = df, kind = 'count', legend = False, palette = 'dark')
sns.set_style('dark')
plt.legend(['No heart disease', 'Heart Disease'], loc = 1)
plt.suptitle("Relationship between age and people affected by heart disease", fontsize = '12')
plt.ylabel("Frequency")
plt.xlabel("Age (in yrs)")
plt.show()

#plotting a relation between male and female affected by disease
affByDis = df[df["target"] == 1]
pieData = affByDis.groupby(["sex"])["target"].count()
plt.pie(pieData, explode = [0.1, 0], labels = ["female", "male"], shadow = True)
plt.suptitle("Relationship between male and female affected by heart disease", fontsize = '12')
plt.show()

#plotting chol vs age
sns.catplot(x = "age", y = "chol", data = df, kind = "boxen")
plt.suptitle("Relationship between age and cholestrol levels", fontsize = '12')
plt.axhline(y = 200, color = 'black')
plt.show()

#plot vessel by age
vesAge = df.groupby(["age"])["vessel"].mean()
vesAge.plot()
plt.suptitle("Relationship between age and mean of vessels affected", fontsize = '12')
plt.ylabel("Mean blood vessels affected")
plt.show()

#plot vessel by gender
vesGend = df.groupby(["sex"])["vessel"].mean()
vesGend.plot(color = "g", kind = 'bar')
plt.suptitle("Relationship between gender and mean of vessels affected", fontsize = '12')
plt.ylabel("Mean blood vessels affected")
plt.xticks([0, 1], ["female", "male"], rotation = 45)
plt.show()

#thal vs gender
thalGend = df[(df["thal"] == 6 )|(df["thal"] == 7)]
sns.catplot(y = "sex", kind = "count", hue = "thal", data = thalGend, legend = False)
plt.suptitle("Relationship between gender and thalassemia", fontsize = '12')
plt.xlabel("Frequency")
plt.ylabel("Gender")
plt.yticks([0, 1], ["female", "male"])
plt.legend(['fixed defect', 'reversible defect'], loc = "best")
plt.show()

#chest pain type vs target
sns.catplot(x = "cPain", hue = "target", data = df, kind = 'count', legend = False, palette = 'dark')
sns.set_style('dark')
plt.legend(['No heart disease', 'Heart Disease'], loc = "best")
plt.xticks([0, 1, 2, 3], ['typical angina', 'atypical angina', 'non-anginal pain', 'asymptomatic'])
plt.suptitle("Relationship between chest-pain type and people affected by heart disease", fontsize = '12')
plt.ylabel("Frequency")
plt.xlabel("Chest-pain type")
plt.show()

#making the predictive algorithms using the scikitLearn module

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

#1. Using a linear regression model

from sklearn.linear_model import LogisticRegression
X = df.loc[:, :"thal"]
y = df.loc[:, "target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
reg = LogisticRegression(max_iter = 1000)

#training
reg.fit(X_train, y_train)

#predicting 
y_pred = reg.predict(X_test)
conMat = confusion_matrix(y_test, y_pred)
print("Accuracy of the logistic regression model:", (conMat[0][0] + conMat[1][1])/(conMat[0][0] + conMat[0][1] + conMat[1][0]+ conMat[1][1]) * 100)
print()
sns.heatmap(conMat, annot = True, annot_kws = {"size": 10})
plt.suptitle("Confusion Matrix for Logistic Regression", fontsize = '12')
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.show()

#2. SVM
from sklearn import svm

#create a classifier
sVec = svm.SVC(kernel = "linear")

#training
sVec.fit(X_train, y_train)

#predicting
y_pred = sVec.predict(X_test)
conMat = confusion_matrix(y_test, y_pred)
print("Accuracy of the SVM model:", (conMat[0][0] + conMat[1][1])/(conMat[0][0] + conMat[0][1] + conMat[1][0]+ conMat[1][1]) * 100)
print()
sns.heatmap(conMat, annot = True, annot_kws = {"size": 10})
plt.suptitle("Confusion Matrix for Support Vector Machines", fontsize = '12')
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.show()

#3. Naive Bayes
from sklearn.naive_bayes import GaussianNB

#creating the model
nBayes = GaussianNB()

#training
nBayes.fit(X_train, y_train)

#predicting
y_pred = nBayes.predict(X_test)
conMat = confusion_matrix(y_test, y_pred)
print("Accuracy of the Naive Bayes model:", (conMat[0][0] + conMat[1][1])/(conMat[0][0] + conMat[0][1] + conMat[1][0]+ conMat[1][1]) * 100)
print()
sns.heatmap(conMat, annot = True, annot_kws = {"size": 10})
plt.suptitle("Confusion Matrix for Naive Bayes", fontsize = '12')
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.show()

#4. Decision Tree
from sklearn.tree import DecisionTreeClassifier

#creating the model
decTree = DecisionTreeClassifier()

# training
decTree.fit(X_train, y_train)

#predicting
y_pred = decTree.predict(X_test)
conMat = confusion_matrix(y_test, y_pred)
print("Accuracy of the Decision Tree model:", (conMat[0][0] + conMat[1][1])/(conMat[0][0] + conMat[0][1] + conMat[1][0]+ conMat[1][1]) * 100)
print()
sns.heatmap(conMat, annot = True, annot_kws = {"size": 10})
plt.suptitle("Confusion Matrix for Decision Tree", fontsize = '12')
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.show()

#calculating and plotting the importance of each variable

#for logistic regression
importance = reg.coef_[0]
sns.barplot(x = importance, y = columnTitle[:13])
sns.set(style = "darkgrid")
plt.suptitle("Feature importance for logistic regression")
plt.show()

#for SVM
importance = sVec.coef_[0]
sns.barplot(x = importance, y = columnTitle[:13])
sns.set(style = "darkgrid")
plt.suptitle("Feature importance for Support Vector Machine")
plt.show()

#for decision trees
importance = decTree.feature_importances_
sns.barplot(x = importance, y = columnTitle[:13])
sns.set(style = "darkgrid")
plt.suptitle("Feature importance for Decision Trees")
plt.show()