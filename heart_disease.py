import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
# thal: thalassemia (1 = normal; 2 = fixed defect; 3 = reversible defect)
# target: (1= heart disease or 0= no heart disease)


#print(df.isnull().sum()) 
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


