# Introduction


According to the W.H.O, heart diseases or Cardiovascular diseases are the **number one cause of death globally** and claimed an estimated **17.9 million each year**. Through this project, I plotted graphs, using matplotlib and seaborn, and observed the various trends between the risk factors for heart diseases. Then, I created a **Logistic Regression, Support Vector Machine (SVM), Decision Trees, and Gaussian Naive Bayes** ML models to try and predict the heart disease. Lastly, utilising those models, I created plots for feature importance to see which risk factors are most important in predicting heart diseases. I believe that this can be play an important role in weighing these features to create more accurate models. 


# Getting Started


I am using the Cleveland Heart Disease data set which is available on the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Heart+Disease)
This dataset consists of data from 303 people. All the columns of the dataset are described below:

1. Age: age of person in years.
2. Sex: gender of person:
      - 1 = male
      - 0 = female
3. Chest-pain type: displays the type of chest-pain:
      - 1 = typical angina
      - 2 = atypical angina
      - 3 = non-anginal pain
      - 4 = asymptomatic
4. Resting Blood Pressure: resting blood pressure value of an individual in mmHg
5. Serum Cholestrol: serum cholesterol in mg/dl
6. Fasting Blood Sugar: compares the fasting blood sugar value of an individual with 120mg/dl.
      - fasting blood sugar > 120mg/dl: 1 (true)
      - else: 0 (false)
7. Resting ECG: electrocardiographic test results:
      - 0 = normal
      - 1 = ST-T wave abnormality
      - 2 = left ventricular hypertrophy
8. Max heart rate: max heart rate achieved
9. Exercise induced angina:
      - 1 = yes
      - 0 = no
10. ST depression induced by exercise relative to rest: displays the value which is an integer or float.
11. Peak exercise ST segment:
       - 1 = upsloping
       - 2 = flat
       - 3 = downsloping
12. Number of major vessels (0–3) colored by flourosopy : indicates number of blood vessels affected.
13. Thalassemia:
      - 3 = normal
      - 6 = fixed defect
      - 7 = reversible defect
14. Diagnosis of heart disease:
      - 0 = absence
      - 1, 2, 3, 4 = present.
      
I started by checking whether my data conatains any missing values. Here are the results:

<img src = "/graphs/missing_data.png" width = "150">

To fix these missing values in the vessels and thal columns, I used imputation and filled the missing values with the mean of the column. Since the dataset is not that big, I did not want to waste rows which were perfextly fine but just had 1 or 2 missing values. 

# Analysing the data


I started by plotting a correlation matrix. This would enable me to observe whether any of the parameters have a strong correlation with each other and further investigate it. 

<img src = "/graphs/correlationMatrix.png" width = "750">
We observed that none of the parameters have especially strong positive or negative correlations with each other. ST Elevation, presence of thalassemia, affected blood vessels, chest pain type, age and sex show a weak positive correlation. 

Next, I created a plot for comparing the number of people by each age who are affected by heart diseases. 

<img src = "/graphs/Age&Target.png" width = "750">

Here, I get some interesting results. I see that after the age of 55 years, there is a spike in number of people affected by heart diseases. I observed an upward trend from which I inferred that older people are more at risk for heart diseases. This finding is corroborated by lots of sources. Mayo Clinic states that "Men age 45 or older and women age 55 or older are more likely to have a heart attack than are younger men and women." Thus, age is a risk factor for heart diseases although we also see from our result that people even younger than that age group are susceptible to heart diseases. 

