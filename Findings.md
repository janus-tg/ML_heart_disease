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
Furthernore, for the 'Target' column, I replaced 2, 3, and 4 with 1 to indicate presence of heart disease as it would aid our analysis and ML model creation later on. 

# Analysing the data


I started by plotting a correlation matrix. This would enable me to observe whether any of the parameters have a strong correlation with each other and further investigate it. 

<img src = "/graphs/correlationMatrix.png" width = "750">
We observed that none of the parameters have especially strong positive or negative correlations with each other. ST Elevation, presence of thalassemia, affected blood vessels, chest pain type, age and sex show a weak positive correlation. 

Next, I created a plot for comparing the number of people by each age who are affected by heart diseases. 

<img src = "/graphs/Age&Target.png" width = "750">

Here, I get some interesting results. I see that after the age of 55 years, there is a spike in number of people affected by heart diseases. I observed an upward trend from which I inferred that older people are more at risk for heart diseases. This finding is corroborated by lots of sources. [Mayo Clinic](https://www.mayoclinic.org/diseases-conditions/heart-attack/symptoms-causes/syc-20373106) states that "Men age 45 or older and women age 55 or older are more likely to have a heart attack than are younger men and women." Thus, age is a risk factor for heart diseases although we also see from our result that people even younger than that age group are susceptible to heart diseases. 

Next, I created a pie chart to see the relation between gender and being affected by a heart disease. 

<img src = "/graphs/maleFemale.png" width = "500">
This shows that a majority of males are affected by heart diseases as compared to females. According to a <a href = "https://www.health.harvard.edu/heart-health/throughout-life-heart-attacks-are-twice-as-common-in-men-than-women">study</a>
done in 2016, "throughout life, men were about twice as likely as women to have a heart attack. That higher risk persisted even after they accounted for traditional risk factors for heart disease, including high cholesterol, high blood pressure, diabetes, body mass index, and physical activity. Earlier studies suggested that women's naturally occurring hormone levels might protect against heart disease before menopause, when hormone levels drop. However, the risk of heart attack changed only slightly as women transitioned through menopause, making it unlikely that female hormone levels explain these findings."

Next, I plotted a boxen plot to see the cholesterol levels by age. 

<img src = "/graphs/ageChol.png" width = "750">

Although a clear correlation was not seen here but a more disturbing fact was observed. I observed that after approximately the age of 37 years, the average cholesterol levels of the people was more than the recommended upper threshold of [200 mg/dl](https://medlineplus.gov/cholesterollevelswhatyouneedtoknow.html). This shows that most people led unhealthy lifestyles which contributed to their higher cholesterol. 

Next, I plotted graphs for the mean blood vessels affected by heart disease compared to age and gender. 

<img src = "/graphs/vesselsAge.png" width = "680"><img src = "/graphs/vesselSex.png" width = "320">

We see that there is a weak upward trend starting from the age of 40 years. This shows that more people aged >= 40 years have got affected blood vessels and thus affected by a heart disease called [Atherosclerosis](https://www.mayoclinic.org/diseases-conditions/arteriosclerosis-atherosclerosis/symptoms-causes/syc-20350569#:~:text=Atherosclerosis%20refers%20to%20the%20buildup,arteries%20anywhere%20in%20your%20body.) and this also complements our earlier finding that older people are more at risk for heart diseases. 
The mean blood vessels of men is 0.7 as compared to 0.55 for women. Although the difference is not large, it still complements our earlier finding that men are more at risk than females for heart diseases.

In the correlation matrix, a positive correlation between chest-pain type and being affected by heart diseases was seen. Now, I plotted a bar plot between them to see how they are correlated. 

<img src = "/graphs/cpainTarget.png" width = "750">
 
This gave us very interesting results. Firstly, we saw that people having any type of chest pain have a chance of being afflicted with a heart disease. Secondly, we saw that many people having asymptomatic chest pain were affected by heart disease. This means that the asymptomatic people were affected by [Silent Ischemia](https://www.texasheart.org/heart-health/heart-information-center/topics/silent-ischemia/). According to the Texas Heart Institute, heart muscle disease (cardiomyopathy) caused by silent ischemia is among the more common causes of heart failure in the United States which is corroborated by our data. 


# Creating ML models for predicting heart disease


I used the scikit-learn module for making the models. First I began by splitting the dataset into 2 sets: training and testing. The training set was allocated 80% of the data and the remaining 20% of the data was alloted to the testing set. The testing set will be used to predict values using the models and that will be compared to the actual values. This will be further used to calculate the accuracy of our model. 

A confusion matrix will be plotted to check for the accuracy of each model. 

### 1. Logistic Regression 


[Logistic regression](https://en.wikipedia.org/wiki/Logistic_regression) is a statistical model that utilizes a logistic function to predict the outcome based on the parameters. In this case, we are using a binary logistic regression as the aim is to predict whether a person has heart disease (1) or is not (0). 

<img src = "/graphs/logReg.png" width = "500"> 
<img src = "/accuracy/LR.png" width = "500">


### 2. Support Vector Machine (SVM)


[SVM](https://en.wikipedia.org/wiki/Support_vector_machine) is essentially a classification algorithm. It takes the input and divides them using a hyperplane. This hyperplane is called the decision boundary. In this project, anything that falls into one side of the decision line is categorised as a person having heart disease (1) or is not having heart disease (0). 

<img src = "/graphs/svm.png" width = "500"> 
<img src = "/accuracy/SVM.png" width = "500">


### 3. Gaussian Naive Bayes


[Naive Bayes](https://en.wikipedia.org/wiki/Naive_Bayes_classifier#Gaussian_na%C3%AFve_Bayes) is a probabilistic classifier which is based on the application of the [Bayes Theorem](https://en.wikipedia.org/wiki/Bayes%27_theorem) and assuming that all the parameters are independant of each other. Gaussian Naive Bayes is based on the fact that all the continous values associated with each parameter are distributed by a normal distribution. 

<img src = "/graphs/naiveBayes.png" width = "500"> 
<img src = "/accuracy/NB.png" width = "500">


### 4. Decision Tree


[Decision Tree](https://en.wikipedia.org/wiki/Decision_tree#Overview) uses a tree-like model of decisions, with subsequent nodes having their outcomes and probabilities. 
In this project, the decision tree starts with 2 burst nodes indicating whether a person has heart disease (1) or does not (0). The subsequent nodes are based on the other 13 parameters which influence whether a person has heart disease or not. 


<img src = "/graphs/decisionTree.png" width = "500"> 
<img src = "/accuracy/DT.png" width = "500">


Based on all these models, it is seen that the **logistic regression model is the most accurate** amongst the bunch having an **accuracy of 86.885%**.


# Feature importance and going forward


An important step in making even more accurate models would be using feature importance. Feature importance is a method using which we can assign importance to each parameter and select the most important parameters or weigh them accordingly. Here are the bar plots of relative feature importance:

