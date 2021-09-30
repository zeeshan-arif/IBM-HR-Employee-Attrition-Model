
# IBM HR Employee Attrition Model

We perform a data exploration and feature engineering to build machine learning model and evaluate model performances to predict if an employee will leave his/her employer and analyze what features will affect employee's attrition.

Exploratory data analysis is one of the first approaches to understand about the dataset. We will get to know about numerical features, categorical features where are the missing values, which features are useful and useless. We will visualize the dataset through python **matplotlib** and **seaborn** libraries. Visualization is an effective way to tell about the data to non technical human.

The dataset is imbalanced, so we will use F1 score for performance. Features of the dataset are :-

| Features | Description|
|----------|------------|
|Age       |Age in years            |
|Attrition       |Yes or No            |
|Department       |Sales, Research & Development, Human Resources            |
|DistanceFromHome       |Number of kilometers from Home            |
|Education       |1-Below College, 2-College, 3-Bachelor, 4-Master, 5-Doctor         |
|EducationField       |Life Sciences, Medical, Marketing, Technical Degree, Human Resources and Other            |
|EnvironmentSatisfaction       |1-Low, 2-Medium, 3-High, 4-Very High            |
|JobSatisfaction       |1-Low, 2-Medium, 3-High, 4-Very High            |
|MaritalStatus       |Married, Single, Divorced            |
|MonthlyIncome       |Monthly Earning            |
|NumCompaniesWorked    |Number between 0-9            |
|WorkLifeBalance       |1-Bad, 2-Good, 3-Better, 4-Best            |
|YearsAtCompany       |Current years of service in IBM            |


## Libraries used

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV
```


#### Steps involved
- Load the dataset in pandas.
- Check the shape of dataset, the data types of features.
- Plot the distribution of numerical features. Plotting will reveal to us whether the feature is normally distributed or skewed. Based on the dataset we infer -
  - Large number of employees are between 30-37 years of age. 17 employees are under 20 years of age and 5 employees are above 60 years of age.
  - 1026 employee reside with 10 km of range from office.
  - 864 employees have worked in less 2 companies.
  - There 4 employees having 35 or more years of experience.
  - 749 employees earn 5000 or less.
  - 961 employees are in R & D.
  - 572 employees hold Bachelor Degree and 48 hold Doctor Degree.
  - 606 are from Life Sciences and 464 are from Medical education EducationField.
  - 899 are full satisfied with environment.
  - 901 are full satisfied with job.
  - 893 have better work life balance.
  - Married Employees have Highest WorkLifeBalance.
  - Employees under the age of 20 are more likely to leave the compnay than there counterpart.
- Plot the categorical features using any plot (usually we use BarPlot)
- Remove the outliers from the dataset.
- We transformed the skewed features into normally distributed.
- Scale the numerical features using StandardScaler.
- Use the Logistic Regression to build the model.
- Tune the parameter using GridSearchCV and use it with Logistic Regression.
- F1 score after hyperparameter tuning using Logistic Regression - 84%
