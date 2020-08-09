## Fifa 2018 Skills Challenge
Andrew Bond <br>
Aug 08, 2020


```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
```

We need to load the data and clean it. The data is not encoded with 'utf-8' so we need to load find its encoding and use that instead


```python
tempdata = open("Skills_Verification_Data_FIFA_18 (2) (1) (1) (1).csv")
print("The file's encoding is ",tempdata.encoding)
fifa_df = pd.read_csv(tempdata,encoding = 'cp1252', low_memory=False).dropna(how="all", inplace=False)
fifa_df.head()
```

    The file's encoding is  cp1252
    




<div>
<style scoped>
    
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>No</th>
      <th>ID</th>
      <th>Name</th>
      <th>Age</th>
      <th>Nationality</th>
      <th>Overall</th>
      <th>Potential</th>
      <th>Club</th>
      <th>Value</th>
      <th>Wage</th>
      <th>...</th>
      <th>Penalties</th>
      <th>Composure</th>
      <th>Marking</th>
      <th>StandingTackle</th>
      <th>SlidingTackle</th>
      <th>GKDiving</th>
      <th>GKHandling</th>
      <th>GKKicking</th>
      <th>GKPositioning</th>
      <th>GKReflexes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>158023.0</td>
      <td>L. Messi</td>
      <td>31.0</td>
      <td>Argentina</td>
      <td>94.0</td>
      <td>94.0</td>
      <td>FC Barcelona</td>
      <td>€110.5M</td>
      <td>$565</td>
      <td>...</td>
      <td>75.0</td>
      <td>96.0</td>
      <td>33.0</td>
      <td>28.0</td>
      <td>26.0</td>
      <td>6.0</td>
      <td>11.0</td>
      <td>15.0</td>
      <td>14.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>20801.0</td>
      <td>Cristiano Ronaldo</td>
      <td>33.0</td>
      <td>Portugal</td>
      <td>94.0</td>
      <td>94.0</td>
      <td>Juventus</td>
      <td>€77M</td>
      <td>$405</td>
      <td>...</td>
      <td>85.0</td>
      <td>95.0</td>
      <td>28.0</td>
      <td>31.0</td>
      <td>23.0</td>
      <td>7.0</td>
      <td>11.0</td>
      <td>15.0</td>
      <td>14.0</td>
      <td>11.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2.0</td>
      <td>190871.0</td>
      <td>Neymar Jr</td>
      <td>26.0</td>
      <td>Brazil</td>
      <td>92.0</td>
      <td>93.0</td>
      <td>Paris Saint-Germain</td>
      <td>€118.5M</td>
      <td>$290</td>
      <td>...</td>
      <td>81.0</td>
      <td>94.0</td>
      <td>27.0</td>
      <td>24.0</td>
      <td>33.0</td>
      <td>9.0</td>
      <td>9.0</td>
      <td>15.0</td>
      <td>15.0</td>
      <td>11.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3.0</td>
      <td>193080.0</td>
      <td>De Gea</td>
      <td>27.0</td>
      <td>Spain</td>
      <td>91.0</td>
      <td>93.0</td>
      <td>Manchester United</td>
      <td>€72M</td>
      <td>$260</td>
      <td>...</td>
      <td>40.0</td>
      <td>68.0</td>
      <td>15.0</td>
      <td>21.0</td>
      <td>13.0</td>
      <td>90.0</td>
      <td>85.0</td>
      <td>87.0</td>
      <td>88.0</td>
      <td>94.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4.0</td>
      <td>192985.0</td>
      <td>K. De Bruyne</td>
      <td>27.0</td>
      <td>Belgium</td>
      <td>91.0</td>
      <td>92.0</td>
      <td>Manchester City</td>
      <td>€102M</td>
      <td>$355</td>
      <td>...</td>
      <td>79.0</td>
      <td>88.0</td>
      <td>68.0</td>
      <td>58.0</td>
      <td>51.0</td>
      <td>15.0</td>
      <td>13.0</td>
      <td>5.0</td>
      <td>10.0</td>
      <td>13.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 55 columns</p>
</div>



We need to clean some of the data. A lot of columns contain data that can be converted to numerical.


```python
fifa_df_clean = fifa_df.copy()
fifa_df_clean['Value'] = fifa_df_clean['Value'].replace('[\€M]', '', regex=True).astype(float)
fifa_df_clean['Wage'] = fifa_df_clean['Wage'].replace('[\$]', '', regex=True).astype(float)

fifa_df_clean["Height"] = fifa_df_clean["Height"].apply(lambda x: int(x.split("'")[0]) * 12 + int(x.split("'")[1]))

fifa_df_clean["Weight"] = fifa_df_clean["Weight"].replace('[lbs]','',regex=True).astype(float)
```

## Predicting 'Value' with Machine Learning Models

Before even building a model, we can find the features that are most likely to predict 'Value', these variables will have strong positive or negative correlations with 'Value', which indicates that they help explain the variance.


```python
correlations = fifa_df_clean.corrwith(fifa_df_clean['Value']).sort_values(ascending=False)
print(correlations[abs(correlations) > 0.2])
```

    Value                       1.000000
    Overall                     0.833731
    Potential                   0.777904
    Wage                        0.754149
    Reactions                   0.623651
    International Reputation    0.423654
    Composure                   0.395891
    Finishing                   0.305001
    Vision                      0.298988
    Agility                     0.297495
    Acceleration                0.281586
    Positioning                 0.270860
    Volleys                     0.268087
    SprintSpeed                 0.258000
    BallControl                 0.257387
    LongShots                   0.252038
    Penalties                   0.250978
    ShortPassing                0.246763
    Dribbling                   0.243432
    FKAccuracy                  0.241397
    Curve                       0.235706
    Skill Moves                 0.231791
    Balance                     0.225658
    LongPassing                 0.210275
    Age                        -0.220359
    No                         -0.734687
    dtype: float64
    

We sorted the correlations in descending order. Clearly 'Value' is perfectly correlated with 'Value', and strangely enough, No is also strongly negatively correlated with the 'Value'. This indicates the rows are in roughly descending order, but is not actually a sensible predictor of 'Value'. We will ignore 'No' and 'Value' and use every other attribute whose correlation is greater than 0.2.


```python
features = correlations[abs(correlations) > 0.2].index
features = features.drop(['Value','No'])
```

Now we will make a new dataframe with the desired features, and another with the 'Value's. We will then perform our train test split to set aside some rows for validation after training the model.


```python
from sklearn.model_selection import train_test_split
X = fifa_df_clean[features]
y = fifa_df_clean['Value']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=5)
```

#### Ordinary Least Square Regression
We will start by training an ordinary least squares regression model on the selected features. Since we are starting with Linear Regression, we are ignoring the variables that could be used categorically. It is possible to make use of them but not necessarily worth the time right off the bat. We could use techniques such as <b> one-hot encoding </b> to try to leverage them in our regression model, but we will not be doing that for now. We will also be using KFold cross-validation method reduce overfitting during training.


```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

fig=plt.figure(figsize=(12,8), dpi= 100, facecolor='w', edgecolor='k')

lr_model = LinearRegression()
r2_scores = []
kfold = KFold(n_splits = 9, shuffle=True, random_state=5)
for i, (train,test) in enumerate(kfold.split(X_train, y_train)):
    lr_model.fit(X_train.iloc[train,:], y_train.iloc[train])
    score = lr_model.score(X_train.iloc[test,:],y_train.iloc[test])
    r2_scores.append(score)
    
plt.plot(r2_scores)
plt.title('R2 score during KFold cross-validation training')
plt.xlabel('Split')
plt.ylabel('R2')

print("The mean R2 score during the training was ",np.array(r2_scores).mean())
```

    The mean R2 score during the training was  0.88486356105551
    


![png](output_14_1.png)


Now we can validate the model on the withheld 'test' set.


```python
validation_score = lr_model.score(X_test,y_test)
print('The R2 on the test set was ', validation_score)
```

    The R2 on the test set was  0.919977904345087
    

This has a pretty good R2 value, but that just tells us that the regression model does a pretty good job representing the variance in the data set. 
## We can determine from the coefficients what the most important variable for prediction is: 


```python
print("The most important feature for predicting 'Value' is ",features[np.argmax(lr_model.coef_)]," which has a coeffient of ",lr_model.coef_.max())
```

    The most important feature for predicting 'Value' is  Overall  which has a coeffient of  5.691820876728239
    

There are other ways that we can assess the model performance.


```python
import sklearn.metrics as metrics
y_train_pred = lr_model.predict(X_train)
y_pred = lr_model.predict(X_test)
print("RMSE:\n Train: ",np.sqrt(metrics.mean_squared_error(y_train,y_train_pred)),"Test: ",np.sqrt(metrics.mean_squared_error(y_test,y_pred)))
print("Max Error:\n Train: ",metrics.max_error(y_train,y_train_pred),"Test: ",metrics.max_error(y_test,y_pred))
print("Mean Absolute Error:\n Train: ",metrics.mean_absolute_error(y_train,y_train_pred),"Test: ",metrics.mean_absolute_error(y_test,y_pred))
```

    RMSE:
     Train:  4.982261013093777 Test:  4.259356249221708
    Max Error:
     Train:  33.84808006416739 Test:  18.634921567114702
    Mean Absolute Error:
     Train:  3.467346061156851 Test:  3.1036554658122224
    

These metrics tell us a little bit more about the performance of our model. The Root Mean Squared error tells us that the standard deviation of our residuals is 4.5 million Euros. The max error shows that the worst prediction we made was off by around 18.5 million(on the test set) and 33.5 million(on the training set) and finally, the Mean Absolute Error tells us that our literal mean error is around 3 million Euros. Scanning the data, these errors seem low in comparison to a lot of the 'Value's, but not all.
<br>
<br>
We can come up with another metric to assess the performance of our model, a heuristic approach. We will find the standard deviation of the 'Value' attribute and use it to give ourselves a threshold or tolerance for the error under which we will declare our model 'Accurate'.


```python
value_std = fifa_df_clean.Value.std()
print("Standard deviation of the 'Value' variable: ",value_std)
```

    Standard deviation of the 'Value' variable:  16.51781644769068
    

With this we could calculate our accuracy by determining what number of the predictions are within one standard deviation of their value, but I think we can be a little bit more stringent than that. We will calculate what percentage the standard deviation is of the maximum 'Value' and use that percentage error as our tolerance for accuracy.


```python
max_value = fifa_df_clean.Value.max()
tolerance = value_std/max_value
print("Our tolerance will be ", tolerance*100, "%")
```

    Our tolerance will be  13.939085609865554 %
    


```python
residuals = (y_pred - y_test)
no_accurate = residuals[abs(residuals)/y_test <= tolerance].count()
print("With this new threshold tolerance for accuracy, we have an accuracy of: ",no_accurate/y_test.count()*100 , "%")
```

    With this new threshold tolerance for accuracy, we have an accuracy of:  64.90066225165563 %
    

Not bad! With our Ordinary Least Squares regression model and our somewhat stringent metric, we managed to achieve an accuracy of ~65% using less than 25 variables. Our metric required that our prediction was within ~13.9% of the true value. Let's take a look at the values for which we weren't able to make this threshold.


```python
wrong = residuals[abs(residuals)/y_test > tolerance]
fig=plt.figure(figsize=(12,8), dpi= 100, facecolor='w', edgecolor='k')
plt.scatter(range(wrong.count()),y_test[wrong.index])
plt.title('Values of the inaccurate predictions')
plt.xlabel('')
plt.ylabel('Value')
print("The mean value of the wrong predictions is: ",y_test[wrong.index].mean())
print("The threshold for the mean of the incorrect predictions is: ",y_test[wrong.index].mean()*tolerance)
```

    The mean value of the wrong predictions is:  19.5188679245283
    The threshold for the mean of the incorrect predictions is:  2.7207517100765877
    


![png](output_27_1.png)


We should expect that the threshold would be harder to meet for the lower 'Value's. The mean of the missed values is 19.5 million, the threshold 13.9% for this mean is around 2.7 million. For a dataset that includes 'Value's over 100 million, this is pretty precise and its no surprise that the model couldn't manage accuracy on these players. This could potentially be addressed by categorical variables or by removing outliers. We will examine outliers such as "Superstars" later as a business question.

#### Decision Trees and more
We also want to try to take advantage of our categorical variables, which will require a little bit more cleaning to prepare. These variables can be very valuable to decision trees and algorithms like Ada Boost and Random Forests. We will use these and see how well they perform.

We start with the cleaning, I have identified some columns as great categorical candidates to inform our model:


```python
fifa_df_clean["Nationality"] = fifa_df_clean["Nationality"].astype("category")
fifa_df_clean["Club"] = fifa_df_clean["Club"].astype("category")
fifa_df_clean["Preferred Foot"] = fifa_df_clean['Preferred Foot'].astype("category")

fifa_df_clean["Work Rate"] = fifa_df_clean["Work Rate"].astype("category")
fifa_df_clean["Position"] = fifa_df_clean["Position"].astype("category")

fifa_df_clean.insert(fifa_df_clean.columns.get_loc("Joined") + 1, "Joined (Month)", pd.DatetimeIndex(pd.to_datetime(fifa_df_clean["Joined"])).strftime("%b").astype('category'))
fifa_df_clean["Joined (Month)"] = fifa_df_clean["Joined (Month)"].cat.reorder_categories(['Jan', 'Feb', 'Mar', 'Apr', 'May','Jun','Jul', 'Aug','Sep','Oct','Nov', 'Dec'])
fifa_df_clean["Joined (Month)"] = fifa_df_clean["Joined (Month)"].fillna('Jul')

fifa_df_clean.insert(fifa_df_clean.columns.get_loc("Joined") + 2, "Joined (Year)", pd.DatetimeIndex(pd.to_datetime(fifa_df_clean["Joined"])).year)
fifa_df_clean["Joined (Year)"] = fifa_df_clean["Joined (Year)"].fillna(2018)
```

We can take a look at a few of these variables and see if they have clear "classes" amongst them.


```python
fig, ax = plt.subplots(2,2, figsize=(12,8))

for j in range(len(ax)):
    for i in range(len(ax[j])):
        ax[j][i].set_xlabel('label')

ax[0][0] = sns.boxplot(x="Club", y="Value", data=fifa_df_clean, ax=ax[0][0])
ax[1][0] = sns.boxplot(x="Nationality", y="Value", data=fifa_df_clean, ax=ax[1][0])
ax[0][1] = sns.boxplot(x="Position", y="Value", data=fifa_df_clean, ax=ax[0][1])
ax[1][1] = sns.boxplot(x="Joined (Year)", y="Value", data=fifa_df_clean, ax=ax[1][1])

for j in range(len(ax)):
    for i in range(len(ax[j])):
        ax[j][i].set_xticks([])

fig.show()
```

    C:\Users\andyr\anaconda3\lib\site-packages\ipykernel_launcher.py:16: UserWarning: Matplotlib is currently using module://ipykernel.pylab.backend_inline, which is a non-GUI backend, so cannot show the figure.
      app.launch_new_instance()
    


![png](output_33_1.png)


It does appear that we have some separate classes going on here. We can also see many of the outliers that probably caused us issues with the Linear Regression model. We will use the codes for the categories rather than the category names themselves.


```python
fifa_df_clean["Nationality"] = fifa_df_clean["Nationality"].cat.codes
fifa_df_clean["Preferred Foot"] = fifa_df_clean["Preferred Foot"].cat.codes
fifa_df_clean["Work Rate"] = fifa_df_clean["Work Rate"].cat.codes
fifa_df_clean["Position"] = fifa_df_clean["Position"].cat.codes
fifa_df_clean["Joined (Month)"] = fifa_df_clean["Joined (Month)"].cat.codes
fifa_df_clean["Club"] = fifa_df_clean["Club"].cat.codes
```

We are also going to include the variables that have the good correlation with 'Value'.


```python
features = features.append(pd.Index(['Nationality','Preferred Foot','Work Rate','Position','Joined (Month)','Club']))
X_dt = fifa_df_clean[features]
y_dt = fifa_df_clean['Value']
X_dt_train, X_dt_test, y_dt_train, y_dt_test = train_test_split(X_dt, y_dt, test_size = 0.3, random_state=5)
```

We are going to use Decision Trees, the most basic. Ada Boost to provide a refined Decision Tree with more and more accurate classes and a Random Forest.


```python
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
```


```python
def train_trees(max_depth, n_estimators, random_state):
    dtmodel = DecisionTreeRegressor(max_depth = max_depth)
    abmodel = AdaBoostRegressor(DecisionTreeRegressor(max_depth = max_depth), n_estimators = n_estimators, random_state = random_state)
    rfmodel = RandomForestRegressor(max_depth=max_depth, random_state = random_state)
    
    dtmodel.fit(X_dt_train, y_dt_train)
    abmodel.fit(X_dt_train, y_dt_train)
    rfmodel.fit(X_dt_train, y_dt_train)
    
    dtscore = dtmodel.score(X_dt_test,y_dt_test)
    abscore = abmodel.score(X_dt_test,y_dt_test)
    rfscore = rfmodel.score(X_dt_test,y_dt_test)
    
    return (dtmodel,abmodel,rfmodel,dtscore,abscore,rfscore)
```


```python
dtscores = []
abscores = []
rfscores = []
labels = []
for i in range(3,16):
    for j in np.arange(200, 351, 50):
        (_,_,_,dts,abs,rfs) = train_trees(i, j, 5)
        labels.append(str(i)+','+str(j))
        dtscores.append(dts)
        abscores.append(abs)
        rfscores.append(rfs)
```


```python
fig=plt.figure(figsize=(12,8), dpi= 100, facecolor='w', edgecolor='k')
plt.plot(labels,dtscores)
plt.plot(labels,abscores)
plt.plot(labels,rfscores)
plt.title("R2 score for each algorithm based on max_depth and estimators")
plt.legend(["Decision Tree","AdaBoost","Random Forest"])
locs, lab = plt.xticks()
plt.setp(lab, rotation=90);
```


![png](output_42_0.png)


AdaBoost and the Random Forest level out for the most part sowe will choose one of the options for AdaBoost that spikes around a max_depth of 13, say 13,250


```python
(_,ab,_,_,_,_) = train_trees(13,250,5)
y_ab_train_pred = ab.predict(X_dt_train)
y_ab_pred = ab.predict(X_dt_test)
print("RMSE:\n Train: ",np.sqrt(metrics.mean_squared_error(y_dt_train,y_ab_train_pred)),"Test: ",np.sqrt(metrics.mean_squared_error(y_dt_test,y_ab_pred)))
print("Max Error:\n Train: ",metrics.max_error(y_dt_train,y_ab_train_pred),"Test: ",metrics.max_error(y_dt_test,y_ab_pred))
print("Mean Absolute Error:\n Train: ",metrics.mean_absolute_error(y_dt_train,y_ab_train_pred),"Test: ",metrics.mean_absolute_error(y_dt_test,y_ab_pred))
```

    RMSE:
     Train:  0.43728182580772934 Test:  3.9020154513592082
    Max Error:
     Train:  2.5 Test:  20.0
    Mean Absolute Error:
     Train:  0.16486263736263737 Test:  2.225953105423304
    

The results above make are suspicious of some overfitting, but the metrics are still pretty good, we will move on to calculating our heuristic.


```python
residuals_ab = (y_ab_pred - y_dt_test)
no_accurate_ab = residuals_ab[np.abs(residuals_ab)/y_dt_test <= tolerance].count()
print("With this new threshold tolerance for accuracy, we have an accuracy of: ",no_accurate_ab/y_dt_test.count()*100 , "%")
```

    With this new threshold tolerance for accuracy, we have an accuracy of:  82.11920529801324 %
    

With this method we got an accuracy of 82%, which is quite good! To improve this score we could also spent some time playing around with features, and investigating the possibility of overfitting as possibly indicated from the features above. We will determine the most important feature really quickly.
## The most important features for prediction are:


```python
top_features = np.array(ab.feature_importances_).argsort()[::-1][:5]
print("The top 5 features are:")
for i in range(len(top_features)):
    print(features[top_features[i]])
```

    The top 5 features are:
    Potential
    Overall
    Reactions
    Wage
    Age
    

Like before we will look at the Values of the rows that we mis-predicted.


```python
wrong_ab = residuals_ab[np.abs(residuals_ab)/y_dt_test > tolerance]
fig=plt.figure(figsize=(12,8), dpi= 100, facecolor='w', edgecolor='k')
plt.scatter(range(wrong_ab.count()),y_dt_test[wrong_ab.index])
plt.title('Values of the inaccurate predictions')
plt.xlabel('')
plt.ylabel('Value')
print("The mean value of the wrong predictions is: ",y_dt_test[wrong_ab.index].mean())
print("The threshold for the mean of the incorrect predictions is: ",y_dt_test[wrong_ab.index].mean()*tolerance)
```

    The mean value of the wrong predictions is:  25.925925925925927
    The threshold for the mean of the incorrect predictions is:  3.6138370099651436
    


![png](output_50_1.png)


While we have much fewer inaccurate predictions, the average is a bit higher and we see some errors right in the middle range. With more time we could take a look at why this is happening. It also seems that accounting for outliers may help quite a bit with the training of these models as well. We could also take a look at other models for the regression problem, or penalties such as Ridge, Lasso or Elastic Net. We could also examine the bias versus the variance in our models to try to find the right balance. But for now, this is all we are going to do when it comes to the prediction of Value.

## Data Exploration and Business Questions
While cleaning the data for the machine learning models, I became interested in the effects of the classes such as 'Club'. I think that a good starting place for exploration and an appropriate Business question is, "How do 'Value's distribute amongst these Clubs". With that in mind we will start splitting up the data and visualizing it. To examine this question, we will create a new attribute,  $VpW = \frac{Value}{Wage}$ which can help us identify business value of a player.
<br>
We also need to bring our categories back to make sense of what we see.


```python
fifa_df_clean['VpW'] = fifa_df_clean['Value']/fifa_df_clean['Wage']
fifa_df_clean["Nationality"] = fifa_df["Nationality"].astype("category")
fifa_df_clean["Club"] = fifa_df["Club"].astype("category")
fifa_df_clean["Preferred Foot"] = fifa_df['Preferred Foot'].astype("category")

fifa_df_clean["Work Rate"] = fifa_df["Work Rate"].astype("category")
fifa_df_clean["Position"] = fifa_df["Position"].astype("category")
```


```python
fig=plt.figure(figsize=(12,8), dpi= 100, facecolor='w', edgecolor='k')
sns.boxplot(x="Club", y="VpW", data=fifa_df_clean)
locs, lab = plt.xticks()
plt.setp(lab, rotation=90);
```


![png](output_54_0.png)


Clearly something fishy is going on here, we can look at the data to find out why these Clubs have such a atypical distribution of VpW's


```python
clubs = (fifa_df_clean['Club'] == "Shakhtar Donetsk") |  (fifa_df_clean['Club'] == "Lokomotiv Moscow")
df_vpw_outliers = fifa_df_clean[clubs]
df_vpw_outliers.Wage
```




    286    1.0
    288    1.0
    301    1.0
    381    1.0
    386    1.0
    405    1.0
    453    1.0
    473    1.0
    484    1.0
    Name: Wage, dtype: float64



Well, it appears that some of these clubs are not paying their players or don't release Wage information, it could also be clerical errors or perhaps autofilled due to lack of data. In reality, this would be a time to go talk to a subject matter expert and figure out what is going on with our data, but we will forge ahead with the assumption that we can hire the players for the wages given. Let's take a look if this lack of pay is related to Nationality.


```python
fig=plt.figure(figsize=(12,8), dpi= 100, facecolor='w', edgecolor='k')
sns.boxplot(x="Nationality", y="VpW", data=fifa_df_clean)
locs, lab = plt.xticks()
plt.setp(lab, rotation=90);
```


![png](output_58_0.png)



```python
countries = (fifa_df_clean['Nationality'] == "Ukraine") |  (fifa_df_clean['Nationality'] == "Russia")
df_vpw_outliers = fifa_df_clean[clubs]
df_vpw_outliers.Wage
```




    286    1.0
    288    1.0
    301    1.0
    381    1.0
    386    1.0
    405    1.0
    453    1.0
    473    1.0
    484    1.0
    Name: Wage, dtype: float64



We found the same indices by looking at wages for Ukraine and Russia, so it seems like these anomolies in the data are related to the country that the player is from. I don't think we can do any more with these underpaid players without more background information so instead we will try to figure out what is going on with the other outliers.


```python
fig=plt.figure(figsize=(12,8), dpi= 100, facecolor='w', edgecolor='k')
outliers = fifa_df_clean['Nationality'] == "Brazil"
df_vpw_outliers = fifa_df_clean[outliers]
sns.scatterplot(x ='Wage', y='Value', data = df_vpw_outliers)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x24b87e46d88>




![png](output_61_1.png)


Once again, it seems like we have some underpaid players. My guess is that unless we eliminate these ridiculously low wages, we would use this data to put together a team of players whose cost is not representative of reality. We could use a Linear Regression model or other Machine Learning model trained on the players whose Wage is greater than 1, and reassign Wage for the players. For the sake of time we will just remove the players whose Wage is 1.


```python
fifa_df_wage = fifa_df_clean[fifa_df_clean['Wage'] > 1]
print(len(fifa_df_wage))
```

    490
    

Now that we got rid of those irrationally paid players, we will look at Clubs again.


```python
fig=plt.figure(figsize=(12,8), dpi= 100, facecolor='w', edgecolor='k')
sns.boxplot(x="Club", y="VpW", data=fifa_df_wage)
locs, lab = plt.xticks()
plt.setp(lab, rotation=90);
```


![png](output_65_0.png)


Things look a lot more interesting now, and it is clear that the Clubs have very different distributions of VpW. Since there are 23 players on a FIFA team (I looked it up) let's take a look at the top 23 'VpW' players and their wages.


```python
top_vpw = fifa_df_wage.sort_values(by=['VpW'], ascending=False).iloc[:23]
top_vpw[['Name','Wage','VpW','Value']]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>Wage</th>
      <th>VpW</th>
      <th>Value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>226</th>
      <td>M. de Ligt</td>
      <td>11.0</td>
      <td>2.454545</td>
      <td>27.0</td>
    </tr>
    <tr>
      <th>166</th>
      <td>Anderson Talisca</td>
      <td>18.0</td>
      <td>2.027778</td>
      <td>36.5</td>
    </tr>
    <tr>
      <th>114</th>
      <td>Bruno Fernandes</td>
      <td>22.0</td>
      <td>1.840909</td>
      <td>40.5</td>
    </tr>
    <tr>
      <th>418</th>
      <td>M. Almirón</td>
      <td>11.0</td>
      <td>1.772727</td>
      <td>19.5</td>
    </tr>
    <tr>
      <th>173</th>
      <td>Y. Carrasco</td>
      <td>20.0</td>
      <td>1.650000</td>
      <td>33.0</td>
    </tr>
    <tr>
      <th>345</th>
      <td>J. Martínez</td>
      <td>14.0</td>
      <td>1.607143</td>
      <td>22.5</td>
    </tr>
    <tr>
      <th>342</th>
      <td>Grimaldo</td>
      <td>14.0</td>
      <td>1.464286</td>
      <td>20.5</td>
    </tr>
    <tr>
      <th>112</th>
      <td>Alex Telles</td>
      <td>22.0</td>
      <td>1.454545</td>
      <td>32.0</td>
    </tr>
    <tr>
      <th>94</th>
      <td>Y. Brahimi</td>
      <td>28.0</td>
      <td>1.392857</td>
      <td>39.0</td>
    </tr>
    <tr>
      <th>323</th>
      <td>F. de Jong</td>
      <td>19.0</td>
      <td>1.368421</td>
      <td>26.0</td>
    </tr>
    <tr>
      <th>293</th>
      <td>S. Giovinco</td>
      <td>15.0</td>
      <td>1.333333</td>
      <td>20.0</td>
    </tr>
    <tr>
      <th>187</th>
      <td>Danilo Pereira</td>
      <td>21.0</td>
      <td>1.285714</td>
      <td>27.0</td>
    </tr>
    <tr>
      <th>229</th>
      <td>G. Donnarumma</td>
      <td>23.0</td>
      <td>1.260870</td>
      <td>29.0</td>
    </tr>
    <tr>
      <th>371</th>
      <td>J. Corona</td>
      <td>18.0</td>
      <td>1.194444</td>
      <td>21.5</td>
    </tr>
    <tr>
      <th>397</th>
      <td>C. Vela</td>
      <td>15.0</td>
      <td>1.166667</td>
      <td>17.5</td>
    </tr>
    <tr>
      <th>171</th>
      <td>H. Ziyech</td>
      <td>28.0</td>
      <td>1.160714</td>
      <td>32.5</td>
    </tr>
    <tr>
      <th>194</th>
      <td>Pizzi</td>
      <td>22.0</td>
      <td>1.159091</td>
      <td>25.5</td>
    </tr>
    <tr>
      <th>274</th>
      <td>S. Coates</td>
      <td>19.0</td>
      <td>1.105263</td>
      <td>21.0</td>
    </tr>
    <tr>
      <th>329</th>
      <td>H. Lozano</td>
      <td>22.0</td>
      <td>1.090909</td>
      <td>24.0</td>
    </tr>
    <tr>
      <th>170</th>
      <td>Q. Promes</td>
      <td>28.0</td>
      <td>1.089286</td>
      <td>30.5</td>
    </tr>
    <tr>
      <th>436</th>
      <td>Rafa</td>
      <td>17.0</td>
      <td>1.088235</td>
      <td>18.5</td>
    </tr>
    <tr>
      <th>205</th>
      <td>Oscar</td>
      <td>29.0</td>
      <td>1.051724</td>
      <td>30.5</td>
    </tr>
    <tr>
      <th>426</th>
      <td>A. Onana</td>
      <td>14.0</td>
      <td>1.035714</td>
      <td>14.5</td>
    </tr>
  </tbody>
</table>
</div>




```python
print("Using our top 23 VpW players, we get a team that costs us: ")
print("$",top_vpw['Wage'].sum())
print("This budget team gives us a Value of: ")
print(top_vpw['Value'].sum()," million Euros")
```

    Using our top 23 VpW players, we get a team that costs us: 
    $ 450.0
    This budget team gives us a Value of: 
    608.5  million Euros
    

For less than a single one of the "superstars", we can get a value of 608.5 million Euros, probably 6 times the value of a superstar...but that is our entire team. Great on a budget, but could we actually win any games? We can't answer that question with our data, but we can certainly look at our "Overall" scores and whether we even have the right players for the positions we need! Before we look at our team, lets get an idea of what the "Overall" scores look like amongst our data. We also should take a look what what positions we need to put together a legitimate team.


```python
fig=plt.figure(figsize=(12,8), dpi= 100, facecolor='w', edgecolor='k')
sns.boxplot(y="Overall", data=fifa_df_clean)
locs, lab = plt.xticks()
plt.setp(lab, rotation=90);
print("The mean 'Overall' score of a player in our data is: ",fifa_df_clean.Overall.mean())
```

    The mean 'Overall' score of a player in our data is:  82.8942115768463
    


![png](output_70_1.png)


So, the average for players in our data is right around 82. Next, we look into team compositions and viable team strategies. I used the following site<br> <a href="https://protips.dickssportinggoods.com/sports-and-activities/soccer/soccer-positions-the-numbers-player-roles-basic-formations">Soccer Team Composition</a>.
<br>
With what we know, lets take a look at our "Budget" team.


```python
print("Our team's mean 'Overall' score is: ", top_vpw.Overall.mean())
top_vpw[['Name','Position','Overall']]
```

    Our team's mean 'Overall' score is:  82.08695652173913
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>Position</th>
      <th>Overall</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>226</th>
      <td>M. de Ligt</td>
      <td>RCB</td>
      <td>82.0</td>
    </tr>
    <tr>
      <th>166</th>
      <td>Anderson Talisca</td>
      <td>CAM</td>
      <td>83.0</td>
    </tr>
    <tr>
      <th>114</th>
      <td>Bruno Fernandes</td>
      <td>LCM</td>
      <td>84.0</td>
    </tr>
    <tr>
      <th>418</th>
      <td>M. Almirón</td>
      <td>CAM</td>
      <td>80.0</td>
    </tr>
    <tr>
      <th>173</th>
      <td>Y. Carrasco</td>
      <td>LM</td>
      <td>83.0</td>
    </tr>
    <tr>
      <th>345</th>
      <td>J. Martínez</td>
      <td>LS</td>
      <td>81.0</td>
    </tr>
    <tr>
      <th>342</th>
      <td>Grimaldo</td>
      <td>LB</td>
      <td>81.0</td>
    </tr>
    <tr>
      <th>112</th>
      <td>Alex Telles</td>
      <td>LB</td>
      <td>84.0</td>
    </tr>
    <tr>
      <th>94</th>
      <td>Y. Brahimi</td>
      <td>LM</td>
      <td>85.0</td>
    </tr>
    <tr>
      <th>323</th>
      <td>F. de Jong</td>
      <td>LDM</td>
      <td>81.0</td>
    </tr>
    <tr>
      <th>293</th>
      <td>S. Giovinco</td>
      <td>CF</td>
      <td>82.0</td>
    </tr>
    <tr>
      <th>187</th>
      <td>Danilo Pereira</td>
      <td>CDM</td>
      <td>83.0</td>
    </tr>
    <tr>
      <th>229</th>
      <td>G. Donnarumma</td>
      <td>GK</td>
      <td>82.0</td>
    </tr>
    <tr>
      <th>371</th>
      <td>J. Corona</td>
      <td>RM</td>
      <td>81.0</td>
    </tr>
    <tr>
      <th>397</th>
      <td>C. Vela</td>
      <td>RW</td>
      <td>81.0</td>
    </tr>
    <tr>
      <th>171</th>
      <td>H. Ziyech</td>
      <td>RAM</td>
      <td>83.0</td>
    </tr>
    <tr>
      <th>194</th>
      <td>Pizzi</td>
      <td>LCM</td>
      <td>83.0</td>
    </tr>
    <tr>
      <th>274</th>
      <td>S. Coates</td>
      <td>RCB</td>
      <td>82.0</td>
    </tr>
    <tr>
      <th>329</th>
      <td>H. Lozano</td>
      <td>LS</td>
      <td>81.0</td>
    </tr>
    <tr>
      <th>170</th>
      <td>Q. Promes</td>
      <td>RM</td>
      <td>83.0</td>
    </tr>
    <tr>
      <th>436</th>
      <td>Rafa</td>
      <td>RW</td>
      <td>80.0</td>
    </tr>
    <tr>
      <th>205</th>
      <td>Oscar</td>
      <td>LCM</td>
      <td>83.0</td>
    </tr>
    <tr>
      <th>426</th>
      <td>A. Onana</td>
      <td>GK</td>
      <td>80.0</td>
    </tr>
  </tbody>
</table>
</div>



Already, we know that our team is slightly below average. As for player positions, we have a bunch of Acronyms, lets use the categories from the link above to analyze if we can manage to put together a '4-3-3' or '4-4-2' formation.
<br>For Defensive Positions we have ['CB','LB','RB','LWB','RWB','SW']
<br>For Midfield Positions we have ['DM','CM','AM','LM','RM']
<br>For Offensive Positions we have ['CF','S','SS']
<br>And we always need a ['GK']
<br>
Let's make a new column to label our players.


```python
defensive = ['CB','LB','RB','LWB','RWB','SW']
midfield = ['DM','CM','AM','LM','RM']
offensive = ['CF','S','SS'] 
offMid = ['RW','LW','LF','RF']
# I added wings(RW, LW, LF, RF) after looking at https://www.fifauteam.com/fifa-ultimate-team-positions-and-tactics/
goalkeeper = ['GK']
top_vpw['Position Type'] = top_vpw['Position'].apply(lambda x: "Defense" if any(pos in x for pos in defensive)
                                                                else "Midfield" if any(pos in x for pos in midfield)
                                                                else "Offensive" if any(pos in x for pos in offensive)
                                                                else "OffMid" if any(pos in x for pos in offMid)
                                                                else "Goalie" if "GK" in x
                                                                else "Error")
```


```python
top_vpw[['Position','Position Type']]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Position</th>
      <th>Position Type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>226</th>
      <td>RCB</td>
      <td>Defense</td>
    </tr>
    <tr>
      <th>166</th>
      <td>CAM</td>
      <td>Midfield</td>
    </tr>
    <tr>
      <th>114</th>
      <td>LCM</td>
      <td>Midfield</td>
    </tr>
    <tr>
      <th>418</th>
      <td>CAM</td>
      <td>Midfield</td>
    </tr>
    <tr>
      <th>173</th>
      <td>LM</td>
      <td>Midfield</td>
    </tr>
    <tr>
      <th>345</th>
      <td>LS</td>
      <td>Offensive</td>
    </tr>
    <tr>
      <th>342</th>
      <td>LB</td>
      <td>Defense</td>
    </tr>
    <tr>
      <th>112</th>
      <td>LB</td>
      <td>Defense</td>
    </tr>
    <tr>
      <th>94</th>
      <td>LM</td>
      <td>Midfield</td>
    </tr>
    <tr>
      <th>323</th>
      <td>LDM</td>
      <td>Midfield</td>
    </tr>
    <tr>
      <th>293</th>
      <td>CF</td>
      <td>Offensive</td>
    </tr>
    <tr>
      <th>187</th>
      <td>CDM</td>
      <td>Midfield</td>
    </tr>
    <tr>
      <th>229</th>
      <td>GK</td>
      <td>Goalie</td>
    </tr>
    <tr>
      <th>371</th>
      <td>RM</td>
      <td>Midfield</td>
    </tr>
    <tr>
      <th>397</th>
      <td>RW</td>
      <td>OffMid</td>
    </tr>
    <tr>
      <th>171</th>
      <td>RAM</td>
      <td>Midfield</td>
    </tr>
    <tr>
      <th>194</th>
      <td>LCM</td>
      <td>Midfield</td>
    </tr>
    <tr>
      <th>274</th>
      <td>RCB</td>
      <td>Defense</td>
    </tr>
    <tr>
      <th>329</th>
      <td>LS</td>
      <td>Offensive</td>
    </tr>
    <tr>
      <th>170</th>
      <td>RM</td>
      <td>Midfield</td>
    </tr>
    <tr>
      <th>436</th>
      <td>RW</td>
      <td>OffMid</td>
    </tr>
    <tr>
      <th>205</th>
      <td>LCM</td>
      <td>Midfield</td>
    </tr>
    <tr>
      <th>426</th>
      <td>GK</td>
      <td>Goalie</td>
    </tr>
  </tbody>
</table>
</div>



It looks like we can put together a team, but we would have no backup Defense players if we played a 4-x-x. We have a massive number of Midfielders, two Goalies and five Offensive players. I think we could probably put together a lot better team composition and probably achieve a better 'Overall' score as well. <br>
For our next business question, we are going to take a closer look at positions and put together a team based on a proposed composition.


```python
fifa_df_clean['Position Type'] = fifa_df_clean['Position'].apply(lambda x: "Defense" if any(pos in x for pos in defensive)
                                                                else "Midfield" if any(pos in x for pos in midfield)
                                                                else "Offensive" if any(pos in x for pos in offensive)
                                                                else "OffMid" if any(pos in x for pos in offMid)
                                                                else "Goalie" if "GK" in x
                                                                else "Error")
fifa_df_positions = fifa_df_clean[fifa_df_clean.Wage > 1]
```


```python
fig=plt.figure(figsize=(12,8), dpi= 100, facecolor='w', edgecolor='k')
sns.boxplot(x="Position Type", y="VpW", data=fifa_df_positions)
locs, lab = plt.xticks()
plt.setp(lab, rotation=90);
```


![png](output_78_0.png)


Let's put together a team! We want to grab the top 5 dedicated Offense, the top 4 dedicated Midfield, the top 4 Offensive/Midfield, the top 7 Defensive, and the top 3 Goalies. 


```python
offMid_df = fifa_df_positions[fifa_df_positions["Position Type"] == "OffMid"]
offensive_df = fifa_df_positions[fifa_df_positions["Position Type"] == "Offensive"]
midfield_df = fifa_df_positions[fifa_df_positions["Position Type"] == "Midfield"]
defensive_df = fifa_df_positions[fifa_df_positions["Position Type"] == "Defense"]
goalie_df = fifa_df_positions[fifa_df_positions["Position Type"] == "Goalie"]
top_OffMid = offMid_df.sort_values(by=['VpW'], ascending=False).iloc[:4]
top_offensive = offensive_df.sort_values(by=['VpW'], ascending=False).iloc[:5]
top_defensive = defensive_df.sort_values(by=['VpW'], ascending=False).iloc[:7]
top_midfield = midfield_df.sort_values(by=['VpW'], ascending=False).iloc[:4]
top_goalie = goalie_df.sort_values(by=['VpW'], ascending=False).iloc[:3]
our_team = top_OffMid.append([top_offensive, top_defensive,top_midfield,top_goalie])
our_team
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>No</th>
      <th>ID</th>
      <th>Name</th>
      <th>Age</th>
      <th>Nationality</th>
      <th>Overall</th>
      <th>Potential</th>
      <th>Club</th>
      <th>Value</th>
      <th>Wage</th>
      <th>...</th>
      <th>Marking</th>
      <th>StandingTackle</th>
      <th>SlidingTackle</th>
      <th>GKDiving</th>
      <th>GKHandling</th>
      <th>GKKicking</th>
      <th>GKPositioning</th>
      <th>GKReflexes</th>
      <th>VpW</th>
      <th>Position Type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>397</th>
      <td>397.0</td>
      <td>169416.0</td>
      <td>C. Vela</td>
      <td>29.0</td>
      <td>Mexico</td>
      <td>81.0</td>
      <td>81.0</td>
      <td>Los Angeles FC</td>
      <td>17.5</td>
      <td>15.0</td>
      <td>...</td>
      <td>31.0</td>
      <td>22.0</td>
      <td>14.0</td>
      <td>8.0</td>
      <td>14.0</td>
      <td>8.0</td>
      <td>13.0</td>
      <td>10.0</td>
      <td>1.166667</td>
      <td>OffMid</td>
    </tr>
    <tr>
      <th>436</th>
      <td>436.0</td>
      <td>216547.0</td>
      <td>Rafa</td>
      <td>25.0</td>
      <td>Portugal</td>
      <td>80.0</td>
      <td>83.0</td>
      <td>SL Benfica</td>
      <td>18.5</td>
      <td>17.0</td>
      <td>...</td>
      <td>23.0</td>
      <td>38.0</td>
      <td>31.0</td>
      <td>9.0</td>
      <td>11.0</td>
      <td>11.0</td>
      <td>12.0</td>
      <td>8.0</td>
      <td>1.088235</td>
      <td>OffMid</td>
    </tr>
    <tr>
      <th>76</th>
      <td>76.0</td>
      <td>41.0</td>
      <td>Iniesta</td>
      <td>34.0</td>
      <td>Spain</td>
      <td>86.0</td>
      <td>86.0</td>
      <td>Vissel Kobe</td>
      <td>21.5</td>
      <td>21.0</td>
      <td>...</td>
      <td>67.0</td>
      <td>57.0</td>
      <td>56.0</td>
      <td>6.0</td>
      <td>13.0</td>
      <td>6.0</td>
      <td>13.0</td>
      <td>7.0</td>
      <td>1.023810</td>
      <td>OffMid</td>
    </tr>
    <tr>
      <th>375</th>
      <td>375.0</td>
      <td>190972.0</td>
      <td>E. Salvio</td>
      <td>27.0</td>
      <td>Argentina</td>
      <td>81.0</td>
      <td>81.0</td>
      <td>SL Benfica</td>
      <td>18.5</td>
      <td>19.0</td>
      <td>...</td>
      <td>49.0</td>
      <td>60.0</td>
      <td>56.0</td>
      <td>9.0</td>
      <td>11.0</td>
      <td>9.0</td>
      <td>5.0</td>
      <td>14.0</td>
      <td>0.973684</td>
      <td>OffMid</td>
    </tr>
    <tr>
      <th>345</th>
      <td>345.0</td>
      <td>207877.0</td>
      <td>J. Martínez</td>
      <td>25.0</td>
      <td>Venezuela</td>
      <td>81.0</td>
      <td>84.0</td>
      <td>Atlanta United</td>
      <td>22.5</td>
      <td>14.0</td>
      <td>...</td>
      <td>20.0</td>
      <td>20.0</td>
      <td>15.0</td>
      <td>12.0</td>
      <td>14.0</td>
      <td>14.0</td>
      <td>12.0</td>
      <td>8.0</td>
      <td>1.607143</td>
      <td>Offensive</td>
    </tr>
    <tr>
      <th>293</th>
      <td>293.0</td>
      <td>184431.0</td>
      <td>S. Giovinco</td>
      <td>31.0</td>
      <td>Italy</td>
      <td>82.0</td>
      <td>82.0</td>
      <td>Toronto FC</td>
      <td>20.0</td>
      <td>15.0</td>
      <td>...</td>
      <td>23.0</td>
      <td>29.0</td>
      <td>28.0</td>
      <td>6.0</td>
      <td>3.0</td>
      <td>6.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>1.333333</td>
      <td>Offensive</td>
    </tr>
    <tr>
      <th>329</th>
      <td>329.0</td>
      <td>221992.0</td>
      <td>H. Lozano</td>
      <td>22.0</td>
      <td>Mexico</td>
      <td>81.0</td>
      <td>86.0</td>
      <td>PSV</td>
      <td>24.0</td>
      <td>22.0</td>
      <td>...</td>
      <td>45.0</td>
      <td>35.0</td>
      <td>29.0</td>
      <td>11.0</td>
      <td>10.0</td>
      <td>14.0</td>
      <td>14.0</td>
      <td>10.0</td>
      <td>1.090909</td>
      <td>Offensive</td>
    </tr>
    <tr>
      <th>204</th>
      <td>204.0</td>
      <td>189068.0</td>
      <td>B. Dost</td>
      <td>29.0</td>
      <td>Netherlands</td>
      <td>83.0</td>
      <td>83.0</td>
      <td>Sporting CP</td>
      <td>26.0</td>
      <td>26.0</td>
      <td>...</td>
      <td>38.0</td>
      <td>45.0</td>
      <td>26.0</td>
      <td>6.0</td>
      <td>12.0</td>
      <td>15.0</td>
      <td>11.0</td>
      <td>8.0</td>
      <td>1.000000</td>
      <td>Offensive</td>
    </tr>
    <tr>
      <th>480</th>
      <td>480.0</td>
      <td>199069.0</td>
      <td>V. Aboubakar</td>
      <td>26.0</td>
      <td>Cameroon</td>
      <td>80.0</td>
      <td>82.0</td>
      <td>FC Porto</td>
      <td>18.0</td>
      <td>19.0</td>
      <td>...</td>
      <td>44.0</td>
      <td>23.0</td>
      <td>19.0</td>
      <td>8.0</td>
      <td>10.0</td>
      <td>9.0</td>
      <td>7.0</td>
      <td>8.0</td>
      <td>0.947368</td>
      <td>Offensive</td>
    </tr>
    <tr>
      <th>226</th>
      <td>226.0</td>
      <td>235243.0</td>
      <td>M. de Ligt</td>
      <td>18.0</td>
      <td>Netherlands</td>
      <td>82.0</td>
      <td>91.0</td>
      <td>Ajax</td>
      <td>27.0</td>
      <td>11.0</td>
      <td>...</td>
      <td>84.0</td>
      <td>84.0</td>
      <td>79.0</td>
      <td>12.0</td>
      <td>11.0</td>
      <td>11.0</td>
      <td>12.0</td>
      <td>10.0</td>
      <td>2.454545</td>
      <td>Defense</td>
    </tr>
    <tr>
      <th>342</th>
      <td>342.0</td>
      <td>210035.0</td>
      <td>Grimaldo</td>
      <td>22.0</td>
      <td>Spain</td>
      <td>81.0</td>
      <td>87.0</td>
      <td>SL Benfica</td>
      <td>20.5</td>
      <td>14.0</td>
      <td>...</td>
      <td>73.0</td>
      <td>78.0</td>
      <td>79.0</td>
      <td>7.0</td>
      <td>13.0</td>
      <td>10.0</td>
      <td>7.0</td>
      <td>13.0</td>
      <td>1.464286</td>
      <td>Defense</td>
    </tr>
    <tr>
      <th>112</th>
      <td>112.0</td>
      <td>212462.0</td>
      <td>Alex Telles</td>
      <td>25.0</td>
      <td>Brazil</td>
      <td>84.0</td>
      <td>87.0</td>
      <td>FC Porto</td>
      <td>32.0</td>
      <td>22.0</td>
      <td>...</td>
      <td>80.0</td>
      <td>81.0</td>
      <td>79.0</td>
      <td>13.0</td>
      <td>8.0</td>
      <td>12.0</td>
      <td>11.0</td>
      <td>14.0</td>
      <td>1.454545</td>
      <td>Defense</td>
    </tr>
    <tr>
      <th>274</th>
      <td>274.0</td>
      <td>197655.0</td>
      <td>S. Coates</td>
      <td>27.0</td>
      <td>Uruguay</td>
      <td>82.0</td>
      <td>83.0</td>
      <td>Sporting CP</td>
      <td>21.0</td>
      <td>19.0</td>
      <td>...</td>
      <td>84.0</td>
      <td>85.0</td>
      <td>85.0</td>
      <td>16.0</td>
      <td>13.0</td>
      <td>15.0</td>
      <td>16.0</td>
      <td>13.0</td>
      <td>1.105263</td>
      <td>Defense</td>
    </tr>
    <tr>
      <th>175</th>
      <td>175.0</td>
      <td>207863.0</td>
      <td>Felipe</td>
      <td>29.0</td>
      <td>Brazil</td>
      <td>83.0</td>
      <td>83.0</td>
      <td>FC Porto</td>
      <td>20.0</td>
      <td>22.0</td>
      <td>...</td>
      <td>85.0</td>
      <td>85.0</td>
      <td>79.0</td>
      <td>9.0</td>
      <td>11.0</td>
      <td>14.0</td>
      <td>9.0</td>
      <td>7.0</td>
      <td>0.909091</td>
      <td>Defense</td>
    </tr>
    <tr>
      <th>488</th>
      <td>488.0</td>
      <td>194022.0</td>
      <td>André Almeida</td>
      <td>27.0</td>
      <td>Portugal</td>
      <td>80.0</td>
      <td>80.0</td>
      <td>SL Benfica</td>
      <td>12.0</td>
      <td>15.0</td>
      <td>...</td>
      <td>82.0</td>
      <td>82.0</td>
      <td>79.0</td>
      <td>14.0</td>
      <td>6.0</td>
      <td>7.0</td>
      <td>12.0</td>
      <td>8.0</td>
      <td>0.800000</td>
      <td>Defense</td>
    </tr>
    <tr>
      <th>428</th>
      <td>428.0</td>
      <td>224334.0</td>
      <td>M. Acuña</td>
      <td>26.0</td>
      <td>Argentina</td>
      <td>80.0</td>
      <td>80.0</td>
      <td>Sporting CP</td>
      <td>12.5</td>
      <td>16.0</td>
      <td>...</td>
      <td>78.0</td>
      <td>80.0</td>
      <td>75.0</td>
      <td>8.0</td>
      <td>14.0</td>
      <td>13.0</td>
      <td>13.0</td>
      <td>14.0</td>
      <td>0.781250</td>
      <td>Defense</td>
    </tr>
    <tr>
      <th>166</th>
      <td>166.0</td>
      <td>212523.0</td>
      <td>Anderson Talisca</td>
      <td>24.0</td>
      <td>Brazil</td>
      <td>83.0</td>
      <td>90.0</td>
      <td>Guangzhou Evergrande Taobao FC</td>
      <td>36.5</td>
      <td>18.0</td>
      <td>...</td>
      <td>55.0</td>
      <td>62.0</td>
      <td>42.0</td>
      <td>13.0</td>
      <td>11.0</td>
      <td>13.0</td>
      <td>12.0</td>
      <td>10.0</td>
      <td>2.027778</td>
      <td>Midfield</td>
    </tr>
    <tr>
      <th>114</th>
      <td>114.0</td>
      <td>212198.0</td>
      <td>Bruno Fernandes</td>
      <td>23.0</td>
      <td>Portugal</td>
      <td>84.0</td>
      <td>88.0</td>
      <td>Sporting CP</td>
      <td>40.5</td>
      <td>22.0</td>
      <td>...</td>
      <td>63.0</td>
      <td>66.0</td>
      <td>53.0</td>
      <td>12.0</td>
      <td>14.0</td>
      <td>15.0</td>
      <td>8.0</td>
      <td>14.0</td>
      <td>1.840909</td>
      <td>Midfield</td>
    </tr>
    <tr>
      <th>418</th>
      <td>418.0</td>
      <td>230977.0</td>
      <td>M. Almirón</td>
      <td>24.0</td>
      <td>Paraguay</td>
      <td>80.0</td>
      <td>84.0</td>
      <td>Atlanta United</td>
      <td>19.5</td>
      <td>11.0</td>
      <td>...</td>
      <td>43.0</td>
      <td>53.0</td>
      <td>49.0</td>
      <td>6.0</td>
      <td>9.0</td>
      <td>13.0</td>
      <td>13.0</td>
      <td>12.0</td>
      <td>1.772727</td>
      <td>Midfield</td>
    </tr>
    <tr>
      <th>173</th>
      <td>173.0</td>
      <td>208418.0</td>
      <td>Y. Carrasco</td>
      <td>24.0</td>
      <td>Belgium</td>
      <td>83.0</td>
      <td>86.0</td>
      <td>Dalian YiFang FC</td>
      <td>33.0</td>
      <td>20.0</td>
      <td>...</td>
      <td>58.0</td>
      <td>39.0</td>
      <td>26.0</td>
      <td>9.0</td>
      <td>11.0</td>
      <td>9.0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>1.650000</td>
      <td>Midfield</td>
    </tr>
    <tr>
      <th>229</th>
      <td>229.0</td>
      <td>230621.0</td>
      <td>G. Donnarumma</td>
      <td>19.0</td>
      <td>Italy</td>
      <td>82.0</td>
      <td>93.0</td>
      <td>Milan</td>
      <td>29.0</td>
      <td>23.0</td>
      <td>...</td>
      <td>20.0</td>
      <td>14.0</td>
      <td>16.0</td>
      <td>88.0</td>
      <td>78.0</td>
      <td>72.0</td>
      <td>78.0</td>
      <td>88.0</td>
      <td>1.260870</td>
      <td>Goalie</td>
    </tr>
    <tr>
      <th>426</th>
      <td>426.0</td>
      <td>226753.0</td>
      <td>A. Onana</td>
      <td>22.0</td>
      <td>Cameroon</td>
      <td>80.0</td>
      <td>85.0</td>
      <td>Ajax</td>
      <td>14.5</td>
      <td>14.0</td>
      <td>...</td>
      <td>12.0</td>
      <td>18.0</td>
      <td>14.0</td>
      <td>83.0</td>
      <td>79.0</td>
      <td>85.0</td>
      <td>75.0</td>
      <td>80.0</td>
      <td>1.035714</td>
      <td>Goalie</td>
    </tr>
    <tr>
      <th>237</th>
      <td>237.0</td>
      <td>221087.0</td>
      <td>Pau López</td>
      <td>23.0</td>
      <td>Spain</td>
      <td>82.0</td>
      <td>87.0</td>
      <td>Real Betis</td>
      <td>21.5</td>
      <td>21.0</td>
      <td>...</td>
      <td>19.0</td>
      <td>20.0</td>
      <td>11.0</td>
      <td>81.0</td>
      <td>82.0</td>
      <td>79.0</td>
      <td>83.0</td>
      <td>81.0</td>
      <td>1.023810</td>
      <td>Goalie</td>
    </tr>
  </tbody>
</table>
<p>23 rows × 59 columns</p>
</div>




```python
print("Using our new team, we get a team that costs us: ")
print("$",our_team['Wage'].sum())
print("Which gives us a Value of: ")
print(our_team['Value'].sum()," million Euros")
print("Now our 'Overall' score is: ")
print("Our team's mean 'Overall' score is: ", our_team.Overall.mean())
```

    Using our new team, we get a team that costs us: 
    $ 416.0
    Which gives us a Value of: 
    526.0  million Euros
    Now our 'Overall' score is: 
    Our team's mean 'Overall' score is:  81.78260869565217
    

Oh no! Our 'Value', 'Overall' and even our cost went down, but we have the right players for the positions! <br>
Perhaps we should take into account "starters" and "relief" players, perhaps we could get more granular into positions and formations, develop teams around strategies we might have. Think of what we could do if we had more data such as team performance, full team player data and a subject matter expert to refer to for even more insight. At this point we could examine the traits like "Potential", create new attributes combining Potential and Overall, create an algorithm to maximize an attribute under a certain budget and more! We will stop here though, as this is a homework assignment for a job interview and I don't want to dive any further down the rabbit hole. Thanks for the fun, now I have to get back to working overtime this weekend for my current job!
