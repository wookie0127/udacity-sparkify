# udacity-sparkify-for-capstone-project
This is one of capstone projects of udacity data scientist nanodegree course.
In this project, called *Sparkify*, we predict churn.
For more details, Check this [Link](https://medium.com/@giwooklee/sparkify-udacity-data-scientist-nanodegree-6548542f53ce)

## 1. Motivation

Most customers often do not use services for any reason or no reason so that appropriate marketing is essential to prevent user churn. However, marketing to prevent churn for all users is very inefficient in terms of cost. What if, along with efficient marketing, you could identify users who are likely to leave in advance? 

In this project, analyze user logs and build machine learning model to find users who are most likely to churn using Spark

## 2. Datasets

- **User activity dataset** from [Udacity](https://www.udacity.com/).  
  The dataset logs user information such as user name, song listened, subscription level and so on.   

The full dataset is ~12GB but only 1% of full dataset is used to analyze and build model

## 3. Steps

1. Load and Clean Dataset

   - Load subset from JSON
   - Assess missing values

2. Exploratory data analysis

   - Defind churn; "Cancellation Confirmation" in *page* field
   - Analyze and Explore data to find characteristics of each group; churn and nonchurn
   - Overview of columns
     |Column name|Description| etc|
     |---|---|---|
     |artist| singer | |
     |auth | user status | Logged Out / Cancelled / Guest / Logged In |
     |firstName| user first name | |
     |gender| user gender | (M / F) |
     |itemInSession| unique ID in Session| |
     |lastName| user last name| |
     |length| song length| |
     |level| subscription level | Free or paid|
     |location| user location | |
     |method| HTTP method | GET or PUT|
     |page| page user visited | ex) About, Cancel |
     |registration| user registered datetime| |
     |sessionId| user session id | |
     |song| song name | |
     |status| HTTP return code | 200 / 307 / 404|
     |ts| time collected data| |
     |userAgent| user web agent | |
     |userId| user id| |

3. Feature engineering

   - Create features on per user basis:
     - user level in a session
     - Gender of user
     - number of artists, number of songs, and number of itemInsession in a session
     - number of page visits
   - Scale features
   - Compile feature engineering code to scale up later

4. Develop machine learning pipeline

   - Split training and testing sets, 8:2
   - Choose evaluation metrics
   - Train machine learning model, and evaluate model performance
   - Initial model evaluation with:
     - Logistic regression ([documentation](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.classification.LogisticRegression.html))
     - Random forest ([documentation](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.classification.RandomForestClassifier.html))
     - LinearSVC ([documentation](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.classification.LinearSVC.html))
     - Gradient-boosted Trees Classifier ([documentation](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.classification.GBTClassifier.html))


## 4. Results

- Model performance on testset:

    |f1 score| 0.97599|
    |---|---|
    |precision| 0.973155|
    |recall| 0.981279|

    Note that model is not tuned. the model performance could be further improved by tuning broader ranges of hyperparameters.

- after applying undersampling, the performance dropped a lot...
  - this needs more time to investigate...


## 5. File structure

- `Sparkify.ipynb`: exploratory data analysis, data preprocessing, and pilot development of machine learning model on local machine using data subset.
- `mini_sparkify_event_data.json`: subset of user activity data.
