# Analysis of Various Machine Learning Classification Techniques in Detecting Heart Disease

Ethan Nguyen, [fa20-523-309](https://github.com/cybertraining-dsc/fa20-523-309), [Edit](https://github.com/cybertraining-dsc/fa20-523-309/blob/master/project/project.md)

{{% pageinfo %}}
## Abstract

This section will be addressed as the project nears completion

Contents

{{< table_of_contents >}}

{{% /pageinfo %}}

**Keywords:** health, healthcare, cardiovascular disease, data analysis

## 1. Introduction

Since cardiovascular diseases are the number 1 cause of death globally, early prevention could help in extending one’s life span and possibly quality of life. Since there are cases where patients do not show any signs of cardiovascular trouble until an event occurs, having an algorithm predict from their medical history would help in picking up on early warning signs a physician may overlook. Or could also reveal additional risk factors and patterns for research on prevention and treatment.

This project will take a high-level overview of common, widely available classification algorithms and analyze their effectiveness for this specific use case. Notable ones include, Gaussian Naive Bayes, K-Nearest Neighbors, and Support Vector Machines.

Additionally, two data sets that contain common features will be used to increase the training and test pool for evaluation. As well as to explore if additional feature types contribute to a better prediction. As it is known that a large set of data is required to reduce the possibility of the algorithm’s overfitting.

## 2. Datasets

-	<https://www.kaggle.com/johnsmith88/heart-disease-dataset>
-	<https://www.kaggle.com/sulianova/cardiovascular-disease-dataset>

The range of creation dates are 1988 and 2019 respectively with different features of which 4 are common between. This does bring up a small hiccup in preprocessing to consider. Namely the possibility of changing diet and culture trends resulting in significantly different trends/patterns within the same age group. As well as possible differences in measurement accuracy. However this large gap is within the scope of the project in exploring which features can help provide an accurate prediction.

This possible phenomenon may be of interest to explore closely if time allows. Whether a trend itself is even present or there is an overarching trend across different cultures and time periods. Or to consider if this difference is significant enough that the data from the various sets needs to be adjusted to normalize the ages to present day.

### 2.1 Dataset Cleaning

The datasets used have already been significantly cleaned from the raw data and has been provided as a csv file. These files were then imported into the python notebook as pandas dataframes for easy manipulation. 

An initial check was made to ensure the integrity of the data matched the description from the source websites. Then some preprocessing was completed to normalize the common features between the datasets. These features were gender, age, and cholesterol levels. The first two adjustments were trivial in conversion however, in the case of cholesterol levels, the 2019 set is on a 1-3 scale while the 1988 dataset provided them as real measurements. A conversion of the 1988 dataset was done based on guidelines found online for the age range of the dataset [^1]. 

### 2.2 Dataset Analysis

From this point on, the 1988 dataset will be referred to as `dav_set` and 2019 data set will be referred to as `sav_set`.

To provide further insight on what to expect and how a model would be applied, the population of the datasets was analysed first. As depicted in Figure 2.1 the population samples of both datasets of gender vs age show the majority of the data is centered around 60 years of age with a growing slope from 30 onwards. 

![Figure 2.1](https://raw.githubusercontent.com/cybertraining-dsc/fa20-523-309/master/project/images/agevssex.jpg)  

**Figure 2.1**: Age vs Gender distributions of the dav_set and sav_set.

This trend appears to signify that the datasets focused solely on an older population or general trend in society of not monitoring heart conditions as closely in the younger generation.

Moving on to Figure 2.2, we see an interesting trend with a significant growing trend in the sav_set in older population having more cardiovascular issues compared to the dav_set. While this cannot be seen in the dav_set. This may be caused by the additional life expectancy or a change in diet as noted in the introduction.

![Figure 2.2](https://raw.githubusercontent.com/cybertraining-dsc/fa20-523-309/master/project/images/agevstarget.jpg)  

**Figure 2.2**: Age vs Target distributions of the dav_set and sav_set.

In Figure 2.3, the probability of having cardiovascular issues between the sets are interesting. In the dav_set the inequality of higher probability could be attributed to the larger female samples in the dataset. With the sav_set having a more equal probability between the genders. 

![Figure 2.3](https://raw.githubusercontent.com/cybertraining-dsc/fa20-523-309/master/project/images/gendervsprobability.jpg)  

**Figure 2.3**: Gender vs Probability of cardiovascular issues of the dav_set and sav_set.

Finally, in Figure 2.4 is the probability vs cholesterol levels. This one is very interesting between the two datasets in terms of trend levels. With the dav_set having a higher risk at normal levels compared to the sav_set. This could be another hint of a societal change across the years or may in fact be due to the low sample size. Especially since the sav_set matches the general consensus of higher cholesterol levels increasing risk of cardiovascular issues [^1].

![Figure 2.4](https://raw.githubusercontent.com/cybertraining-dsc/fa20-523-309/master/project/images/cholesterolvsprobability.jpg)  

**Figure 2.4**: Cholesterol levels vs Probability of cardiovascular issues of the dav_set and sav_set.

To close out this initial analysis is the correlation map of each of the features. From Figure 2.5 and 2.6 it can be concluded that both of these datasets are viable to conduct machine learning as the correlation factor is below the recommended value of 0.8 [^2]. Although we do see the signs of a low sample amount in the dav_set with a higher correlation factor compared to the sav_set.

![Figure 2.5](https://raw.githubusercontent.com/cybertraining-dsc/fa20-523-309/master/project/images/davsetcorrelation.jpg)  

**Figure 2.5**: dav_set correlation matrix.

![Figure 2.6](https://raw.githubusercontent.com/cybertraining-dsc/fa20-523-309/master/project/images/savsetcorrelation.jpg)  

**Figure 2.6**: sav_set correlation matrix.

## 3. Machine Learning Algorithms and Implementation

With many machine learning algorithms already available and many more in development. Selecting the optimal one for an application can be a challenging balance since each algorithm has both its advantages and disadvantages. As mentioned in the introduction, we will explore applying the most common and established algorithms available to the public. 

Starting off, is selecting a library from the most popular ones available. Namely Keras, Pytorch, Tensorflow, and Scikit-Learn. Upon further investigation it was determined that Scikit-Learn would be used for this project. The reason being Scikit-Learn is a great general machine learning library that also includes pre and post processing functions. While Keras, Pytorch, and Tensorflow are targeted for neural networks and other higher-level deep learning algorithms which are outside of the scope of this project at this time [^3]. 

### 3.1 Scikit-Learn and Algorithm Types

Diving further into the Scikit-Learn library, its key strength appears to be the variety of algorithms available that are relatively easy to implement against a dataset. Of those available, they are classified under three different categories based on the approach each takes. They are as follows:

- Classification
    - Applied to problems that require identifying the category an object belongs to.
- Regression
    - For predicting or modeling continuous values.
- Clustering
    - Grouping similar objects into groups.

For this project, we will be investigating the Classification and Clustering algorithms offered by the library due to the nature of our dataset. Since it is a binary answer, the continuous prediction capability of regression algorithms will not fair well. Compared to classification type algorithms which are well suited for determining binary and multi-class classification on datasets [^4]. Along with Clustering algorithms being capable of grouping unlabeled data which is one of the key problem points mentioned in the introduction [^5].

### 3.2 Classification Algorithms

The following algorithms were determined to be candidates for this project based on the documentation available on the Scikit-learn for supervised learning [^6]. 

#### 3.2.1 Support Vector Machines

This algorithm was chosen because classification is one of the target types and has a decent list of advantages that appear to be applicable to this dataset [^4]. 
- Effective in high dimensional spaces as well as if the number dimensions out number samples.
- Is very versatile.

#### 3.2.2 K-Nearest Neighbors

This algorithm was selected due to being a non-parametric method that has been successful in classification applications [^7]. From the dataset analysis, it is appears that the decision boundary may be very irregular which is a strong point of this type of method.

#### 3.2.3 Gaussian Naive Bayes

Is an implementation of the Naive Bayes theorem that has been targeted for classification. The advantages of this algorithm is its speed and requires a small training set compared to more advanced algorithms [^8].

#### 3.2.4 Decision Trees

This algorithm was chosen to investigate another non-parametric method to determine their efficacy against this dataset application. This algorithm also has some advantages over K-Nearest namely [^9]. 
- Simple to interpret and visualize
- Requires little data preparation 
    - Handles numerical and categorical data instead of needing to normalize
- Can validate the model and is possible to audit from a liability standpoint.

### 3.3 Clustering Algorithms

The following algorithms were determined to be candidates for this project based on the table of clustering algorithms available on the Scikit-learn [^10]. 

#### 3.3.1 K-Means 

The usecase for this algorithm is general purpose with even and low number of clusters [^10]. Of which the sav_set appears to have with the even distribution across most of the features.

#### 3.3.2 Mean-shift

This algorithm was chosen for its strength in dealing with uneven cluster sizes and non-flat geometry [^10]. Though it is not easily scalable the application of our small dataset size might be of interest.

#### 3.3.3 Spectral Clustering

As an inverse, this algorithm was chosen for its strength with fewer uneven clusters [^10]. In comparison to Mean-shift, this maybe the better algorithm for this application. 

## 4. Results & Discussion

### 4.1 Algorithm Metrics

The metrics used to determine the viability of each of the algorithms are precision, recall, and f1-score. These are simple metrics based on the values from a confusion matrix which is a visualization of the False and True Positives and Negatives. Precision is essentially how accurate was the algorithm in classifying each data point. This however, is not a good metric to solely base performance as precision does not account for imbalanced distributions within a dataset [^11]. 

This is where the recall metric comes in which is defined as how many samples were accurately classified by the algorithm. This is a more versatile metric as it can compensate for imbalanced datasets. While it may not be in our case as seen in the dataset analysis where we have a relatively balanced ratio. It still gives great insight on the performance for our application.

Finally is the f1-score which is the harmonic mean of the precision and recall metric [^11]. This will be the key metric we will mainly focus on as it strikes a good balance between the two more primitive metrics. Since one may think in medical applications one would want to maximize recall, it is at the cost of precision which ends up in more false predictions which is essentially an overfitting scenario [^11]. Something that reduces the viability of the model to the application especially since we have a relatively balanced dataset, more customized weighting is not as necessary.

The metrics for each algorithm implementation is as follows.

#### 4.1.1 Support Vector Machines

**Table 4.1:** dav_set metrics
|             | Precision | Recall | f1-score |
|-------------|-----------|--------|----------|
| No Disease  | 0.99      | 0.94   | 0.96     |
| Has Disease | 0.95      | 0.99   | 0.97     |

**Table 4.2:** sav_set metrics
|             | Precision | Recall | f1-score |
|-------------|-----------|--------|----------|
| No Disease  | 0.99      | 0.94   | 0.96     |
| Has Disease | 0.95      | 0.99   | 0.97     |

#### 4.1.2 K-Nearest Neighbors

**Table 4.3:** dav_set metrics
|             | Precision | Recall | f1-score |
|-------------|-----------|--------|----------|
| No Disease  | 0.88      | 0.86   | 0.87     |
| Has Disease | 0.87      | 0.90   | 0.88     |

**Table 4.4:** sav_set metrics
|             | Precision | Recall | f1-score |
|-------------|-----------|--------|----------|
| No Disease  | 0.62      | 0.74   | 0.67     |
| Has Disease | 0.67      | 0.54   | 0.60     |

#### 4.1.3 Gaussian Naive Bayes

**Table 4.5:** dav_set metrics
|             | Precision | Recall | f1-score |
|-------------|-----------|--------|----------|
| No Disease  | 0.88      | 0.81   | 0.84     |
| Has Disease | 0.83      | 0.90   | 0.86     |

**Table 4.6:** sav_set metrics
|             | Precision | Recall | f1-score |
|-------------|-----------|--------|----------|
| No Disease  | 0.56      | 0.90   | 0.69     |
| Has Disease | 0.72      | 0.28   | 0.40     |

#### 4.1.4 Decision Trees

**Table 4.7:** dav_set metrics
|             | Precision | Recall | f1-score |
|-------------|-----------|--------|----------|
| No Disease  | 0.92      | 0.97   | 0.95     |
| Has Disease | 0.97      | 0.93   | 0.95     |

**Table 4.8:** sav_set metrics
|             | Precision | Recall | f1-score |
|-------------|-----------|--------|----------|
| No Disease  | 0.71      | 0.80   | 0.75     |
| Has Disease | 0.76      | 0.66   | 0.71     |

#### 4.1.5 K-Means 

**Table 4.7:** dav_set metrics
|             | Precision | Recall | f1-score |
|-------------|-----------|--------|----------|
| No Disease  | 0.22      | 0.29   | 0.25     |
| Has Disease | 0.12      | 0.09   | 0.10     |

**Table 4.8:** sav_set metrics
|             | Precision | Recall | f1-score |
|-------------|-----------|--------|----------|
| No Disease  | 0.51      | 0.69   | 0.59     |
| Has Disease | 0.52      | 0.34   | 0.41     |

#### 4.1.6 Mean-shift

Algorithm is still being implemented

#### 4.1.7 Spectral Clustering

 **Table 4.7:** dav_set metrics
|             | Precision | Recall | f1-score |
|-------------|-----------|--------|----------|
| No Disease  | 0.86      | 0.74   | 0.79     |
| Has Disease | 0.79      | 0.89   | 0.84     |

**Table 4.8:** sav_set metrics
|             | Precision | Recall | f1-score |
|-------------|-----------|--------|----------|
| No Disease  | 0.56      | 0.57   | 0.57     |
| Has Disease | 0.56      | 0.56   | 0.56     |

## 5. Conclusion

be addressed as the project nears completion

## Project Timeline

The following is a plan for the rest of the semester, using the due dates for Assignments 8-11 as milestone dates.

### October 26

- Explore options on normalizing or conversion of the features to connect between the datasets and determine if they are viable to add to the project.
    - A large portion of time was dedicated to researching this point. Unfortunately, nothing of note was discovered to realize this aspect of the plan. 
- Explore ML models and frameworks and determine viability for the project.
    - Additional information on the algorithms selected. Mainly the underlying calculation, theory, and disadvantages to compensate for has been added to the Nov 2nd milestone date.
- Update Report.

### November 2

- Commence build out of ML models.
    - Tuning of hyperparameters as necessary.
- Update Report.
- Milestone Update
    - Delays were encountered due to external events.
    - Underestimated the amount of time needed to building the ML models. 
        - Task of adding additional information on algorithms selected has been pushed back further due to above.
    - Initial implementations do not look very promising
        - It is suspected that over fitting is occurring with the base line hyperparameters for the dav_set.
        - Accuracy results for the sav_set are not as high as expected for the selected algorithms.
        - Further investigation and experimentation is required to implement the clustering type algorithms. Suspect would need to look into dimensionality reduction in order to obtain better accuracy results.
    - A general download function appears to not be possible. The website that hosts the datasets used requires an API key for script interactions that is unique to each user. 

### November 9

- ML Models should be complete.
- Start analysis of the various models and their viability.
    - Additional tuning of hyperparameters as necessary.
- Update Report
- Milestone Update
    - Additional delays were encountered due to external classes.
        - It is expected to have a surplus of time between this milestone and the deadline to catch up.
    - All but one algorithm has been implemented.
        - Some light tuning was conducted however, results did not significantly improve.

### November 16

- Finalize Report and findings.

## References

[^1]: WebMD. 2020. Understanding Your Cholesterol Report. [online] Available at: <https://www.webmd.com/cholesterol-management/understanding-your-cholesterol-report> [Accessed 21 October 2020].  

[^2]: R, V., 2020. Feature Selection — Correlation And P-Value. [online] Medium. Available at: <https://towardsdatascience.com/feature-selection-correlation-and-p-value-da8921bfb3cf> [Accessed 21 October 2020].

[^3]: Stack Overflow. 2020. Differences In Scikit Learn, Keras, Or Pytorch. [online] Available at: <https://stackoverflow.com/questions/54527439/differences-in-scikit-learn-keras-or-pytorch> [Accessed 27 October 2020].

[^4]: Scikit-learn.org. 2020. 1.4. Support Vector Machines — Scikit-Learn 0.23.2 Documentation. [online] Available at: <https://scikit-learn.org/stable/modules/svm.html#classification> [Accessed 27 October 2020].

[^5]: Scikit-learn.org. 2020. 2.3. Clustering — Scikit-Learn 0.23.2 Documentation. [online] Available at: <https://scikit-learn.org/stable/modules/clustering.html#clustering> [Accessed 27 October 2020].

[^6]: Scikit-learn.org. 2020. 1. Supervised Learning — Scikit-Learn 0.23.2 Documentation. [online] Available at: <https://scikit-learn.org/stable/supervised_learning.html#supervised-learning> [Accessed 27 October 2020].

[^7]: Scikit-learn.org. 2020. 1.6. Nearest Neighbors — Scikit-Learn 0.23.2 Documentation. [online] Available at: <https://scikit-learn.org/stable/modules/neighbors.html> [Accessed 27 October 2020].

[^8]: Scikit-learn.org. 2020. 1.9. Naive Bayes — Scikit-Learn 0.23.2 Documentation. [online] Available at: <https://scikit-learn.org/stable/modules/naive_bayes.html> [Accessed 27 October 2020].

[^9]: Scikit-learn.org. 2020. 1.10. Decision Trees — Scikit-Learn 0.23.2 Documentation. [online] Available at: <https://scikit-learn.org/stable/modules/tree.html> [Accessed 27 October 2020].

[^10]: Scikit-learn.org. 2020. 2.3. Clustering — Scikit-Learn 0.23.2 Documentation. [online] Available at: <https://scikit-learn.org/stable/modules/clustering.html#clustering> [Accessed 27 October 2020].

[^11]: Mianaee, S., 2020. 20 Popular Machine Learning Metrics. Part 1: Classification & Regression Evaluation Metrics. [online] Medium. Available at: <https://towardsdatascience.com/20-popular-machine-learning-metrics-part-1-classification-regression-evaluation-metrics-1ca3e282a2ce> [Accessed 10 November 2020].