# Coursera Practical Machine Learning Project
Aaron Galluzzi  
December 1, 2017  

# Introduction

## Background

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behaviour, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, we will be using data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways:

1. Exactly according to specification (Class A)
2. Throwing the elbows to the front (Class B)
3. Lifting the dumbell only halfway (Class C)
4. Lowering the dumbell only halfway (Class D)
5. Throwing the hips to the front (Class E)

Only Class A represents the correct performance. More information is available from the following website: (http://groupware.les.inf.puc-rio.br/har) (see the section on Weight Lifting Exercise Dataset).

## Data

The training data is available from the following link:

(https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv)

The test data is available is available from the following link:

(https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv)

## Objective

The objective of the project is to predict the manner in which they exercise. This is based on the "classe" variable in the exercise dataset.

# Data Processing

## Getting, Loading and Cleaning the Data

We first load the R packages needed for the analysis and then download the training and testing datasets from the given URLs.

```r
library(caret)
```

```
## Loading required package: lattice
```

```
## Loading required package: ggplot2
```

```r
library(rpart)
library(rpart.plot)
library(RColorBrewer)
library(rattle)
```

```
## Rattle: A free graphical interface for data science with R.
## Version 5.1.0 Copyright (c) 2006-2017 Togaware Pty Ltd.
## Type 'rattle()' to shake, rattle, and roll your data.
```

```r
library(randomForest)
```

```
## randomForest 4.6-12
```

```
## Type rfNews() to see new features/changes/bug fixes.
```

```
## 
## Attaching package: 'randomForest'
```

```
## The following object is masked from 'package:rattle':
## 
##     importance
```

```
## The following object is masked from 'package:ggplot2':
## 
##     margin
```


```r
trainURL <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
testURL <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

training <- read.csv(url(trainURL),na.strings=c("NA","#DIV/0!",""))
testing <- read.csv(url(testURL),na.strings=c("NA","#DIV/0!",""))
```
The training dataset contains 19622 observations and 160 variables, and the testing dataset contains 20 observations and the same variables as the training set. Our goal is to predict the outcome of the classe variable using the training dataset.

Please note that the first 7 variables in both the training and the testing dataset contain system-generated information that is not particularly useful in terms of prediction (e.g. user name, time stamp), so these will be deleted from both datasets, as shown below.


```r
training <- training[,-c(1:7)]
testing <- testing[,-c(1:7)]
```

In addition, we will be deleting columns (predictors) that contain any missing values.

```r
training <- training[,colSums(is.na(training)) == 0]
testing <- testing[,colSums(is.na(testing)) == 0]
```

The cleaned datasets now have 53 variables, the same first 52 variables plus the classe variable we will be using for prediction.

## Data Partitioning
In order to get out-of-sample errors, we split the cleaned training dataset into a training dataset for prediction and a validation dataset.


```r
set.seed(1975)
inTrain <- createDataPartition(training$classe, p=0.6, list=FALSE)
trainData <- training[inTrain,]; validData <- training[-inTrain,]
```

# Prediction Algorithms

We will be using classification trees and random forests.

## Classification Trees
Here we will be considering a k-fold cross-validation prior to fitting the classification tree, where k=5, using the trainControl function (the default setting is k=10). No variables were transformed.

```r
# Apply 5-fold cross-validation
control <- trainControl(method="cv", number=5)

# Fit a classification tree, using the cross-validation as control
modFit_tree <- train(classe ~ ., data=trainData, method="rpart", trControl=control)
print(modFit_tree, digits=4)
```

```
## CART 
## 
## 11776 samples
##    52 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (5 fold) 
## Summary of sample sizes: 9421, 9421, 9420, 9421, 9421 
## Resampling results across tuning parameters:
## 
##   cp       Accuracy  Kappa  
##   0.03560  0.5086    0.35788
##   0.05984  0.4411    0.25086
##   0.11450  0.2995    0.02318
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was cp = 0.0356.
```

```r
# Plot the trees 
fancyRpartPlot(modFit_tree$finalModel)
```

![](pracmlproject_files/figure-html/unnamed-chunk-6-1.png)<!-- -->

```r
# Generate predictions using the validaton set
predict_tree <- predict(modFit_tree, validData)

# Show the prediction results
conf_tree <- confusionMatrix(validData$classe, predict_tree)
print(conf_tree)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2028   32  167    0    5
##          B  658  495  365    0    0
##          C  609   46  713    0    0
##          D  586  226  474    0    0
##          E  202  193  390    0  657
## 
## Overall Statistics
##                                           
##                Accuracy : 0.4962          
##                  95% CI : (0.4851, 0.5073)
##     No Information Rate : 0.5204          
##     P-Value [Acc > NIR] : 1               
##                                           
##                   Kappa : 0.3415          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.4967  0.49899  0.33807       NA  0.99245
## Specificity            0.9458  0.85074  0.88583   0.8361  0.89073
## Pos Pred Value         0.9086  0.32609  0.52120       NA  0.45562
## Neg Pred Value         0.6340  0.92146  0.78450       NA  0.99922
## Prevalence             0.5204  0.12643  0.26880   0.0000  0.08437
## Detection Rate         0.2585  0.06309  0.09087   0.0000  0.08374
## Detection Prevalence   0.2845  0.19347  0.17436   0.1639  0.18379
## Balanced Accuracy      0.7212  0.67487  0.61195       NA  0.94159
```

```r
# Show the accuracy.
accuracy_tree <- conf_tree$overall[1]
accuracy_tree
```

```
##  Accuracy 
## 0.4961764
```
From the confusion matrix above, we see that the accuracy rate is approximately 50%, and thus the out-of-sample error is approximately 50%. The low accuracy rate demonstrates that classification trees do not fit the classe variable well.

## Random Forests
We will now try fitting a prediction model using random forests.

```r
# Fit a random forest model, using the cross-validation as control
modFit_rf <- train(classe ~ ., data=trainData, method="rf", trControl=control)
print(modFit_rf, digits=4)
```

```
## Random Forest 
## 
## 11776 samples
##    52 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (5 fold) 
## Summary of sample sizes: 9421, 9421, 9421, 9421, 9420 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy  Kappa 
##    2    0.9888    0.9858
##   27    0.9901    0.9874
##   52    0.9823    0.9777
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 27.
```

```r
# Plot the fit.
plot(modFit_rf)
```

![](pracmlproject_files/figure-html/unnamed-chunk-7-1.png)<!-- -->

```r
# Generate predictions using the validation set
predict_rf <- predict(modFit_rf, validData)

# Show the prediction results
conf_rf <- confusionMatrix(validData$classe, predict_rf)
print(conf_rf)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2227    4    0    0    1
##          B   21 1494    3    0    0
##          C    0    3 1360    5    0
##          D    0    1   22 1263    0
##          E    0    1    1    5 1435
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9915          
##                  95% CI : (0.9892, 0.9934)
##     No Information Rate : 0.2865          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9892          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9907   0.9940   0.9812   0.9921   0.9993
## Specificity            0.9991   0.9962   0.9988   0.9965   0.9989
## Pos Pred Value         0.9978   0.9842   0.9942   0.9821   0.9951
## Neg Pred Value         0.9963   0.9986   0.9960   0.9985   0.9998
## Prevalence             0.2865   0.1916   0.1767   0.1622   0.1830
## Detection Rate         0.2838   0.1904   0.1733   0.1610   0.1829
## Detection Prevalence   0.2845   0.1935   0.1744   0.1639   0.1838
## Balanced Accuracy      0.9949   0.9951   0.9900   0.9943   0.9991
```

```r
# Show the accuracy.
accuracy_rf <- conf_rf$overall[1]
accuracy_rf
```

```
##  Accuracy 
## 0.9914606
```
As can be seen above, the accuracy rate is approximately 99%, and thus the out-of-sample error rate is about 1%. The high accuracy rate may be due to the fact that the predictors are highly correlated. The random forest method chooses a subset of predictors at each split and decorrelates the trees, leading to high accuracy, although at the expense of interpretability and computational efficiency.

# Prediction on Test Dataset
We will now use random forests to predict the outcome variable classe for the test dataset. 

```r
# Generate predictions using the test data.
finalPredict <- predict(modFit_rf, testing)
finalPredict
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```
