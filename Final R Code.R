
library(psych)
library(ggplot2)
library(ggthemes)
library(dplyr)
library(DataExplorer)
library(knitr)
library(GGally)
library(missForest)
library(randomForest)
library(Hmisc)
library(earth)
library(caret)
library(gbm)
library(randomForest)
library(parallel)
library(doParallel)


#For reproducibility of the results, the data was loaded to and accessed from a Github repository. 


#Data to build models
data1 <- read.csv("https://raw.githubusercontent.com/624-Group2/Project-2/master/StudentData.csv", header=TRUE, sep=",")

#Data to make predictions on 
toPred <- read.csv("https://raw.githubusercontent.com/624-Group2/Project-2/master/StudentEvaluation-%20TO%20PREDICT.csv")



#Data Exploration and Statistic Measures

##Missing and Zero Values

plot_missing(data1, title="Manufacturing Data - Missing Values (%)")
kable(sapply(data1, FUN = function(x) sum(is.na(x))))

data1 %>%
  group_by(Brand.Code) %>%
  summarise_all(funs(sum(is.na(.))))

table(data1$Brand.Code)




##Descriptive Statistics and Data Exploration

#Use Describe Package to calculate Descriptive Statistic
(CC_des <- describe(data1, na.rm=TRUE, interp=FALSE, skew=TRUE, ranges=TRUE, trim=.1, type=3, check=TRUE, fast=FALSE, quant=c(.1,.25,.75,.90), IQR=TRUE))




##We have siginificant skewness in a number of variables, and, depending on our choice of regression technique may require transformation.   


vis1 <- data1 %>% select(-PH -Brand.Code)

par(mfrow=(c(1,1)))
ggplot(stack(vis1), aes(values, fill=ind))+
  facet_wrap(~ind, scales = "free") + 
  geom_histogram() +
  theme_pander()+
  theme(legend.position="none")



ggplot(stack(vis1), aes(x = ind, y = values, fill=ind))+
  facet_wrap(~ind, scales = "free") + 
  geom_boxplot() +
  theme_pander() +
  theme(legend.position="none")




##Correlation

ggcorr(data1, method = "pairwise", label=TRUE, nbreaks=6)



#Below is a look at the unique values for each variable. As mentioned, it appears that some of these variables represent various non continuous settings. 

apply(data1, 2, function(x)length(unique(x)))

##Data Manipulation

###Dummy Variables

data1$A <- ifelse(data1$Brand.Code == "A", 1, 0)
data1$B <- ifelse(data1$Brand.Code == "B", 1, 0)
data1$C <- ifelse(data1$Brand.Code == "C", 1, 0)
data1$D <- ifelse(data1$Brand.Code == "D", 1, 0)
data1 <- data1 %>% select(-Brand.Code)


toPred$A <- ifelse(toPred$Brand.Code == "A", 1, 0)
toPred$B <- ifelse(toPred$Brand.Code == "B", 1, 0)
toPred$C <- ifelse(toPred$Brand.Code == "C", 1, 0)
toPred$D <- ifelse(toPred$Brand.Code == "D", 1, 0)
toPred <- toPred %>% select(-Brand.Code)



###Handling Missing Values

missingPH <- which(is.na(data1$PH))
data1 <- data1[-missingPH, ]


#The following will impute the data. For efficiency the data sets can be loaded from github below.
# ##For Windows to run in parallel
# library(parallel)
# library(doParallel)
# 
# cluster <- makeCluster(detectCores() - 1) # convention to leave 1 core for OS
# registerDoParallel(cluster)
# 
# #impute missing training data
# set.seed(123)
# dfImputed <- missForest(data1, parallelize = 'forests')
# write.csv(dfImputed$ximp, "StudentDataImputedMF")
# 
# #impute missing test data
# set.seed(123)
# predImputed <- missForest(toPred, parallelize = 'forests')
# write.csv(predImputed$ximp, "PredictImputedMF")
# 
# 
# #turn off parallel processing
# stopCluster(cluster)
# #resume use of the sequential backend
# registerDoSEQ()

#read imputed train set
imputed <- read.csv("https://raw.githubusercontent.com/624-Group2/Project-2/master/StudentDataImputedMF.csv")
imputed <- imputed %>% select(-X)
sum(is.na(imputed))

#read imputed test set
predictImp <- read.csv("https://raw.githubusercontent.com/624-Group2/Project-2/master/PredictImputedMF")
predictImp <- predictImp %>% select(-X)
sum(is.na(predictImp))


#Test & Training Sets
#Prior to testing our models on the actual prediction data set, it is prudent to evaluate our models against data where the response is known, so our predictions can be compared. 

smp <- floor(0.70 * nrow(imputed))


set.seed(123)
train_index <- sample(seq_len(nrow(imputed)), size = smp, replace = FALSE)

train_set <- imputed[train_index, ]
validation_set <- imputed[-train_index, ]


#Separate X(predictors) and Y(response) values for the validation set for use when necessary
validX <- validation_set %>%  select(-PH)
validY <- validation_set %>%  select(PH)





# Linear Regression


## Ordinary Linear Regression


set.seed(123)
lmTune <- train(PH ~ ., data=train_set, method='lm', preProcess=c('BoxCox', 'center', 'scale'))


RMSE(pred = lmTune$finalModel$fitted.values, obs = train_set$PH)
R2(pred = lmTune$finalModel$fitted.values, obs = train_set$PH)

#Ordinary Linear Regression against holdout validation set
linP <- predict(lmTune, newdata = validX)
postResample(pred = linP, obs = validY$PH)[1:2]



## Partial Least Squares

ctrl <- trainControl(method = "cv")

set.seed(123)
plsTune <- train(PH ~ ., data=train_set, method='pls', preProcess=c('BoxCox', 'center', 'scale'), tuneLength=5, trControl=ctrl)
summary(plsTune)

RMSE(pred = plsTune$finalModel$fitted.values, obs = train_set$PH)
R2(pred = plsTune$finalModel$fitted.values, obs = train_set$PH)

#PLS against holdout validation set
plsP <- predict(plsTune, newdata = validX)
postResample(pred = plsP, obs = validY$PH)[1:2]




## Ridge-Regression


set.seed(100)
ridgeGrid <- data.frame(.lambda=seq(0, 0.1, length=15))
ridgeTune <- train(PH ~ ., data=train_set, method='ridge', preProcess=c('BoxCox', 'center', 'scale'), tuneGrid=ridgeGrid, trControl=ctrl)
ridgeTune


ridP <- predict(ridgeTune, newdata = validX)
postResample(pred = ridP, obs = validY$PH)[1:2]




# MARS

#parallel processing
library(parallel)
library(doParallel)

cluster <- makeCluster(detectCores() - 1) 
registerDoParallel(cluster)


trainX <- train_set %>%  select(-PH)
trainY <- train_set %>%  select(PH)



indx <- createFolds(trainY$PH, returnTrain = TRUE)
ctrl <- trainControl(method = "cv", index = indx)

set.seed(100)
mars1 <- train(x = trainX, y = trainY$PH,
               method = "earth",
               tuneGrid = expand.grid(degree = 1:3, nprune = 1:30),
               trControl = ctrl)

mars1

plot(mars1)


#MARS against holdout validation set
marsP <- predict(mars1, newdata = validX)
postResample(pred = marsP, obs = validY$PH)[1:2]


##Support Vector Machines

set.seed(123)
svmPTune <- train(x = trainX, y = trainY$PH,
                  method = "svmRadial",
                  preProc = c("center", "scale"),
                  tuneLength = 14,
                  trControl = trainControl(method = 'cv'))
svmPTune
plot(svmPTune, 
     scales = list(x = list(log = 2), 
                   between = list(x = .5, y = 1))) 


vip <- varImp(svmPTune)
df2 <- data.frame(vip$importance, stringsAsFactors = FALSE)
vip

#SVM against holdout validation set
svmP <- predict(svmPTune, newdata = validX)
postResample(pred = svmP, obs = validY$PH)[1:2]


##Neural Network
  
nnetGrid <- expand.grid(decay = c(0, 0.01, .1), 
                        size = c(1:10), 
                        bag = FALSE)


set.seed(100)
nnetTune <- train(x = trainX, y = trainY$PH,
                  method = "avNNet",
                  tuneGrid = nnetGrid,
                  trControl = ctrl,
                  preProc = c("center", "scale"),
                  linout = TRUE,
                  trace = FALSE,
                  maxit = 100, 
                  allowParallel = TRUE)

#turn off parallel processing
stopCluster(cluster)
#resume use of the sequential backend
registerDoSEQ()

nnetTune

#NNET against holdout validation set
nnetP <- predict(nnetTune, newdata = validX)
postResample(pred = nnetP, obs = validY$PH)[1:2]




##K-Nearest Neighbors

##The optimal model using K-Nearest Neighbors shows an RMSE of 0.1221 and R^2 of 0.496 The corresponding final value used for the model was k = 5. The most important predictor is Mnf.Flow which takes about 50% of the weight followed by Bowl.Setpoint, Filler.Lever, Usage.cont Pressure.Setpoint etc. For resample data, the end result for RMSE is 0.1234 and R^2 is 0.526

set.seed(100)
knnTune <- train(PH ~ ., data=train_set, method = "knn", preProcess = c('BoxCox', 'center', 'scale'), tuneLength = 10, trControl = ctrl)

plot(varImp(knnTune), top = 10)
plot(knnTune)
knnTune

knnP <- predict(knnTune, newdata = validX)
postResample(pred = knnP, obs = validY$PH)


#Trees

#Impute the data with MissingForest, a Random Forests approach


data <- read.csv("https://raw.githubusercontent.com/624-Group2/Project-2/master/StudentDataImputedMF.csv")
# Take 80% of data as training
rownums <- sample(1:nrow(data), round(nrow(data)*0.8))
train <- data[rownums, ]
test <- data[-rownums, ]

x <- train
x$PH <- NULL

# Get Best mTry parameter = 20
set.seed(1)
res <- tuneRF(x = x, y = train$PH, ntreeTry = 500, doBest = TRUE)
print(res)



#Use Grid Search to identify optimal hyperparameters

# Establish a list of possible values for mtry, nodesize and sampsize
mtry <- 20 #From doBest above; alternately, can use: seq(4, ncol(x) * 0.8, 2)
nodesize <- seq(4, 6, 1) # Originally seq(3,8,2), which returned value of 5 as optimal
sampsize <- 2054 # 80% more effective than 70%: round(nrow(x) * c(0.7, 0.8))

# Create a data frame containing all combinations 
hyper_grid <- expand.grid(mtry = mtry, nodesize = nodesize, sampsize = sampsize)

# Create an empty vector to store OOB error values
var_exp <- c()

# Write a loop over the rows of hyper_grid to train the grid of models
for (i in 1:nrow(hyper_grid)) {
  
  # Train a Random Forest model
  model <- randomForest(formula = train$PH ~ ., 
                        data = x,
                        mtry = mtry,
                        nodesize = hyper_grid$nodesize[i],
                        sampsize = sampsize)
  
  # Store OOB error for the model                      
  var_exp[i] <- model$rsq[500]
}

# Identify optimal set of hyperparmeters based on OOB error
opt_i <- which.max(var_exp)
print(hyper_grid[opt_i,]) 

# Results
mtry = 20
nodesize = 4
sampsize = 1796

# Create model off training set
modelRF <- randomForest(formula = PH ~ ., 
                        data = train_set,
                        mtry = mtry,
                        nodesize = nodesize,
                        sampsize = sampsize)
modelRF
summary(modelRF)


plot(modelRF, main = "Error vs Tree Count")

predictRF <- predict(object = modelRF, 
                     newdata = validX,
                     type = "response")

predDF <- data.frame(predRF=predictRF, actual = validY)
predDF$SE.RF <- (predDF$PH - predDF$predRF)^2
RMSE.RF <- sqrt(mean(predDF$SE.RF)) 
RMSE.RF
cor(predDF$PH, predDF$predRF)




# Gradient Boosting Models



set.seed(10)
modelGBM <- gbm(formula = PH~., 
                data = train_set,
                distribution = "gaussian", # hist(data$PH) is approx. normal
                n.trees = 10000)
modelGBM
topFactors <- summary(modelGBM, cBars = 15, plotit=FALSE)[1:15,]

topFactors$var <- factor(topFactors$var, levels= topFactors$var[order(topFactors$rel.inf)])

topFactors$RelativeImportance <- topFactors$rel.inf


p <- ggplot(data=topFactors, aes(x=var, y=RelativeImportance)) + geom_bar(stat = "identity") +theme_minimal() + coord_flip() + ggtitle("GBM: Most Influential Factors For Determining PH") + labs(x= "Relative Importance")
p

# Predict test set values and add to predictions dataframe
predDF$predGBM <- predict(object = modelGBM, 
                          newdata = validX,
                          n.trees = 10000,
                          type = "response")

predDF$SE.GBM <- (predDF$PH - predDF$predGBM)^2
RMSE.GBM <- sqrt(mean(predDF$SE.GBM)) #0.1278

cor(predDF$PH, predDF$predGBM)

# Train a CV GBM model
set.seed(1)
PH_model_cv <- gbm(formula = train$PH ~ ., 
                   distribution = "gaussian", 
                   data = x,
                   n.trees = 92669,
                   cv.folds = 4)


# Optimal ntree estimate based on CV = 92669
ntree_opt_cv <- gbm.perf(object = PH_model_cv, 
                         method = "cv")


# Predict test set values and add to predictions dataframe
predDF$predGBMCV <- predict(object = PH_model_cv, 
                            newdata = test,
                            n.trees = 100000,
                            type = "response")

predDF$SE.GBMCV <- (predDF$actual - predDF$predGBMCV)^2



RMSE.GBMCV <- sqrt(mean(predDF$SE.GBMCV)) # 0.1236


# Table Comparison with Crossfold Validation, complete rows only

df <- read.csv("https://raw.githubusercontent.com/624-Group2/Project-2/master/StudentData.csv")
dfComplete <- df[complete.cases(df),]
# Create folds for 10-fold cross validation
#install.packages("caret")

set.seed(10)
folds <- createFolds(dfComplete$PH, k = 10, list = TRUE, returnTrain = FALSE)
compare <- data.frame(GBM= c(rep(0,10)), RF = c(rep(0,10)))

for (i in 1:10) {
  train <- dfComplete[-folds[[i]], ]
  test <- dfComplete[folds[[i]], ]
  x <- train
  x$PH <- NULL
  
  modelRFtrial <- randomForest(formula = train$PH ~ ., 
                               data = x,
                               mtry = mtry,
                               nodesize = nodesize,
                               sampsize = round(dim(dfComplete)[1]*0.8))
  
  predRFtrial <- predict(object = modelRFtrial, 
                         newdata = test,
                         type = "response")
  
  SE.RF <- (test$PH - predRFtrial)^2
  compare$RF[i] <-  sqrt(mean(SE.RF)) 
  
  modelGBMtrial <- gbm(formula = train$PH~., 
                       data = x,
                       distribution = "gaussian", # hist(data$PH) is approx. normal
                       n.trees = 10000)
  
  predGBMtrial<- predict(object = modelGBMtrial, 
                         newdata = test,
                         n.trees = 10000,
                         type = "response")
  
  SE.GBM <- (test$PH - predGBMtrial)^2
  compare$GBM[i] <- sqrt(mean(SE.GBM)) # 0.124096
}

# Make Final Test Prediction

#Trees

#Impute the data with MissingForest, a Random Forests approach


data <- read.csv("https://raw.githubusercontent.com/624-Group2/Project-2/master/StudentDataImputedMF.csv")

#Imputed Test Set
test_set <- read.csv("https://raw.githubusercontent.com/624-Group2/Project-2/master/PredictImputedMF")

# Results
mtry = 20
nodesize = 4
sampsize = 1796

set.seed(123)
# Create model off training set
modelRF <- randomForest(formula = PH ~ ., 
                        data = data,
                        mtry = mtry,
                        nodesize = nodesize,
                        sampsize = sampsize)
modelRF
summary(modelRF)

predictRFTest <- predict(object = modelRF, 
                         newdata = test_set,
                         type = "response")

test_set$PH <- predictRFTest
#write.csv(test_set,"C:/Users/asher/Documents/GitHub/Project-2/PredictionsRF.csv")


