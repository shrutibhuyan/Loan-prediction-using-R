path="/Users/abhilekh/Downloads/"
setwd(path)
install.packages("caret", dependencies = c("Depends","Suggests"))
install.packages("RANN")
library(caret)
library(RANN)
train<-read.csv("train_u6lujuX_CVtuZ9i.csv",stringsAsFactors = T)
test<-read.csv("test_Y3wMUE5_7gLdaTN.csv")
str(train)

#1. Data pre processing
#check for missing values
sum(is.na(train)) #86 missing values(i.e. NA's found)

#use knn to impute these missing values
#scale and centre the numerical columns using preProcess() in caret

preprocValues<-preProcess(train,method = c("knnImpute","center","scale"))
train_processed<-predict(preprocValues,train)
sum(is.na(train_processed))
#convert loan status to numeric
train_processed$Loan_Status<-ifelse(train_processed$Loan_Status=="N",0,1)
train_processed$Loan_ID=as.numeric(train_processed$Loan_ID)

#create dummy vars for categorical vars using one hot encoding
dmy<-dummyVars("~.", data=train_processed,fullRank = T)
train_transformed<-data.frame(predict(dmy,newdata=train_processed))
str(train_transformed)
train_transformed$Loan_Status=as.factor(train_transformed$Loan_Status)
#splitting train set into two parts for cross validation 75% and 25%
index<-createDataPartition(train_transformed$Loan_Status,p=0.75,list = FALSE)
trainSet<-train_transformed[index,]
testSet<-train_transformed[-index,]

#Features selection using Recursive feature elimination in caret
control<-rfeControl(functions = rfFuncs, method="repeatedcv", repeats = 3,verbose = FALSE)
predictors<-setdiff(names(trainSet),"Loan_Status")
Loan_Pred_Profile<-rfe(trainSet[,predictors], trainSet[,"Loan_Status"],rfeControl = control)

names(getModelInfo())#gives list of all ML algo's in caret package

#using caret package any model can be applied by only changing the method in the train function
model_rf<-train(trainSet[,predictors],trainSet[,"Loan_Status"],method='rf') #random forest
model_nnet<-train(trainSet[,predictors],trainSet[,"Loan_Status"],method='nnet') #neuralnet
model_glm<-train(trainSet[,predictors],trainSet[,"Loan_Status"],method='glm') #generalised linear model

fitControl <- trainControl(
  method = "repeatedcv",
  number = 5,
  repeats = 5)

modelLookup(model = "gbm")

#Creating grid for grid search
grid<-expand.grid(n.trees=c(10,20,50,100,500,1000),shrinkage=c(0.01,0.05,0.1,0.5),
                  n.minobsinnode = c(3,5,10),interaction.depth=c(1,5,10))

#training the model
model_gbm<-train(trainSet[,predictors], trainSet[,"Loan_Status"],method="gbm",trControl=fitControl,tuneGrid=grid)
print(model_gbm)
plot(model_gbm)
#using tune length to train the model
#in tune length instead of explicitly giving the parameters in grid search, 
#any number of possible values for each tuning parameter through tuneLength can be given

model_gbm<-train(trainSet[,predictors],trainSet[,"Loan_Status"],method="gbm",trControl=fitControl,tuneLength=10)
print(model_gbm)
plot(model_gbm)

#variable importance using caret functions
varImp(model_gbm)
plot(varImp(model_gbm),main="GBM-Variable Importance")

varImp(model_rf)
plot(varImp(model_rf),main="Random forest-Variable Importance")

varImp(model_nnet)
plot(varImp(model_nnet),main="Neural Net-Variable Importance")

varImp(object=model_glm)
plot(varImp(object=model_glm),main="Generalized Linear Model-Variable Importance")

#Predictions
predictions<-predict.train(object = model_gbm,testSet[,predictors],type="raw")
table(predictions)
confusionMatrix(predictions,testSet[,"Loan_Status"])
