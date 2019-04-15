library(caret)
library(dplyr)
library(e1071)
library(neuralnet)
library(randomForest)
library(corrplot)
library(leaps)
df=read.csv('C:\\Users\\13541\\Desktop\\online_shoppers_intention.csv')

corrplot(corr=cor(df[1:10]))

df$Month=as.factor(df$Month)
df$OperatingSystems=as.factor(df$OperatingSystems) 
df$Browser=as.factor(df$Browser) 
df$Region=as.factor(df$Region) 
df$TrafficType=as.factor(df$TrafficType) 
df$VisitorType=as.factor(df$VisitorType) 
df$Weekend=as.factor(df$Weekend) 
df$Revenue=as.factor(df$Revenue)

standard=preProcess(df,method='range')
df=predict(standard,df)

set.seed(1)
intrain=createDataPartition(df$Revenue,p=0.6,list=F)
train=df[intrain,]
test=df[-intrain,]

glmback=step(glm(Revenue~.,data=train,family="binomial"),direction="backward")

glm1=train(Revenue~ProductRelated_Duration+ExitRates+PageValues+Month+TrafficType+VisitorType,data=train,method="glm")
predictions=predict(glm1,newdata=test)
confusionMatrix(predictions,test$Revenue)

glm2=train(Revenue~.,data=train,method="glm")
predictions=predict(glm2,newdata=test)
confusionMatrix(predictions,test$Revenue)

rf=randomForest(Revenue~.,train)
predictions=predict(rf,newdata=test)
confusionMatrix(predictions,test$Revenue)

rf1=randomForest(Revenue~ProductRelated_Duration+ExitRates+PageValues+Month+TrafficType+VisitorType,train)
predictions=predict(rf1,newdata=test)
confusionMatrix(predictions,test$Revenue)


