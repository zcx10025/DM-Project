library(caret)
library(dplyr)
library(e1071)
library(neuralnet)
library(randomForest)
library(corrplot)
library(leaps)
library(ggplot2)

df=read.csv('D:\\chrome download\\Âí¿­ÌØÍø¿Î\\semester 2\\data mining\\Project\\online_shoppers_intention.csv')

df$OperatingSystems=as.factor(df$OperatingSystems) 
df$Browser=as.factor(df$Browser) 
df$Region=as.factor(df$Region) 
df$TrafficType=as.factor(df$TrafficType) 
df$VisitorType=as.factor(df$VisitorType) 
df$Weekend=as.factor(df$Weekend) 
df$Revenue=as.factor(df$Revenue)

standard=preProcess(df,method='range')
df=predict(standard,df)

head(df)
str(df)

corrplot(corr=cor(df[1:10]))

ggplot(df,aes(x=Revenue,y=Administrative))+geom_boxplot()
ggplot(df,aes(x=Revenue,y=Administrative_Duration))+geom_boxplot()
ggplot(df,aes(x=Revenue,y=Informational))+geom_boxplot()
ggplot(df,aes(x=Revenue,y=Informational_Duration))+geom_boxplot()
ggplot(df,aes(x=Revenue,y=ProductRelated))+geom_boxplot()
ggplot(df,aes(x=Revenue,y=ProductRelated_Duration))+geom_boxplot()
ggplot(df,aes(x=Revenue,y=BounceRates))+geom_boxplot()
ggplot(df,aes(x=Revenue,y=ExitRates))+geom_boxplot()
ggplot(df,aes(x=Revenue,y=PageValues))+geom_boxplot()
ggplot(df,aes(x=Revenue,y=SpecialDay))+geom_boxplot()

ggplot(df,aes(x=Revenue,y=Month))+geom_jitter(aes(color=Revenue))
ggplot(df,aes(x=Revenue,y=OperatingSystems))+geom_jitter(aes(color=Revenue))
ggplot(df,aes(x=Revenue,y=Browser))+geom_jitter(aes(color=Revenue))
ggplot(df,aes(x=Revenue,y=Region))+geom_jitter(aes(color=Revenue))
ggplot(df,aes(x=Revenue,y=TrafficType))+geom_jitter(aes(color=Revenue))
ggplot(df,aes(x=Revenue,y=VisitorType))+geom_jitter(aes(color=Revenue))
ggplot(df,aes(x=Revenue,y=Weekend))+geom_jitter(aes(color=Revenue))

dfstep=df
glm_res0=glm(Revenue~1, data=dfstep, family=binomial)
glm_res1=glm(Revenue~., data=dfstep, family=binomial)
sp=step(glm_res1, scope=list(glm_res0, glm_res1), direction="both")

set.seed(1)
intrain=createDataPartition(df$Revenue,p=0.8,list=F)
train=df[intrain,]
test=df[-intrain,]

glm1=train(Revenue~ProductRelated+ProductRelated_Duration+ExitRates+PageValues+Month+TrafficType+VisitorType,data=train,method="glm")
predictions=predict(glm1,newdata=test)
confusionMatrix(predictions,test$Revenue)

glm2=train(Revenue~.,data=train,method="glm")
predictions=predict(glm2,newdata=test)
confusionMatrix(predictions,test$Revenue)

rf1=randomForest(Revenue~ProductRelated+ProductRelated_Duration+ExitRates+PageValues+Month+TrafficType+VisitorType,train)
predictions=predict(rf,newdata=test)
confusionMatrix(predictions,test$Revenue)

rf2=randomForest(Revenue~.,train)
predictions=predict(rf1,newdata=test)
confusionMatrix(predictions,test$Revenue)

knn1=train(Revenue~ProductRelated+ProductRelated_Duration+ExitRates+PageValues+Month+TrafficType+VisitorType,data=train,method="knn")
predictions=predict(knn1,newdata=test)
confusionMatrix(predictions,test$Revenue)

knn2=train(Revenue~.,data=train,method="knn")
predictions=predict(knn2,newdata=test)
confusionMatrix(predictions,test$Revenue)

svm1=svm(Revenue~ProductRelated+ProductRelated_Duration+ExitRates+PageValues+Month+TrafficType+VisitorType,train)
predictions=predict(svm1,newdata=test)
confusionMatrix(predictions,test$Revenue)

svm2=svm(Revenue~.,train)
predictions=predict(svm2,newdata=test)
confusionMatrix(predictions,test$Revenue)

nb1=naiveBayes(Revenue~ProductRelated+ProductRelated_Duration+ExitRates+PageValues+Month+TrafficType+VisitorType,train,laplace=0)
predictions=predict(nb1,newdata=test)
confusionMatrix(predictions,test$Revenue)

nb2=naiveBayes(Revenue~.,train,laplace=0)
predictions=predict(nb2,newdata=test)
confusionMatrix(predictions,test$Revenue)

dummy=dummyVars(~.,data=train)
trainnew=data.frame(predict(dummy,train))
dummy=dummyVars(~.,data=test)
testnew=data.frame(predict(dummy,test))

trainnew$Revenue.FALSE=as.factor(trainnew$Revenue.FALSE)
trainnew$Revenue.TRUE=as.factor(trainnew$Revenue.TRUE)
testnew$Revenue.FALSE=as.factor(testnew$Revenue.FALSE)
testnew$Revenue.TRUE=as.factor(testnew$Revenue.TRUE)

trainnew00=select(trainnew,5,6,8,9,11:20,51:70,71:73,76:77)
testnew00=select(testnew,5,6,8,9,11:20,51:70,71:73,76:77)

nn1=neuralnet(Revenue.FALSE+Revenue.TRUE~.,data=trainnew00,linear.output=F,hidden=3)
predict=neuralnet::compute(nn1,testnew00[1:37])
predicted.class=apply(predict$net.result,1,which.max)-1
predicted.class=as.factor(predicted.class)
confusionMatrix(predicted.class,testnew00$Revenue.FALSE)

nn2=neuralnet(Revenue.FALSE+Revenue.TRUE~.,data=trainnew,linear.output=F,hidden=3)
predict=neuralnet::compute(nn2,testnew[1:75])
predicted.class=apply(predict$net.result,1,which.max)-1
predicted.class=as.factor(predicted.class)
confusionMatrix(predicted.class,testnew$Revenue.FALSE)

nn11=neuralnet(Revenue.FALSE+Revenue.TRUE~.,data=trainnew00,linear.output=F,hidden=5)
predict=neuralnet::compute(nn11,testnew00[1:37])
predicted.class=apply(predict$net.result,1,which.max)-1
predicted.class=as.factor(predicted.class)
confusionMatrix(predicted.class,testnew00$Revenue.FALSE)

nn22=neuralnet(Revenue.FALSE+Revenue.TRUE~.,data=trainnew,linear.output=F,hidden=5)
predict=neuralnet::compute(nn22,testnew[1:75])
predicted.class=apply(predict$net.result,1,which.max)-1
predicted.class=as.factor(predicted.class)
confusionMatrix(predicted.class,testnew$Revenue.FALSE)

nn111=neuralnet(Revenue.FALSE+Revenue.TRUE~.,data=trainnew00,hidden=c(3,5))
predict=neuralnet::compute(nn111,testnew00[1:37])
predicted.class=apply(predict$net.result,1,which.max)-1
predicted.class=as.factor(predicted.class)
confusionMatrix(predicted.class,testnew00$Revenue.FALSE)

nn222=neuralnet(Revenue.FALSE+Revenue.TRUE~.,data=trainnew,linear.output=F,hidden=c(3,5))
predict=neuralnet::compute(nn222,testnew[1:75])
predicted.class=apply(predict$net.result,1,which.max)-1
predicted.class=as.factor(predicted.class)
confusionMatrix(predicted.class,testnew$Revenue.FALSE)