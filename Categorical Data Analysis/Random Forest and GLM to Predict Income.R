require(randomForest)

adult <- read.table("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",sep=",")
names(adult) <- c("age", "workclass", "fnlwgt", "edu", "edu_num", "marital_status", "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss", "hours_per_week", "native_country", "income")
adult[["edu_num"]] <- NULL
adult[["fnlwgt"]] <- NULL




#EXPLORATORY DATA ANALYSIS------------------
boxplot(adult$age,ylab="Age",main="Boxplot of Age")
with(adult,plot(age,income))

#age and income
hist(adult[adult$income==" <=50K","age"], col=rgb(1,0,0,0.5),xlim=c(15,100), ylim=c(0,4000), main="Age and Income", xlab="Age")
hist(adult[adult$income!=" <=50K","age"], col=rgb(0,0,1,0.5), add=T)
legend(x="topright", c("Income less than 50k","Greater than 50k"), lty=c(1,1), lwd=c(10,10),col=c(rgb(1,0,0,0.5),rgb(0,0,1,0.5)))



#PENALIZED LOGISTIC REGRESSION---------------
require(glmnet)
x <- model.matrix(income~.,data=adult)[,-1]
glmmod <- glmnet(x,y=adult[["income"]],alpha=1,family="binomial")

plot(glmmod,xvar="lambda")
grid()
abline(v=-8,col="red",lwd=3)

cv.glmmod<-cv.glmnet(x,y=as.numeric(adult[["income"]])-1,alpha=1)
plot(cv.glmmod,ylab="Misclassification Rate")
cv.glmmod$lambda.min #cross-validated lambda is almost zero
sum(glmmod[["beta"]][,77]==0) #7 coefficients shrunk to zero

#ORDINARY LOGISTIC REGRESSION---------------

fit.log <- glm(formula = income ~ .,data = adult, family = binomial(link = 'logit'))

fit.log[["deviance"]]/32463


#RANDOM FOREST------------------------------
require(randomForest)
require(caret)
set.seed(123)

bestmtry <- tuneRF(adult[,-13],adult[,13], ntreeTry=100, stepFactor=1.5,improve=0.01, trace=TRUE, plot=TRUE, dobest=FALSE)

fit.rf <- randomForest(income ~ ., data=adult, mtry=2, importance=TRUE, do.trace=100)

varImpPlot(fit.rf,type=1)

#COMPARISON---------------------------------

#misclassification error rates
fit.rf
#confusion matrices
fit.rf[["confusion"]]




#5-fold cross-validation for logistic regression confusion matrix
set.seed(12)
adult_nh<-adult[-which(adult$native_country==" Holand-Netherlands"),]

folds_i <- sample(rep(1:5, length.out = nrow(adult_nh))) #assign observations to folds
for (k in 1:5) {
  test_index <- which(folds_i == k)
  train_fold <- adult_nh[-test_index, ]
  test_fold <- adult_nh[test_index, ]
  
  fit.log <- glm(formula = income ~ .,data = train_fold, family = binomial(link = 'logit'))
  y <- test_fold[["income"]]!=" <=50K"
  yhat<-round(predict(fit.log,newdata=test_fold,type="response"),0)
  assign(paste("confusion",k,sep="_"),table(y,yhat))
}

log.confusion=confusion_1+confusion_2+confusion_3+confusion_4+confusion_5