install.packages("glmnet")
install.packages("caret")
install.packages("mlbench")
install.packages("psych")


library(glmnet)
library(caret)
library(mlbench)
library(psych)


?BostonHousing
data("BostonHousing")
mydata <- BostonHousing


nrow(mydata)
ncol(mydata)
str(mydata)


pairs.panels(mydata[c(-4,-14)],cex = 1)


## Data Division


set.seed(222)            ## to get rpeatable results
independent <- sample(2,nrow(mydata),replace = TRUE,prob = c(0.7,0.3)) # sampling with replacement
train <- mydata[independent == 1,]
test <- mydata[independent == 2,]


## Cross Validation


custom <- trainControl(method = "repeatedcv",
                       number = 10,
                       repeats = 5, verboseIter = T)


## Linear Model


set.seed(1234)
lm <- train(medv ~. ,
            train,
            method = 'lm',trControl = custom)

lm
lm$results
summary(lm)
plot(lm$finalModel)



## Ridge


set.seed(1234)
RIDGE <- train(medv~.,train,
               method = 'glmnet',
               tuneGrid = expand.grid(alpha=0,
                                      lambda = seq(0.0001,1,length = 5)),
               trControl = custom)

RIDGE
plot(RIDGE)
RIDGE$modelInfo
RIDGE$method
RIDGE$modelType
RIDGE$dots
?plot
plot(RIDGE$finalModel,xvar = "lambda", label = TRUE)
plot(RIDGE$finalModel,xvar = "dev")
plot(varImp(RIDGE, scale=T))



## LASSO


set.seed(1234)
lasso <- train(medv~.,train,
               method = 'glmnet',
               tuneGrid = expand.grid(alpha=1,
                                      lambda = seq(0.0001,0.2,length = 5)),
               trControl = custom)
lasso
lasso$results
plot(lasso)
plot(lasso$finalModel,xvar = "lambda",label = T)
plot(lasso$finalModel,xvar = "dev")
plot(varImp(lasso,scale = 'F'))


## Elastic Net


set.seed(1234)
elasticnet <- train(medv~.,train,
               method = 'glmnet',
               tuneGrid = expand.grid(alpha=seq(0,1,length=10),
                                      lambda = seq(0.0001,0.2,length = 5)),
               trControl = custom)



plot(elasticnet,xvar='lambda',label = T)
plot(elasticnet$finalModel,xvar = 'lambda',label = TRUE)


# Comparing the models


models <- list(linear_model = lm, Ridge = RIDGE, Lasso = lasso,Elastic = elasticnet)
resamples <- resamples(models)
summary(resamples)


##box and whiskers plot
bwplot(resamples,col = 'red')
?xyplot
xyplot(resamples,metric = 'RMSE',col = 'blue')


lasso$bestTune
bestmodel <- lasso$bestTune
coef(bestmodel, s = lasso$bestTune$alpha)


## Predicition 


predection <- predict(lasso,train)
sqrt(mean(train$medv-predection)^2)

predection1 <- predict(lasso,test)
sqrt(mean(test$medv-predection1)^2)