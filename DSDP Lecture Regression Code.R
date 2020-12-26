#Load library 
#REMINDER - packages must be installed first
library(xlsx)
library(ggplot2)
library(rcompanion)
library(ggpubr)
library(car) 
library(Metrics)

#Create vector with File Path information
RegressionPath <-"C:/Users/nyzw/OneDrive - Chevron/DSDP/DSDP 2019/Lectures/Week 4 Regression/GraduateEarnings.xlsx"

#Import in file
GraduateEarn <- read.xlsx(RegressionPath,1)

#Look at data - visually look!
View(GraduateEarn)

#Look at the dataset dimensions
dim(GraduateEarn)

#Simple linear regression (1 DV, 1 IV - both continuous)
EarnCostModel <- lm(Earn ~ Price, data=GraduateEarn)
summary(EarnCostModel)

#Plot the regression model
ggplot(GraduateEarn, aes(x=Price, y=Earn)) +
    xlab("Price") +
    ylab("Earnings") +
    geom_point(size = 2, colour = "blue", alpha=0.3) +
    geom_smooth(method = "lm", colour = "red")

#Calculate Predicted Values on current dataset
GraduateEarn$Y_EarnCostModel <- predict(EarnCostModel)
View(GraduateEarn)

#Assessing prediction accuracy
#Create Training and Test data 
set.seed(123)  # setting seed to reproduce results of random sampling
trainingRowIndex <- sample(1:nrow(GraduateEarn), 0.8*nrow(GraduateEarn)) #row indices for training data
trainingData <- GraduateEarn[trainingRowIndex, ]  #model training data
testData  <- GraduateEarn[-trainingRowIndex, ]   #test data

#Build the model on training data
TrainModel <- lm(Earn ~ Price, data=trainingData) 
trainingDataEarnPred = predict(TrainModel, trainingData)

#Predict earnings on test data using model from training data
testDataEarnPred <- predict(TrainModel, testData)  

#compare RMSE between train and test datasets
rmse(trainingData$Earn, trainingDataEarnPred)
rmse(testData$Earn, testDataEarnPred)

#look at correlation between actual and predicited in test data
actuals_preds <- data.frame(cbind(actuals=testData$Earn, predicteds=testData$EarnPred)) 
correlation_accuracy <- cor(actuals_preds) 
View(correlation_accuracy)
plot(actuals_preds$actuals, actuals_preds$predicteds)

#Diagnostics for our simple linear regression (Full Model)
par(mfrow = c(2,2))
EarnCostModel <- lm(Earn ~ Price, data=GraduateEarn, plot(EarnCostModel))

#Statistical Assumptions
#Linearity (DV and IV)
par(mfrow = c(1,1)) #return plot frame to 1:1
plot(GraduateEarn$Price, GraduateEarn$Earn)

#Normality
#For the IV
shapiro.test(GraduateEarn$Price)
plotNormalHistogram(GraduateEarn$Price, breaks=30, col="grey", linecol="red", xlab="Price")
ggqqplot(GraduateEarn$Price)
#For the Residuals
GraduateEarn$resids <- resid(lm(Earn ~ Price, data=GraduateEarn))
shapiro.test(GraduateEarn$resids)
ggqqplot(GraduateEarn$resids)

#Multicollinearity
#Grab the numerical data for the correlation matrix
CorrData <- GraduateEarn[,c(4,5,6,7)]
CorrMatrixAll <- cor(CorrData)
View(CorrMatrixAll)
vif(lm(Earn ~ Price + SAT + ACT, data = GraduateEarn))

#Autocorrelation
durbinWatsonTest(EarnCostModel)

#Multivariate regression
MultiModel <- lm(Earn ~ Price + SAT + Public, data=GraduateEarn)
summary(MultiModel)

#Model Selection
#Comparison with ANOVA
BaseModel <- lm(Earn ~ Price + SAT + Public, data=GraduateEarn)
Model1 <- lm(Earn ~ Price + SAT, data=GraduateEarn)
Model2 <- lm(Earn ~ Price, data=GraduateEarn)
anova(Model2, Model1)
anova(Model1, BaseModel)

#Stepwise Regression
MultiModel <- lm(Earn ~ Price + SAT + Public, data=GraduateEarn)
SelectModel <-step(MultiModel)
summary(SelectModel)

