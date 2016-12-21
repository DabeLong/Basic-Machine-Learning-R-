file = "/~/diabetes_data.csv"
diabetesData = read.csv(file, header = TRUE)

library("data.table")

summary(diabetesData)
str(diabetesData)

table(diabetesData$Class)

# Partition data into training and testing sets into 20% / 80%
library(caret)
set.seed(500)
training = createDataPartition(diabetesData$Class, p = 0.20, list = FALSE)
dataTrain = diabetesData[training,]
dataTest = diabetesData[-training,]

# PART A, BUILDING
# Simple Naive Bayes train
prop.table(table(dataTrain$Class))
probDiabetes = prop.table(table(dataTrain$Class))[["1"]][1]

meanTrain = aggregate(dataTrain[, 1:8], list(dataTrain$Class), mean)
sdTrain = aggregate(dataTrain[, 1:8], list(dataTrain$Class), sd)

meanTrain[2:9][1,]
sdTrain[2:9][1,]

classify = function(sample){
  posterior <- function (sample, class_prior, mean, sd, class) {
    p_pregnant <- dnorm(sample$Times.Pregnant, mean[class,2], sd[class,2])
    p_glucose <- dnorm(sample$Plasma.Glucose.Concentration, mean[class,3], sd[class,3])
    p_blood_pressure <- dnorm(sample$Blood.Pressure, mean[class,4], sd[class,4])
    p_triceps <- dnorm(sample$Triceps.Skin.Thickness, mean[class,5], sd[class,5])
    p_insulin <- dnorm(sample$Serum.Insulin, mean[class,6], sd[class,6])
    p_bmi <- dnorm(sample$BMI, mean[class,7], sd[class,7])
    p_diabetes <- dnorm(sample$Diabetes.Pedigree.Function, mean[class,8], sd[class,8])
    p_age <- dnorm(sample$Age, mean[class,9], sd[class,9])
    
    # returns probability
    return(log(class_prior) + log(p_pregnant) + log(p_glucose) + log(p_blood_pressure) + log(p_triceps)
           + log(p_insulin) + log(p_bmi) + log(p_diabetes) + log(p_age))
  }
  
  meanTrain[2:8][1,]
  sdTrain[2:8][1,]
  
  prior_diabetes = probDiabetes
  prior_no_diabetes = 1 - probDiabetes
  
  return(list(diabetes = posterior(sample, prior_diabetes, meanTrain, sdTrain, class = 2),
              no_diabetes = posterior(sample, prior_no_diabetes, meanTrain, sdTrain, class = 1)))
}
result = classify(dataTest)
cat('posterior(diabetes) =', result$diabetes)
cat('posterior(no_diabetes) =', result$no_diabetes)

classification = (result$diabetes > result$no_diabetes)
actual = (dataTest$Class == 1)
correctness = (classification == actual) 
prop.table(table(correctness))
#FALSE      TRUE 
#0.2589577 0.7410423 



# PART B
# remove data with 0 as a value for attributes 3, 4, 6, and 8
diabetesDataFiltered = subset(diabetesData, (diabetesData$Blood.Pressure != 0 & 
        diabetesData$Triceps.Skin.Thickness != 0 & diabetesData$BMI != 0 & diabetesData$Age != 0) )
#partition data
set.seed(500)
training = createDataPartition(diabetesDataFiltered$Class, p = 0.20, list = FALSE)
dataTrain = diabetesDataFiltered[training,]
dataTest = diabetesDataFiltered[-training,]
#compute probability
probDiabetes = prop.table(table(dataTrain$Class))[["1"]][1]
#compute means and standard deviation
meanTrain = aggregate(dataTrain[, 1:8], list(dataTrain$Class), mean)
sdTrain = aggregate(dataTrain[, 1:8], list(dataTrain$Class), sd)
# compute classification w/ normal distribution for priors
result = classify(dataTest)
cat('posterior(diabetes) =', result$diabetes)
cat('posterior(no_diabetes) =', result$no_diabetes)
classification = (result$diabetes > result$no_diabetes)
actual = (dataTest$Class == 1)
correctness = (classification == actual) 
prop.table(table(correctness))
#FALSE      TRUE 
#0.2494172 0.7505828



# Part C
# use klaR library
library(klaR)
library(e1071)
# NaiveBayes
set.seed(1234)
training = createDataPartition(diabetesData$Class, p = 0.20, list = FALSE)
dataTrain = diabetesData[training,]
dataTest = diabetesData[-training,]
diabetesData$Class = factor(diabetesData$Class)

fit = NaiveBayes(dataTrain$Class ~ . , data = dataTrain)
pred = predict(fit, dataTest)
pred$class

fit = NaiveBayes(diabetesData, diabetesData$Class, data=dataTrain)
pred = predict(fit, dataTest)
head(predict(fit))
dataTest$Class
temp = pred$class==dataTest$Class
prop.table(table(temp))
#FALSE     TRUE 
#0.252443 0.747557




# d
# SVMLight
x = svmlight(diabetesData$Class ~ ., data = diabetesData)
predict(x, diabetesData)

install.packages('klaR')
library('klaR')
data(iris)
x <- svmlight(Species ~ ., data = iris)
