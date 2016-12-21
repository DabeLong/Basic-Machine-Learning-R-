# The goal of SVM is to:
# minimize |w|^2 + C summation episilon, 
# WHERE:
# C is regularizing constant
# w is weight vector (normal)
# episilon is margin violation

# An optimized version:
# minimize |w|^2 + C summation max(0, 1-yi f(xi))

# f(xi) = (w-transpose * xi + b)

# We will use gradient descent:
# minimize cost function C(w) using iterative update Wt+1 = Wt - nt gradient C(Wt) where n is learning rate
# min C(w) = 1/N summation (lambda/2 |w|^2 + max(0, 1-yi f(xi)) )

# iterative update
# Wt+1 = Wt -n * gradient C(wt)

# THEREFORE:
# we have Wt+1 = Wt - n(lambda Wt - yixi) if (yi f(xi) < 1)
# otherwise, Wt+1 = Wt - nlambda Wt
# WHERE:
# yi is +1 or -1 depending on classifier
# xi is the vector of the data point
# Wt is the W computed after t iterations of gradient descent
# n is the learning rate. 1 way to choose is to set nt = 1/(lambda*t)
# lambda is 2/NC, with N being number of data points and C being the regularizing constant

# With stochastic gradient descent, we just take a random sample from the training data to compute the gradient

library(klaR)
library(caret)
library(stringr)

file = "~/breast_cancer_data.csv"
breastCancerData = read.csv(file, header = TRUE)

# remove id
breastCancerData = breastCancerData[,-c(1)]
data = breastCancerData[,-c(1)]
classifier = as.factor(breastCancerData[ ,1])

# normalize data
data = scale(data)

#convert classifier into 1 or -1 (will be our y values)
classifier.temp = rep(0,length(labels))
classifier.temp[classifier=="M"] = -1
classifier.temp[classifier=="B"] = 1
classifier = classifier.temp
rm(classifier.temp)

# Separate the resulting dataset randomly
# 369 out of 569 should be training
set.seed(1234)
partition = createDataPartition(y = classifier, p = 369/569, list = FALSE)
trainingData = data[partition,]
trainingClassifier = classifier[partition]

# rest of 200 data points
remainingClassifier = classifier[-partition]
remainingData = data[-partition,]
# 100 testing
test = createDataPartition(y = remainingClassifier, p = 0.5, list = FALSE)
testingClassifier = remainingClassifier[test]
testingData = remainingData[test,]
# 100 validation
validationClassifier = remainingClassifier[-test]
validationData = remainingData[-test,]

# classifier steps / epochs
numEpochs = 50;
numStepsPerEpoch = 100;
numStepsPerPlot = 10;
epochValidationSetSize = 50;

# lambda = 2/NC
lambda = c(0.001, 0.01, 0.1, 1);
N = 369 # size of set
C = 2/(N*lambda) # regularizing constant
C = signif(C,4)

# define accuracy function
getAccuracy <- function(w,b,data,classifier){
  estimate = data %*% w + b;
  predicted = rep(0,length(classifier));
  predicted [estimate < 0] = -1 ;
  predicted [estimate >= 0] = 1 ;
  return (sum(predicted == classifier) / length(classifier))
}

# initialize accuracy matrices for each epoch of all lambda values
accuracyMatrix = matrix(NA, 1 + numEpochs * ( numStepsPerEpoch / numStepsPerPlot) , length(lambda))
validationAccuracyMatrix = matrix(NA, 1 + numEpochs * ( numStepsPerEpoch / numStepsPerPlot) , length(lambda))
bestAccuracy = 0
best_w = rep(0, ncol(data));
best_b = 0
best_lambda = 1

for (i in 1:4){
  l = lambda[i] # lambda value
  row = 1
  col = i;
  stepIndex = 0;
  
  # initialize w and b as 0, the SVM we are trying to find
  w = rep(0, ncol(data));
  b = 0;
  
  for (epoch in 1:numEpochs){
    # for each epoch, partition data into training and validation
    # we are splitting 369 into 319 and 50
    epochPartition = createDataPartition(y=trainingClassifier, p=(1 - epochValidationSetSize/length(trainingClassifier)), list=FALSE)
    epochTrainingData = trainingData[epochPartition,]
    epochTrainingClassifier = trainingClassifier[epochPartition]
    epochValidationData = trainingData[-epochPartition,]
    epochValidationClassifier = trainingClassifier[-epochPartition]
    
    # learning rate
    n = 1/ (0.01*epoch + 50)
    
    for (step in 1:numStepsPerEpoch){
      stepIndex = stepIndex + 1
      # select index randomly
      index = sample.int(nrow(epochTrainingData),1);
      
      xi = epochTrainingData[index, ]
      yi = epochTrainingClassifier[index]
      value = yi * (t(w) %*%xi + b) #yk (w * xk + b) 
      
      if (value >= 1) {
        w = w - n*l*w
        b = b
      } else {
        w = w - n * (l*w - yi*xi)
        b = b + n*yi
      }
      
      # Update using:
      # Wt+1 = Wt - n(lambda Wt - yixi) if (yi f(xi) < 1)
      # Wt+1 = Wt - nlambda Wt Otherwise
      
      #fxi = t(w) %*% xi + b # f(xi) = (w-transpose * xi + b)
      # if (yi*fxi < 1){
      #   w = w - n * (l*w - yi * xi)
      #   b = b
      # } else {
      #   w = w - n*l*w
      #   b = b + n * (yi)
      # }
      
      # C(w) = 1/N summation (lambda/2 |w|^2 + max(0, 1-yi f(xi)) )
      # cost = 0
      # for (z in 1:length(epochTrainingClassifier)){
      #   temp1 = l / 2 * sum(abs(w)^2) # lambda/2 |w|^2
      #   fxi = t(w) %*% epochTrainingData[z, ] + b # f(xi) = (w-transpose * xi + b)
      #   temp2 = max(0 , 1 - epochTrainingClassifier[z] * fxi) # max(0, 1-yi f(xi))
      #   tempCost = temp1 + temp2 # lambda/2 |w|^2 + max(0, 1-yi f(xi))
      #   cost = cost + (tempCost)
      # }
      # cost = cost/N
      
      # save accuracy in matrix
      if (stepIndex %% numStepsPerPlot == 1){
        accuracyMatrix[row,col] = getAccuracy(w,b,epochValidationData,epochValidationClassifier);
        validationAccuracyMatrix[row,col] = getAccuracy(w,b,validationData,validationClassifier);
        row = row + 1;
      }
    }
  }
  
  # check accuracy after every epoch
  tempAccuracy = getAccuracy(w,b,validationData,validationClassifier)
  print(str_c("Temp Accuracy = ", tempAccuracy," and Best Accuracy = ", bestAccuracy) )
  # improve accuracy if better
  if(tempAccuracy > bestAccuracy){
    bestAccuracy = tempAccuracy
    best_w = w;
    best_b = b;
    best_lambda = l;
  }
}

getAccuracy(best_w,best_b, testingData, testingClassifier)

#Plot accuracy
colors = c("red","blue","green","black");
xLabel = "Number of Steps";
yLabel = "Accuracy on Randomized Epoch Validation Set"
#yLabel = "Accuracy on Validation Set"
title="Accuracy as a Function of Step and Regularizing Constant";

totalSteps = numStepsPerEpoch*numEpochs
plotSteps = numStepsPerPlot*numEpochs
stepValues = seq(1,totalSteps,length=plotSteps)

mat = accuracyMatrix
#mat = validationAccuracyMatrix

for(i in 1:2){
  # create graph once first
  if(i == 1){
    plot(stepValues, mat[1:plotSteps,i], type = "l",xlim=c(0, totalSteps), ylim=c(0,1),
         col=colors[i],xlab=xLabel,ylab=yLabel,main=title)
  } else{
    lines(stepValues, mat[1:plotSteps,i], type = "l",xlim=c(0, totalSteps), ylim=c(0,1),
          col=colors[i],xlab=xLabel,ylab=yLabel,main=title)
  }
}

legend(x=3000,y=.5,
       legend=c( paste("C =",C[1]), paste("C =",C[2]), paste("C =",C[3]), paste("C =",C[4]) ),
       fill=colors,
       text.width = 800)



plot(stepValues, mat[1:plotSteps,4], type = "l",xlim=c(0, totalSteps), ylim=c(0,1),
     col=colors[4],xlab=xLabel,ylab=yLabel,main=title)
legend(x=3000,y=.5,
       legend=c( paste("C =",C[4])),
       fill=colors[4],
       text.width = 800)
