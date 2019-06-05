#Load Data
if(!dir.exists("G:/My Drive/Coursera Data Science Specialization/Practical Machine Learning/Fitness Data")) {
        setwd("G:/My Drive/Coursera Data Science Specialization/Practical Machine Learning")
        dir.create(path = "G:/My Drive/Coursera Data Science Specialization/Practical Machine Learning/Fitness Data")
}
setwd("G:/My Drive/Coursera Data Science Specialization/Practical Machine Learning/Fitness Data")
if(!file.exists("pml-training.csv")) {
        fileURL <-
                "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
        download.file(url = fileURL
                      , destfile = "pml-training.csv"
                      , mode = "w"
        )
} else {
        print("pml-training.csv is already available")
        
}
if(!file.exists("pml-testing.csv")) {
        fileURL <-
                "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
        download.file(url = fileURL
                      , destfile = "pml-testing.csv"
                      , mode = "w"
        )
} else {
        print("pml-testing.csv is already available")
        
}

pmlTrainFile <- "pml-training.csv"
file.info(pmlTrainFile)$size #12,222,368 bytes
pmlTestFile <- "pml-testing.csv"
file.info(pmlTestFile)$size #15,134 bytes

a <- Sys.time(); pmlTrain <- read.csv(file = pmlTrainFile, as.is = TRUE); b <- Sys.time(); print(paste("Reading the file from", a, "to", b, ".")) 
#between 1 and 3 seconds
dim(pmlTrain) #19622x160
a <- Sys.time(); pmlProject <- read.csv(file = pmlTestFile, as.is = TRUE); b <- Sys.time(); print(paste("Reading the file from", a, "to", b, ".")) 
#less than a second
dim(pmlProject) #20x160
#160 features is a lot.

#Clean dataset
library(caret)
nzvDF <- nearZeroVar(pmlTrain, saveMetrics = TRUE) #show a display of the variables and
        #the determination whether they are zero variance or near zero variance
        #if column zeroVar is TRUE then the predictor has one distinct value.
        #if column nzv is true then the predictor has near zero variance
table(nzvDF[,3]) #All false. Nothing has exactly zero variance
table(nzvDF[,4]) #60 have near zero variance, though.
nzvDontUse <- nearZeroVar(pmlTrain, saveMetrics = FALSE)
#new window is worth a look
table(pmlTrain$new_window, pmlTrain$classe)
#Doesn't appear to be any correlation between "new window" and "classe"
pmlTrain1 <- pmlTrain[,-nzvDontUse]
pmlProject1 <- pmlProject[,-nzvDontUse]
dim(pmlTrain1) #19622x100
dim(pmlProject1) #20x100
#That's 60 variables that we don't need to consider. 100 is still a lot.
#and, unfortunately, they are all still "character" rather than numeric.
#Are any of the features more than half NAs?
fracNAs <- apply(pmlTrain1, MARGIN = 2, function(x) {mean(is.na(x))})
table(fracNAs)
# 0     0.979308938946081 
# 59    41
#Let's remove those 41 that are mostly NAs
mostlyNAs <- which(fracNAs > .5)
mostlyNAs
pmlTrain2 <- (pmlTrain1[,-mostlyNAs])[,-c(1:5)] #and we remove the index along 
        #with the timestamp features
dim(pmlTrain2) #19622x54 #We've eliminated 106 features (159 to 53)
fracNAs1 <- apply(pmlTrain2, MARGIN = 2, function(x) {mean(is.na(x))})
table(fracNAs1)
#There are no NAs in the data anymore.
#Keep the data to predict up to date with the transformations.
pmlProject2 <- (pmlProject1[,-mostlyNAs])[,-c(1:5)]
dim(pmlProject2) #20x54
fracNAsProj <- apply(pmlProject2, MARGIN = 2, function(x) {mean(is.na(x))})
table(fracNAsProj)


#Now split into testing and training
#set.seed(20190531)
library(caret)
#use 2/3 of the rows to train the model. Then test the model and estimate 
        #out-of-sample error on the remaining 1/3 of the data
# pmlInTrain <- createDataPartition(y = pmlTrain2$classe, p = 2/3, list = FALSE)
# pmlTrain2[,54] <- sapply(pmlTrain2[,54],as.factor) #set the classe variable to a factor
# pmlTrain3 <- pmlTrain2[pmlInTrain,]
# pmlTest3 <- pmlTrain2[-pmlInTrain,]
# dim(pmlTrain3) #13083x54
# dim(pmlTest3)  #6539x54

#Set up parallel processing
library(parallel)
library(doParallel)
cluster <- makeCluster(detectCores() - 1) # convention to leave 1 core for OS
registerDoParallel(cluster)
#Then Set up trainControl for training the random forest
pmlRFControl <- trainControl(allowParallel = TRUE) #use the cluster from the previous section
        #Note that I am intentionally leaving the validation method to the default of "boot"
        #(meaning standard bootstrap) due to this note by the creator of random forest.
        #https://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm#ooberr
        #"In random forests, there is no need for cross-validation or a separate
                #test set to get an unbiased estimate of the test set error. It 
                #is estimated internally, during the run, as follows: ..."
a <- Sys.time(); pmlRF3 <- train(classe ~.,method="rf",data=pmlTrain2, trControl = pmlRFControl); b <- Sys.time(); print(paste("Training Random Forest model", a, "to", b, ".")) 
#If I use cross validation then it takes 54 minutes on a single processing thread and  11:58 with 7 processing threads
#Using the default bootstrapping with 7 threads takes 68 minutes.
stopCluster(cluster)
registerDoSEQ()
save(pmlRF3, file = "Practical Machine Learning Random Forest Algorithm.rda")
load(file = "Practical Machine Learning Random Forest Algorithm.rda")
#How well does the model do?
(pmlRF3$finalModel)$confusion
#Class prediction error is less than 4e-3 for all 5 classes
#Over all out-of-bag estimate is .0014 (1.4e-3)

plot(pmlRF3, main = 'Accuracy by Predictor Count')
#This plot confirms that R got the best forest when it used 27 predictors
library(randomForest)
varImpPlot(pmlRF3$finalModel, main = 'Variable Importance Plot: Random Forest')
#Topmost effective predictor was num_window.
        #then roll_belt, pitch_forearm, yaw_belt, magnet_dumbbell_z, pitch_belt, magnet_dumbell_y, and roll_forearm.
        #The rest had progressively less effect on the Gini value.

#Find predictions for the unknown samples
predict(pmlRF3, pmlProject2)
#[1] B A B A A E D B A A B C B A E E A B B B

#There are other classification methods of course.
        #Linear Discriminant Analysis and Support Vector Machines are a couple.
        #However, given the accuracy of this model I am disinclined to pursue
                #other classification methods.