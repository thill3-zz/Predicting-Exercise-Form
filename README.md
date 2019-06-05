# Predicting-Exercise-Form
Random Forest Model to determine correct or incorrect form of a unilateral dumbbell bicep curl.

In general, in the past it has been common to track how much of an exercise someone is doing. This data collection has not, however, addressed whether an exercise is done correctly. The dataset in this paper was collected from accelerometers on the belt, forearm, arm, and dumbell of 6 participants (males aged 20-28), and it relates specifically to how an exercise is performed. What kind of action the participant took (either correct or one of 4 incorrect ways) is categorized as one of 5 letters. ‘A’ corresponds to performing the dumbbell curl correctly. ‘B’, ‘C’, ‘D’, and ‘E’ are all variations that perform the exercise incorrectly. The ultimate goal is to use the data to determine in which manner the exercise is performed.
Usefully the random forest model in the caret package in R provides an oob (out-of-bag or out-of-sample) error estimate while predicting the “classe” variable on the dataset. We see that the random forest produces a model with over 99.8% prediction accuracy on an separate data sample (this is estimated out-of-sample error). This lends confidence to its predictions on the assigned testing set that has no given “classe” values


Training file is located at https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv
Testing file is located at https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv
Dataset citation: Ugulino, W.; Cardador, D.; Vega, K.; Velloso, E.; Milidiu, R.; Fuks, H. Wearable Computing: Accelerometers’ Data Classification of Body Postures and Movements. Proceedings of 21st Brazilian Symposium on Artificial Intelligence. Advances in Artificial Intelligence - SBIA 2012. In: Lecture Notes in Computer Science. , pp. 52-61. Curitiba, PR: Springer Berlin / Heidelberg, 2012. ISBN 978-3-642-34458-9. DOI: 10.1007/978-3-642-34459-6_6.
