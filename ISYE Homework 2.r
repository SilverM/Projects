
dataset <- read.csv('/Users/soonleqi/Desktop/ISYE6501/Week2/credit_card_data-headers.txt',sep = '\t')

head(dataset)

library(kknn)
library(e1071)
library(kernlab)
library(caret)

# Normalize data 
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x))) }

prc_n <- as.data.frame(lapply(dataset[,1:10], normalize))

## 75% of the sample size
smp_size <- floor(1 * nrow(prc_n))

## set the seed to make your partition reproducible
set.seed(123)
train_ind <- sample(seq_len(nrow(prc_n)), size = smp_size)

prc_train <- prc_n[train_ind, ]
prc_test <- prc_n[-train_ind, ]

prc_train_labels <- dataset[train_ind, 11]
prc_test_labels <- dataset[-train_ind, 11]

anyNA(dataset)

trctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 3)

set.seed(200)
prc_train_labels <- as.factor(prc_train_labels)
knn_fit <- train(prc_train, prc_train_labels, method = "knn",
                 trControl=trctrl,
                 preProcess = c("center", "scale"),
                 tuneLength = 20)

x = list(knn_fit$results$Accuracy)
y = list(c(5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43)) 
test1 <- list(c(k = y, Accuracy = x))
as.data.frame(test1)

y = (knn_fit$results$Accuracy)
x = (c(5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43))

y_range = range(knn_fit$results$Accuracy)
x_range = range(c(5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43) )

plot(x_range, y_range, type="n" ) 
lines(x, y) 
cat("Best tune is:", "k =" ,as.character(knn_fit$bestTune))

head(prc_n)
