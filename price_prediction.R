library(glm2)
library(randomForest)
library(e1071)

train_size <- dim(data)[1] * 0.80
model_ols <- lm(label ~ open + high + low + close, data=data[0:train_size,])
predictions <- predict(model_ols, data[train_size:dim(data)[1],])
ols_accuracy <- mean((predictions > 0.5) == data[train_size:dim(data)[1],]$label)

model_logit <- glm(label ~ open + high + low + close, data = data[0:train_size,], family = "binomial")
predictions <- predict(model_logit, data[train_size:dim(data)[1],])
logit_accuracy <- mean((predictions > 0.5) == data[train_size:dim(data)[1],]$label)

model_forest <- randomForest(label ~ open + high + low + close, data = data[0:train_size,])
predictions <- predict(model_forest, data[train_size:dim(data)[1],])
forest_accuracy <- mean((predictions > 0.5) == data[train_size:dim(data)[1],]$label)

model_svm <- svm(label ~ open + high + low + close, data = data[0:train_size,])
predictions <- predict(model_svm, data[train_size:dim(data)[1],])
svm_accuracy <- mean((predictions > 0.5) == data[train_size:dim(data)[1],]$label)
