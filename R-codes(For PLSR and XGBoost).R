#Set your working directory here
setwd(" ")

#Load your dataset 
load("Dataset//nonRaCAspec.RData")

#install.packages(c("caret", "doSNOW", "tictoc", "xgboost", "pls", "plotly", "dplyr"))

library(doSNOW)
library(caret)
library(tictoc)
library(pls)
library(xgboost)
library(ggplot2)

length(unique(spec.M$smp_id)) == nrow(spec.M) #No duplicated samples found

df <- dplyr::select(spec.M, "TN", "TS", "EOC", dplyr::num_range("X",350:2500))

#to remove the spectrum that has value greater than or equal to 1
sum(df[,-(1:3)] >= 1) #None found

# Set up for parallel processing
cl <- makeCluster(3, type = "SOCK")
# Register the cluster
registerDoSNOW(cl)
# Use tic() and toc() to keep track of model run time
tic("Runtime: ")

# Find Principal components
PC <- prcomp(df[,-(1:3)])

stopCluster(cl)
toc()

#head(PC$x[,1:10])

# % variance explained by the top 17 PCs
round((PC$sdev/sum(PC$sdev))*100, 2)[1:17]
screeplot(PC)

# Use ggplot to plot PC 1 and 2
ggplot(as.data.frame(PC$x[, 1:2]), aes(PC1, PC2 )) +
  geom_point(colour ="grey") +
  labs(title="First two PCs in original dataset") +
  theme_bw()

par(mfrow=c(1,1))

#To get indexes of samples to remove
out <- which(PC$x[,2]>8); out

#Lets remove these samples from the dataset
df <- df[-out,]

#Set seed for random number generations
set.seed(1521)

#Get random samples to break into train and test dataset
indexes <- sample(1:nrow(df), (0.7*nrow(df)))

datatrain <- df[indexes, 3:ncol(df)]; colnames(datatrain)[1] <- "RES"

datatest <- df[-indexes, 3:ncol(df)]; colnames(datatest)[1] <- "RES"

rm(spec.M)

#==============================================================#
#               To employ PLSR
#==============================================================#

#Tuning parameters for PLS modeling
plsGrid <-  expand.grid(ncomp = seq(1:100))

#Cross validation parameters
cvCtrl <- trainControl(method = "cv", 
                       number = 10)

# Create a socket cluster using 3 processes
cl <- makeCluster(3, type = "SOCK")

#Register cluster
registerDoSNOW(cl)

Sys.time()

#tic() and toc() used to capture model training time
tic("Model Fitting")

#Model calibration
tune.pls <- train(RES~.,
                  data = datatrain,
                  method = 'kernelpls',
                  tuneGrid = plsGrid,
                  trControl = cvCtrl,
                  na.action = na.omit,
                  verbose = F)

stopCluster(cl)

toc()

Sys.time()

#Best models cross-validation results
print(tune.pls$results[which(tune.pls$results$RMSE==min(tune.pls$results$RMSE)),c('ncomp','RMSE','Rsquared')])

sd(datatrain$RES)/min(tune.pls$results$RMSE) #<- RPD( Ratio of Performance to deviation)

tune.pls

#Check the model plot to choose the number of latent vaiables
plot(tune.pls, main = "Latent factors against CV-RMSE")

#======================On training dataset===================#

preds1 <- predict(tune.pls, datatrain) # ncomp based on the RMSE
combined1 <- data.frame(preds1, datatrain[1])
head(combined1)
rmsep. <- sqrt(sum((combined1$preds1 - combined1$RES)^2)/length(combined1$RES)); rmsep.
r2. <- (cor(combined1$preds1, combined1$RES))^2;r2.

plot(combined1, xlim=c(0,10), ylim=c(0,10))

#======================On test dataset========================#

preds <- predict(tune.pls, datatest)
combined <- data.frame(preds, datatest[1])
head(combined)
rmsep <- sqrt(sum((combined$preds - combined$RES)^2)/length(combined$RES)); rmsep
r2 <- (cor(combined$preds, combined$RES))^2;r2

sd(datatest$RES)/rmsep #<- RPD( Ratio of Performance to deviation)

#Plotting the predicted vs actual of test dataset
ggplot(data = combined[(combined$RES > 0 & combined$RES < 20), ], aes(x = RES, y = preds)) +
  geom_point(col = "steelblue", size = 2) +
  geom_smooth(method = 'lm', col = "firebrick", size = 1) + 
  coord_cartesian(xlim=c(0,15), ylim=c(0, 10)) + 
  labs(title="Predicted vs Actual Soil Organic Carbon", subtitle = "Using PLSR", y="Predicted(%)", x="Actual(%)") +
  theme_bw()

# Saving the model to disk
#saveRDS(tune.pls, "model-plsr.rds")

#==============================================================#
#               To employ xgBoost
#==============================================================#

train.control <- trainControl(method = "repeatedcv",
                              number = 10,
                              repeats = 3,
                              search = "grid")

tune.grid <- expand.grid(eta = c(0.02, 0.03, 0.04),
                         nrounds = c(110, 120, 130),
                         lambda = 0,
                         alpha = 1)

cl <- makeCluster(3, type = "SOCK")

# Register cluster
registerDoSNOW(cl)

Sys.time()
tic("Model Fitting")

#Model calibration
tune.xgb <- train(RES~.,
                  data = datatrain,
                  method = 'xgbLinear', 
                  tuneGrid = tune.grid,
                  trControl = train.control,
                  na.action = na.omit)

stopCluster(cl)
toc()
Sys.time()

tune.xgb

preds <- predict(tune.xgb, datatest)
preds
combined <- data.frame(preds, datatest[1])
rmsep <- sqrt(sum((combined$preds - combined$RES)^2)/length(combined$RES))
r2 <- (cor(combined$preds, combined$RES))^2
rmsep
r2

ggplot(data = combined[(combined$RES > 0 & combined$RES < 20), ], aes(x = RES, y = preds)) +
  geom_point(col = "steelblue", size = 2) +
  geom_smooth(method = 'lm', col = "firebrick", size = 1) + 
  coord_cartesian(xlim=c(0,15), ylim=c(0, 10)) + 
  labs(title="Predicted vs Actual Soil Organic Carbon", subtitle = "Using XGBoost", y="Predicted(%)", x="Actual(%)") +
  theme_bw()


#================== Extract significant features with PCA for XGBoost ==================#

PC.train <- prcomp(datatrain[,-1])
plot(round((PC.train$sdev/sum(PC.train$sdev))*100, 2)[1:20], main = "Train dataset variance explained by PCs", xlab = "Number of PC's", ylab = "% SD explained", type = "l")
round((PC.train$sdev/sum(PC.train$sdev))*100, 2)[1:17]

cl <- makeCluster(3, type = "SOCK")
registerDoSNOW(cl)
Sys.time()
tic("Model Fitting")

tune.xgb1 <- train(EOC~.,
                  data = data.frame(EOC = datatrain$RES, PC.train$x[,1:17]),
                  method = 'xgbLinear', 
                  tuneGrid = tune.grid,
                  trControl = cvCtrl,
                  na.action = na.omit)

stopCluster(cl)
toc()                               
tune.xgb1

#Converting the test dataset
datatest1 <- predict(PC.train, datatest[,-1])
#Predicting on the transformed test dataset
preds. <- predict(tune.xgb1, datatest1)
head(preds.)
combined. <- data.frame(preds., datatest[1])
rmsep. <- sqrt(sum((combined.$preds. - combined.$RES)^2)/length(combined.$RES))
r2. <- (cor(combined.$preds., combined.$RES))^2
rmsep.
r2.

ggplot(data = combined.[(combined.$RES > 0 & combined.$RES < 20), ], aes(x = RES, y = preds.)) +
  geom_point(col = "steelblue", size = 2) +
  geom_smooth(method = 'lm', col = "firebrick", size = 1) + 
  coord_cartesian(xlim=c(0,15), ylim=c(0, 10)) + 
  labs(title="Predicted vs Actual Soil Organic Carbon", subtitle = "Using XGBoost using PCs", y="Predicted(%)", x="Actual(%)") +
  theme_bw()

