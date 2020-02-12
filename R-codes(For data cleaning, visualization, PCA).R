#Set your working directory here
setwd("")

#Load your dataset 
load("Dataset\\nonRaCAspec.RData")

#install.packages(c(doSNOW", tictoc", "dplyr", "plotly"))

library(doSNOW)
library(tictoc)

utils::View(colnames(spec.M))

#find duplicate spectra, if any and remove if found
length(unique(spec.M$smp_id)) #No duplicated samples found

df <- dplyr::select(spec.M, "TN", "TS", "EOC", dplyr::num_range("X",350:2500))

#remove the spectrum that has value greater than or equal to 1
sum(df[,-(1:3)] >= 1) #None found

plot(350:2500, df[2345,-(1:3)], main = "Spectral reflectance curve of Soil", xlab = "Wavelength(nm)", ylab = "% Reflectance", type = "l", ylim = c(0,1))

utils::View(head(df))

# Ribbon plot (ggplot2 ribbon) to observe the original dataset
library(ggplot2)
#a <- data.frame(wavelength = 350:2500, level = unlist(df[5, -(1:3)], use.names = F))
a <- data.frame(wavelength = 350:2500, Reflectance = sapply(df[, 4:2154], mean))
ggplot(a, aes(wavelength)) +
  geom_ribbon(aes(ymin = Reflectance - sapply(df[,4:2154], sd), ymax = Reflectance + sapply(df[,4:2154], sd)), fill = "grey") +
  geom_line(aes(y = Reflectance), col = "red", size = 1) +
  labs(title="Ribbon plot showing mean and range of reflectance curve in visNIR region", y="Reflectance(%)", x="Wavelength(nm)") +
  theme_bw()

# Set up parallel processing
cl <- makeCluster(3, type = "SOCK")
# Register the cluster
registerDoSNOW(cl)
# tic() and toc()
tic("Runtime: ")

# Find Principal components
PC <- prcomp(df[,-(1:3)])

stopCluster(cl)
toc()

head(PC$x[,1:10])

# First 17 PCs found to explain total of ~95% variance in dataset
round((PC$sdev/sum(PC$sdev))*100, 2)[1:17]
screeplot(PC)

plot(round((PC$sdev/sum(PC$sdev))*100, 2)[1:15], xlab = "Number of PC's", ylab = "% SD explained", type = "l")

# Make 3D interactive plot
library(plotly)
p<-plot_ly(data=data.frame(PC$x[,1:3]),x=~PC1,y=~PC2,z=~PC3,colors='black',opacity=0.5,color="rgb(0, 0, 0)",marker = list(size = 3))
p #For offline plotting

# #For online plotting, API keys
# Sys.setenv("plotly_username" = "UjjwolBhandari")
# Sys.setenv("plotly_api_key" = "A9Vih7o36SSM3scLWm6x")
# plotly_POST(p, filename="spec.pca.3d")
# api_create(p, filename = "ujj.3d")

par(mfrow=c(1,1))

plot(PC$x[,1],PC$x[,2])

# Using ggplot
ggplot(as.data.frame(PC$x[, 1:2]), aes(PC1, PC2 )) +
  geom_point(colour ="grey") +
  labs(title="First two PCs in original dataset") +
  theme_bw()

#To get indexes of samples to remove
out <- which(PC$x[,2]>8); out
#Checking one sample
plot(350:2500, df[out[22],-(1:3)], main = "One outlier detected from PC plot", xlab = "Wavelength(nm)", ylab = "Reflectance", type = "l", ylim = c(0,1))

#Lets remove these samples from the dataset and see new PCs
df <- df[-out,]

cl <- makeCluster(3, type = "SOCK")
registerDoSNOW(cl)
tic("Runtime: ")

PC.new <- prcomp(df[,-(1:3)])

stopCluster(cl)
toc()
plot(round((PC.new$sdev/sum(PC.new$sdev))*100, 2)[1:17], xlab = "Number of PC's", ylab = "% SD explained", type = "l")
round((PC.new$sdev/sum(PC.new$sdev))*100, 2)[1:17]

#Set seed for random number generations
set.seed(1521)

#Get random samples to break into train and test dataset
indexes <- sample(1:nrow(df), (0.7*nrow(df)))

datatrain <- df[indexes, 3:ncol(df)]; colnames(datatrain)[1] <- "RES"

datatest <- df[-indexes, 3:ncol(df)]; colnames(datatest)[1] <- "RES"

rm(spec.M)

# Saving the dataset to disk 
#save(df,file = "nonRaCA.RData")

