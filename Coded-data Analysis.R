DataCoded <- read.table("spambasedata-Coded.csv",sep=",",header=T,
                        stringsAsFactors=F)


# Doing a 60-40 split
ord <- sample(nrow(DataCoded))
DataCoded <- DataCoded[ord,]

TrainInd <- ceiling(nrow(DataCoded)*0.6)
TrainData <- DataCoded[1:TrainInd,]
ValData <- DataCoded[(TrainInd+1):nrow(DataCoded),]


######1.Naive Bayes######

#calculate PWordsGivenSpam & PWordsGivenHam
P <- function(x) {
  Probs <- table(x)
  Probs <- Probs/sum(Probs)
  return(Probs)
}

PGivenSpam <- lapply(TrainData[TrainData$IsSpam==1,],FUN=P)
PGivenHam <- lapply(TrainData[TrainData$IsSpam!=1,],FUN=P)

#calculate PSpam 
PSpam <- mean(TrainData$IsSpam)

#calculate PSpamGivenWords
PP <- function(DataRow,PGivenSpam,PGivenHam,PSpam) {
  tmp1 <- 1.0
  tmp2 <- 1.0
  for(x in names(DataRow)) {
    tmp1 <- tmp1*PGivenSpam[[x]][DataRow[x]]
    tmp2 <- tmp2*PGivenHam[[x]][DataRow[x]]
  }
  out <- tmp1*PSpam/(tmp1*PSpam+tmp2*(1-PSpam))
  return(out)
}

#ROC function
source("ROCPlot.r")

#Naive Bayes-forward selection
#1-create "AUC space"
Train <- TrainData[-58]
AUC <- rep(NA,length(Train))

#2-select features (The best 10)
Begin <- NULL
Pick <- 1:57
AUCbefore <-0
Run <- TRUE

while (Run){
  AUC <- rep(NA,length(Pick))
  for (i in 1:length(Pick)){
    Final <- colnames(TrainData)[c(Begin, Pick[i])]
    PPSpam <- apply(ValData[,Final,drop=F],1,FUN=PP,PGivenSpam,PGivenHam,PSpam)
    PPSpam[is.na(PPSpam)] <- mean(PPSpam,na.rm=T)
    AUC[i] <- ROCPlot(PPSpam,ValData[,"IsSpam"],Plot=F)$AUC
  }
  AUCAfter <- max(AUC,na.rm=T)
  if (AUCAfter>AUCbefore){
    Begin <- c(Begin, Pick[which.max(AUC)])
    Pick <- Pick[-which.max(AUC)]
  }
  if (AUCAfter>AUCbefore && length(Begin)==10){
    Run = FALSE
  }
}

Begin
Final <- colnames(TrainData)[c(Begin)]
PPSpam <- apply(ValData[,Final,drop=F],1,FUN=PP,PGivenSpam,PGivenHam,PSpam)
ROCPlot(PPSpam,ValData[,"IsSpam"])


######2. Logistic Regression######
LRlower <- glm(IsSpam~ 1, data=TrainData)
LRupper=glm(IsSpam~., data=TrainData)
LRcoded <- step(LRlower, scope=list(lower=LRlower, upper=LRupper), direction="forward", steps = 10)
predLR <- predict(LRcoded, newdata=ValData)


######3. Decision-Tree######
library(rpart)
tree<-rpart(IsSpam~., data=TrainData)
printcp(tree)
library(maptree)
draw.tree(tree,nodeinfo=T,cex=0.5,col=gray(0:8/8))
predDT <- predict(tree, newdata=ValData)


######4. ensemble approach######

#Predicting the probabilities
pred_NB <- apply(ValData[,Final,drop=F],1,FUN=PP,PGivenSpam,PGivenHam,PSpam)
pred_LR <- predict(LRcoded, newdata=ValData)
pred_DT <- predict(tree, newdata=ValData)

#1. Averaging
pred_avg<-(pred_NB+pred_LR+pred_DT)/3
AUC_avg <- ROCPlot(pred_avg,ValData[,"IsSpam"],Plot=F)$AUC

#2. Weighted average
#Since AUCLR is the highest, we give logistic regression model more weight.
pred_weighted_avg<-(pred_NB*0.2)+(pred_LR*0.7)+(pred_DT*0.1)
AUC_weighted_avg <- ROCPlot(pred_weighted_avg,ValData[,"IsSpam"],Plot=F)$AUC







######Compare the models######
#1.Naive Bayes
AUCNB <- AUCAfter

#2.Logistic Regression
AUCLR <- ROCPlot(predLR,ValData[,"IsSpam"],Plot=F)$AUC

#3.Decision Tree
AUCDT <- ROCPlot(predDT,ValData[,"IsSpam"],Plot=F)$AUC

#4.Ensemble
AUC_weighted_avg

#The best model based on AUC is weighted average model.




