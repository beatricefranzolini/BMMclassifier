#Code used for 
#How to leverage Bayesian mixtures for longitudinal clustering and classification
#by B. Franzolini (2024)

rm(list = ls())
library(VGAM) #Load package VGAM for multinomial logistic
library(randomForest) #Load package randomForest for decision trees

#############################################################

#DPM for classification
#functions: 

# Compute all the marginal likelihood of a norm-norm model
# (marg lik of a new cluster) multivariate with diagonal covariance matrix

marg_new_all_multi <- function(y, m = 0, s2 = 1, sigma2 = 1, log = TRUE){
  
  J = dim(y)[2]; sigma = sqrt(sigma2); s = sqrt(s2)
  
  p = - J / 2 * log(2 * pi) -
    J/2 * log(s2 + sigma2) -
    rowSums(y^2) / (2 * sigma2) -
    J*m^2 / (2 * s2) +
    rowSums((s * y / sigma + sigma * m / s )^2) / (2 * (s2 + sigma2))
  
  return(p)
}

#MCMC alg for DPM 
mcmc_DPM_norm_norm_multi <- function(y,
                                     hyper=c(0.1, 0, 1, 0.1),
                                     clus = FALSE, #do not use 0 as label  
                                     hprior = FALSE, #gamma prior on concentration
                                     totiter = 1000, #totiter (include burn in)
                                     verbose = max(round(totiter/20),1)){
  
  alpha  = hyper[1] #concentration parameter 
  m      = hyper[2] #mean of the base measure
  s2     = hyper[3] #variance of the base measure 
  sigma2 = hyper[4] #kernel's variance 
  s      = sqrt(s2)
  sigma  = sqrt(sigma2)
  J      = ncol(y) #number of covariates
  n      = nrow(y) #dimension of the train-set

  #if no inizialization is provided, clusters are set to k-means with k=3  
  if(!clus[1]){clus = kmeans(y, centers = 3, nstart = 20)$cluster}
  
  print(paste("initialization to", length(unique(clus)), "clusters"))
  
  c_saved = matrix(nrow=totiter, ncol=n)
  theta_saved = array(dim=c(totiter, n, J))
  
  #compute the marginal likelihood of each data point
  mnew = marg_new_all_multi(y, m, s2, sigma2, log = TRUE)
  
  for (iter in 1:totiter){
    
    for (i in 1:n){
      clus[i] = NA
      c_unique = unique(clus[!is.na(clus)])
      prob = double(length(c_unique)+1)
      y_i  = y[i,]
      j    = 1 #cluster counter
      temp = 0
      
      for (cc in c_unique){
        which_cc = which(clus == cc & !is.na(clus))
        n_cc     = length(which_cc)
        
        # difference of marginal liks in cc
        temp     = - J / 2 * log(2 * pi) -
          J * log(sigma) -
          sum(y_i^2)/ (2 * sigma2) +
          log(n_cc)  +
          J/2 * log( (n_cc *s2 + sigma2)/((n_cc+1) * s2 + sigma2))
        
        for (jj in 1:J){
          
          Y_cc_jj = sum(y[which_cc,jj])
          temp = temp +
            (s * (Y_cc_jj+y_i[jj])/sigma+sigma*m/s)^2 / (2*((n_cc+1)*s2+sigma2)) -
            (s * Y_cc_jj/sigma+sigma*m/s)^2 / (2*(n_cc*s2+sigma2))
        }
        
        prob[j] = temp
        
        j = j + 1
      }
      
      # new cc
      prob[j] = mnew[i] + log(alpha)
      
      #sample cluster assignment
      prob    = exp(prob - max(prob))
      clus[i] = sample(c(c_unique, setdiff(1:n,c_unique)[1]), 1, prob = prob)
    }
    
    #sample cluster locations 
    for (cc in unique(clus)){ #for each cluster
      n_cc     = sum(clus == cc)
      theta_saved[iter, clus == cc, ] = (m/s2 + 
                                colSums(y[clus == cc,,drop = FALSE])/sigma2)/
                                (1/s2 + n_cc/sigma2)
      #theta_saved[iter, clus == cc, ] =  colMeans(y[clus == cc,,drop = FALSE])
      }
    
    c_saved[iter,] = clus
    
    if(hprior[1]){
      k = length(unique(clus))
      eta  = rbeta(1, alpha+1, n)
      zeta = sample(c(0,1), 1, prob = c(hprior[1] + k + 1, n*(hprior[2] - log(eta))))
      
      alpha = rgamma(1, hprior[1] + k - zeta, hprior[2] - log(eta) )
    }
    
    if(iter%%verbose==0){
      print(iter)
    }
  }
  
  return(list(c = c_saved, mean = theta_saved))
}


#IRIS dataset #train = 80%
table(iris$Species)

#division between train and test
train.data = iris[c(1:40, 5 1:90, 101:140), ]
test.data = iris[c(41:50, 91:100, 141:150), ]


#classification via BMM #train = 80%
#train the model
train_setosa = scale(as.matrix(train.data[train.data$Species == "setosa", 1:4 ]))
dens_setosa = mcmc_DPM_norm_norm_multi(train_setosa, totiter = 10000)

train_versicolor = scale(as.matrix(train.data[train.data$Species == "versicolor", 1:4 ]))
dens_versicolor = mcmc_DPM_norm_norm_multi(train_versicolor, totiter = 10000)

train_virginica  = scale(as.matrix(train.data[train.data$Species == "virginica", 1:4 ]))
dens_virginica  = mcmc_DPM_norm_norm_multi(train_virginica , totiter = 10000)

#predict the test set
p_setosa = rep(0, nrow(test.data))
p_versicolor = rep(0, nrow(test.data))
p_virginica = rep(0, nrow(test.data))

#standardization remapping 
mean_setosa = colMeans(train.data[train.data$Species == "setosa", 1:4 ])
sd_setosa = 1/ sapply(train.data[train.data$Species == "setosa", 1:4 ], sd)
mean_versicolor = colMeans(train.data[train.data$Species == "versicolor", 1:4 ])
sd_versicolor = 1/ sapply(train.data[train.data$Species == "versicolor", 1:4 ], sd)
mean_virginica = colMeans(train.data[train.data$Species == "virginica", 1:4 ])
sd_virginica = 1/ sapply(train.data[train.data$Species == "virginica", 1:4 ], sd)

for(i in 1:nrow(test.data)){
  temps = 0 
  tempve = 0
  tempvi = 0
  for(j in 1:4){
    temps = temps + dnorm((test.data[i,j] - mean_setosa[j]) * sd_setosa[j],
                   dens_setosa$mean[5001:10000,,j],sqrt(0.1), log = TRUE)
    tempve = tempve + 
      dnorm((test.data[i,j] - mean_versicolor[j]) * sd_versicolor[j],
                dens_versicolor$mean[5001:10000,,j], sqrt(0.1), log = TRUE)
    tempvi = tempvi +
      dnorm((test.data[i,j] - mean_virginica[j]) * sd_virginica[j],
                dens_virginica$mean[5001:10000,,j], sqrt(0.1), log = TRUE)
  }
  p_setosa[i] = mean(exp(temps)); p_versicolor[i] = mean(exp(tempve))
  p_virginica[i] = mean(exp(tempvi))
}
class_predict = rep(NA, nrow(test.data))
class_predict[( p_setosa >= p_versicolor) & ( p_setosa >= p_virginica )] = "setosa" 
class_predict[( p_versicolor >= p_setosa) & ( p_versicolor >= p_virginica )] = "versicolor" 
class_predict[( p_virginica >= p_setosa) & ( p_virginica >= p_versicolor )] = "virginica" 

table(test.data$Species, class_predict) 
accuracy = sum(class_predict == test.data$Species) / nrow(test.data)

print("accuracy of non-naive Bayes is")

print(accuracy)


#classification via logistic regression #train = 80%
fit.MLR <- vglm( Species ~ 
                   Sepal.Length + Sepal.Width + Petal.Length + Petal.Width, 
                 family=multinomial, train.data)
summary(fit.MLR)
probabilities.MLR <- predict(fit.MLR, test.data[,1:4], type="response")
predictions <- apply(probabilities.MLR, 1, which.max)

table(test.data$Species, predictions) 

#classification via random forest #train = 80%
decision_tree <- randomForest(Species ~., 
                              data = train.data, #train data set 
                              importance = T) 
predicted_table <- predict(decision_tree, test.data[,1:4])

accuracy = sum(predicted_table == test.data$Species) / nrow(test.data)

print("accuracy of randomForest is")

print(accuracy)

###############################################################################
#IRIS dataset #train = 50%
table(iris$Species)

#division between train and test
train.data = iris[c(1:25, 51:75, 101:125), ]
test.data = iris[c(26:50, 76:100, 126:150), ]


#classification via BMM #train = 50%
#train the model
train_setosa = scale(as.matrix(train.data[train.data$Species == "setosa", 1:4 ]))
dens_setosa = mcmc_DPM_norm_norm_multi(train_setosa, totiter = 10000)

train_versicolor = scale(as.matrix(train.data[train.data$Species == "versicolor", 1:4 ]))
dens_versicolor = mcmc_DPM_norm_norm_multi(train_versicolor, totiter = 10000)

train_virginica  = scale(as.matrix(train.data[train.data$Species == "virginica", 1:4 ]))
dens_virginica  = mcmc_DPM_norm_norm_multi(train_virginica , totiter = 10000)

#predict the test set
p_setosa = rep(0, nrow(test.data))
p_versicolor = rep(0, nrow(test.data))
p_virginica = rep(0, nrow(test.data))

#standardization remapping 
mean_setosa = colMeans(train.data[train.data$Species == "setosa", 1:4 ])
sd_setosa = 1/ sapply(train.data[train.data$Species == "setosa", 1:4 ], sd)
mean_versicolor = colMeans(train.data[train.data$Species == "versicolor", 1:4 ])
sd_versicolor = 1/ sapply(train.data[train.data$Species == "versicolor", 1:4 ], sd)
mean_virginica = colMeans(train.data[train.data$Species == "virginica", 1:4 ])
sd_virginica = 1/ sapply(train.data[train.data$Species == "virginica", 1:4 ], sd)

for(i in 1:nrow(test.data)){
  temps = 0 
  tempve = 0
  tempvi = 0
  for(j in 1:4){
    temps = temps + dnorm((test.data[i,j] - mean_setosa[j]) * sd_setosa[j],
                          dens_setosa$mean[5001:10000,,j],sqrt(0.1), log = TRUE)
    tempve = tempve + 
      dnorm((test.data[i,j] - mean_versicolor[j]) * sd_versicolor[j],
            dens_versicolor$mean[5001:10000,,j], sqrt(0.1), log = TRUE)
    tempvi = tempvi +
      dnorm((test.data[i,j] - mean_virginica[j]) * sd_virginica[j],
            dens_virginica$mean[5001:10000,,j], sqrt(0.1), log = TRUE)
  }
  p_setosa[i] = mean(exp(temps)); p_versicolor[i] = mean(exp(tempve))
  p_virginica[i] = mean(exp(tempvi))
}
class_predict = rep(NA, nrow(test.data))
class_predict[( p_setosa >= p_versicolor) & ( p_setosa >= p_virginica )] = "setosa" 
class_predict[( p_versicolor >= p_setosa) & ( p_versicolor >= p_virginica )] = "versicolor" 
class_predict[( p_virginica >= p_setosa) & ( p_virginica >= p_versicolor )] = "virginica" 

table(test.data$Species, class_predict) 
accuracy = sum(class_predict == test.data$Species) / nrow(test.data)

print("accuracy of non-naive Bayes is")

print(accuracy)


#classification via logistic regression #train = 50%
fit.MLR <- vglm( Species ~ 
                   Sepal.Length + Sepal.Width + Petal.Length + Petal.Width, 
                 family=multinomial, train.data)
summary(fit.MLR)
probabilities.MLR <- predict(fit.MLR, test.data[,1:4], type="response")
predictions <- apply(probabilities.MLR, 1, which.max)

class_predict_log = rep(NA, nrow(test.data))
class_predict_log[predictions == 1] = "setosa" 
class_predict_log[predictions == 2] = "versicolor" 
class_predict_log[predictions == 3] = "virginica"
#table(test.data$Species, predictions) 
accuracy = sum(class_predict_log == test.data$Species) / nrow(test.data)

print("accuracy of logistic is")

print(accuracy)

#classification via random forest #train = 50%
decision_tree <- randomForest(Species ~., 
                              data = train.data, #train data set 
                              importance = T) 
predicted_table <- predict(decision_tree, test.data[,1:4])

accuracy = sum(predicted_table == test.data$Species) / nrow(test.data)

print("accuracy of randomForest is")

print(accuracy)

###############################################################################
#IRIS dataset #train = 30%
table(iris$Species)

#division between train and test
train.data = iris[c(1:15, 51:65, 101:115), ]
test.data = iris[c(16:50, 66:100, 116:150), ]


#classification via BMM #train = 30%
#train the model
train_setosa = scale(as.matrix(train.data[train.data$Species == "setosa", 1:4 ]))
dens_setosa = mcmc_DPM_norm_norm_multi(train_setosa, totiter = 10000)

train_versicolor = scale(as.matrix(train.data[train.data$Species == "versicolor", 1:4 ]))
dens_versicolor = mcmc_DPM_norm_norm_multi(train_versicolor, totiter = 10000)

train_virginica  = scale(as.matrix(train.data[train.data$Species == "virginica", 1:4 ]))
dens_virginica  = mcmc_DPM_norm_norm_multi(train_virginica , totiter = 10000)

#predict the test set
p_setosa = rep(0, nrow(test.data))
p_versicolor = rep(0, nrow(test.data))
p_virginica = rep(0, nrow(test.data))

#standardization remapping 
mean_setosa = colMeans(train.data[train.data$Species == "setosa", 1:4 ])
sd_setosa = 1/ sapply(train.data[train.data$Species == "setosa", 1:4 ], sd)
mean_versicolor = colMeans(train.data[train.data$Species == "versicolor", 1:4 ])
sd_versicolor = 1/ sapply(train.data[train.data$Species == "versicolor", 1:4 ], sd)
mean_virginica = colMeans(train.data[train.data$Species == "virginica", 1:4 ])
sd_virginica = 1/ sapply(train.data[train.data$Species == "virginica", 1:4 ], sd)

for(i in 1:nrow(test.data)){
  temps = 0 
  tempve = 0
  tempvi = 0
  for(j in 1:4){
    temps = temps + dnorm((test.data[i,j] - mean_setosa[j]) * sd_setosa[j],
                          dens_setosa$mean[5001:10000,,j],sqrt(0.1), log = TRUE)
    tempve = tempve + 
      dnorm((test.data[i,j] - mean_versicolor[j]) * sd_versicolor[j],
            dens_versicolor$mean[5001:10000,,j], sqrt(0.1), log = TRUE)
    tempvi = tempvi +
      dnorm((test.data[i,j] - mean_virginica[j]) * sd_virginica[j],
            dens_virginica$mean[5001:10000,,j], sqrt(0.1), log = TRUE)
  }
  p_setosa[i] = mean(exp(temps)); p_versicolor[i] = mean(exp(tempve))
  p_virginica[i] = mean(exp(tempvi))
}
class_predict = rep(NA, nrow(test.data))
class_predict[( p_setosa >= p_versicolor) & ( p_setosa >= p_virginica )] = "setosa" 
class_predict[( p_versicolor >= p_setosa) & ( p_versicolor >= p_virginica )] = "versicolor" 
class_predict[( p_virginica >= p_setosa) & ( p_virginica >= p_versicolor )] = "virginica" 

table(test.data$Species, class_predict) 
accuracy = sum(class_predict == test.data$Species) / nrow(test.data)

print("accuracy of non-naive Bayes is")

print(accuracy)


#classification via logistic regression #train = 30%
fit.MLR <- vglm( Species ~ 
                   Sepal.Length + Sepal.Width + Petal.Length + Petal.Width, 
                 family=multinomial, train.data)
summary(fit.MLR)
probabilities.MLR <- predict(fit.MLR, test.data[,1:4], type="response")
predictions <- apply(probabilities.MLR, 1, which.max)

class_predict_log = rep(NA, nrow(test.data))
class_predict_log[predictions == 1] = "setosa" 
class_predict_log[predictions == 2] = "versicolor" 
class_predict_log[predictions == 3] = "virginica"
#table(test.data$Species, predictions) 
accuracy = sum(class_predict_log == test.data$Species) / nrow(test.data)

print("accuracy of logistic is")

print(accuracy)

#classification via random forest #train = 30%
decision_tree <- randomForest(Species ~., 
                              data = train.data, #train data set 
                              importance = T) 
predicted_table <- predict(decision_tree, test.data[,1:4])

accuracy = sum(predicted_table == test.data$Species) / nrow(test.data)

print("accuracy of randomForest is")

print(accuracy)

###############################################################################
#IRIS dataset #train = 10%
table(iris$Species)

#division between train and test
train.data = iris[c(1:5, 51:55, 101:105), ]
test.data = iris[c(6:50, 56:100, 106:150), ]


#classification via BMM #train = 10%
#train the model
train_setosa = scale(as.matrix(train.data[train.data$Species == "setosa", 1:4 ]))
train_setosa[,4] = 0
dens_setosa = mcmc_DPM_norm_norm_multi(train_setosa, clus = rep(1,5), totiter = 10000)

train_versicolor = scale(as.matrix(train.data[train.data$Species == "versicolor", 1:4 ]))
dens_versicolor = mcmc_DPM_norm_norm_multi(train_versicolor, clus = rep(1,5), totiter = 10000)

train_virginica  = scale(as.matrix(train.data[train.data$Species == "virginica", 1:4 ]))
dens_virginica  = mcmc_DPM_norm_norm_multi(train_virginica , clus = rep(1,5), totiter = 10000)

#predict the test set
p_setosa = rep(0, nrow(test.data))
p_versicolor = rep(0, nrow(test.data))
p_virginica = rep(0, nrow(test.data))

#standardization remapping 
mean_setosa = colMeans(train.data[train.data$Species == "setosa", 1:4 ])
sd_setosa = 1/ sapply(train.data[train.data$Species == "versicolor", 1:4 ], sd)
sd_setosa[4] = 1
mean_versicolor = colMeans(train.data[train.data$Species == "versicolor", 1:4 ])
sd_versicolor = 1/ sapply(train.data[train.data$Species == "versicolor", 1:4 ], sd)
mean_virginica = colMeans(train.data[train.data$Species == "virginica", 1:4 ])
sd_virginica = 1/ sapply(train.data[train.data$Species == "virginica", 1:4 ], sd)

for(i in 1:nrow(test.data)){
  temps = 0 
  tempve = 0
  tempvi = 0
  for(j in 1:4){
    temps = temps + dnorm((test.data[i,j] - mean_setosa[j]) * sd_setosa[j],
                          dens_setosa$mean[5001:10000,,j],sqrt(0.1), log = TRUE)
    tempve = tempve + 
      dnorm((test.data[i,j] - mean_versicolor[j]) * sd_versicolor[j],
            dens_versicolor$mean[5001:10000,,j], sqrt(0.1), log = TRUE)
    tempvi = tempvi +
      dnorm((test.data[i,j] - mean_virginica[j]) * sd_virginica[j],
            dens_virginica$mean[5001:10000,,j], sqrt(0.1), log = TRUE)
  }
  p_setosa[i] = mean(exp(temps)); p_versicolor[i] = mean(exp(tempve))
  p_virginica[i] = mean(exp(tempvi))
}
class_predict = rep(NA, nrow(test.data))
class_predict[( p_setosa >= p_versicolor) & ( p_setosa >= p_virginica )] = "setosa" 
class_predict[( p_versicolor >= p_setosa) & ( p_versicolor >= p_virginica )] = "versicolor" 
class_predict[( p_virginica >= p_setosa) & ( p_virginica >= p_versicolor )] = "virginica" 

table(test.data$Species, class_predict) 
accuracy = sum(class_predict == test.data$Species) / nrow(test.data)

print("accuracy of non-naive Bayes is")

print(accuracy)


#classification via logistic regression #train = 10%
fit.MLR <- vglm( Species ~ 
                   Sepal.Length + Sepal.Width + Petal.Length + Petal.Width, 
                 family=multinomial, train.data)
summary(fit.MLR)
probabilities.MLR <- predict(fit.MLR, test.data[,1:4], type="response")
predictions <- apply(probabilities.MLR, 1, which.max)

class_predict_log = rep(NA, nrow(test.data))
class_predict_log[predictions == 1] = "setosa" 
class_predict_log[predictions == 2] = "versicolor" 
class_predict_log[predictions == 3] = "virginica"
#table(test.data$Species, predictions) 
accuracy = sum(class_predict_log == test.data$Species) / nrow(test.data)

print("accuracy of logistic is")

print(accuracy)

#classification via random forest #train = 10%
decision_tree <- randomForest(Species ~., 
                              data = train.data, #train data set 
                              importance = T) 
predicted_table <- predict(decision_tree, test.data[,1:4])

accuracy = sum(predicted_table == test.data$Species) / nrow(test.data)

print("accuracy of randomForest is")

print(accuracy)

