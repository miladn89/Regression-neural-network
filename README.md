# Regression-neural-network

```{r}
set.seed(12305946)
library(tidyverse)
library(neuralnet)
library(GGally)

getwd()
setwd("C:\\Users\\Owner\\OneDrive - Mississippi State University\\00. NSF\\00. Second paper\\0. last analysis\\00. Dr Eakin analysis outline")
list.files()
Ourdata <-read.csv("datafeatures.csv")

dim(Ourdata)

Ourdata1 <- read_table(file="datafeatures.csv", col_names = c('Total_agg_ST', 'Gender','Age','Ethinicity','Father_education','Mother_education','Sibling','Houshold_status','Houshold_living_status','Car_status','Parents_checking','Parent_concern_need','GPA','Time_Academic','Time_Extracurricular','Time_work','Time_social','Time_family','Time_life','Co_op','Internship','study_major','Financial_aids','Grant','Scholarship','Loan')) %>%
  na.omit()
#ggpairs(Ourdata1, title = "Scatterplot Matrix of the Features of the Data Set")

# Split into test and train sets
set.seed(12305946)
Ourdata_Train <- sample_frac(tbl = Ourdata, replace = FALSE, size = 0.80)
Ourdata_Test <- anti_join(Ourdata, Ourdata_Train)


########NN1 1st Regression ANN, 1-Hidden Layer, 1-neuron

set.seed(12305946)
Our_NN1 <- neuralnet(Total_agg_ST ~ Gender+Age+Ethinicity+Father_education+Mother_education+Sibling+Houshold_status+Houshold_living_status+Car_status+Parents_checking+Parent_concern_need+GPA+Time_Academic+Time_Extracurricular+Time_work+Time_social+Time_family+Time_life+Co_op+Internship+study_major+Financial_aids+Grant+Scholarship+Loan, data = Ourdata_Train)

#Our_NN1

NN1_Train_RMSE <- (sum((as.numeric(as.character(unlist(Our_NN1$net.result))) - Ourdata_Train[, 10])^2)/65)^.5
paste("Training NN1 RMSE: ", round(NN1_Train_RMSE, 2))

Test_NN1_Output <- compute(Our_NN1, Ourdata_Test[, 14:38])$net.result


NN1_Test_RMSE <- (sum((as.numeric(as.character(unlist(Test_NN1_Output))) - Ourdata_Test[, 10])^2)/65)^.5

paste("Testing NN1 RMSE: ", round(NN1_Test_RMSE, 2))

plot(Our_NN1, rep = 'best')


#########Regression Hyperparameters


############ NN2 2-Hidden Layers, Layer-1 4-neurons, Layer-2, 1-neuron, logistic activation function
set.seed(12305946)
Our_NN2 <- neuralnet(Total_agg_ST ~ Gender+Age+Ethinicity+Father_education+Mother_education+Sibling+Houshold_status+Houshold_living_status+Car_status+Parents_checking+Parent_concern_need+GPA+Time_Academic+Time_Extracurricular+Time_work+Time_social+Time_family+Time_life+Co_op+Internship+study_major+Financial_aids+Grant+Scholarship+Loan, 
                       data = Ourdata_Train, 
                       hidden = c(4, 1), 
                       act.fct = "logistic",
                     rep = 40)

## Training Error
NN2_Train_RMSE <- (sum((as.numeric(as.character(unlist(Our_NN2$net.result))) - Ourdata_Train[, 10])^2)/65)^.5

## Test Error
Test_NN2_Output <- compute(Our_NN2, Ourdata_Test[, 14:38])$net.result
#show(Test_NN2_Output)
NN2_Test_RMSE <- (sum((as.numeric(as.character(unlist(Test_NN2_Output))) - Ourdata_Test[, 10])^2)/65)^.5

# Rescale for tanh activation function
scale11 <- function(x) {
    (2 * ((x - min(x))/(max(x) - min(x)))) - 1
}
Ourdata_Train <- Ourdata_Train %>% mutate_all(scale11)
Ourdata_Test <- Ourdata_Test %>% mutate_all(scale11)

paste("Training NN2 RMSE: ", round(NN2_Train_RMSE, 2))
paste("Testing NN2 RMSE: ", round(NN2_Test_RMSE, 2))

plot(Our_NN2, rep = "best")



########NN3 2-Hidden Layers, Layer-1 4-neurons, Layer-2, 1-neuron, tanh activation {tangent hyperbolicus}
# function
set.seed(12305946)
Our_NN3 <- neuralnet(Total_agg_ST ~ Gender+Age+Ethinicity+Father_education+Mother_education+Sibling+Houshold_status+Houshold_living_status+Car_status+Parents_checking+Parent_concern_need+GPA+Time_Academic+Time_Extracurricular+Time_work+Time_social+Time_family+Time_life+Co_op+Internship+study_major+Financial_aids+Grant+Scholarship+Loan, 
                       data = Ourdata_Train, 
                       hidden = c(4, 1), 
                       act.fct = "tanh",
                     rep = 40)

## Training Error
NN3_Train_RMSE <- (sum((as.numeric(as.character(unlist(Our_NN3$net.result))) - Ourdata_Train[, 10])^2)/65)^.5

## Test Error
Test_NN3_Output <- compute(Our_NN3, Ourdata_Test[, 14:38])$net.result
NN3_Test_RMSE <- (sum((as.numeric(as.character(unlist(Test_NN3_Output))) - Ourdata_Test[, 10])^2)/65)^.5

paste("Training NN3 RMSE: ", round(NN3_Train_RMSE, 2))
paste("Testing NN3 RMSE: ", round(NN3_Test_RMSE, 2))

plot(Our_NN3, rep = "best")


#########NN4 1-Hidden Layer, 1-neuron, tanh activation function
set.seed(12305946)
Our_NN4 <- neuralnet(Total_agg_ST ~ Gender+Age+Ethinicity+Father_education+Mother_education+Sibling+Houshold_status+Houshold_living_status+Car_status+Parents_checking+Parent_concern_need+GPA+Time_Academic+Time_Extracurricular+Time_work+Time_social+Time_family+Time_life+Co_op+Internship+study_major+Financial_aids+Grant+Scholarship+Loan, 
                       data = Ourdata_Train, 
                       act.fct = "tanh",
                     rep = 40)

## Training Error
NN4_Train_RMSE <- (sum((as.numeric(as.character(unlist(Our_NN4$net.result))) - Ourdata_Train[, 10])^2)/65)^.5

## Test Error
Test_NN4_Output <- compute(Our_NN4, Ourdata_Test[, 14:38])$net.result
NN4_Test_RMSE <- (sum((as.numeric(as.character(unlist(Test_NN4_Output))) - Ourdata_Test[, 10])^2)/65)^.5


paste("Training NN4 RRMSE: ", round(NN4_Train_RMSE, 2))
paste("Testing NN4 RMSE: ", round(NN4_Test_RMSE, 2))

plot(Our_NN4, rep = "best")


######## Bar plot of results
Regression_NN_Errors <- tibble(Network = rep(c("NN1", "NN2", "NN3", "NN4"), each = 2), 
                               DataSet = rep(c("Train", "Test"), time = 4), 
                               RMSE = c(NN1_Train_RMSE, NN1_Test_RMSE, 
                                       NN2_Train_RMSE, NN2_Test_RMSE, 
                                       NN3_Train_RMSE, NN3_Test_RMSE, 
                                       NN4_Train_RMSE, NN4_Test_RMSE))

Regression_NN_Errors %>% 
  ggplot(aes(Network, RMSE, fill = DataSet)) + 
  geom_col(position = "dodge") + 
  ggtitle("Regression ANN's RMSE")
```
