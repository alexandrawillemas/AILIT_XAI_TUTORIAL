# ------------------------------------- Load Libraries
list.of.packages <- c("vtable","dplyr","ggplot","reshape","sjPlot","caret","ROCR","randomForest","tree")
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)

library(vtable)
library(dplyr)
library(ggplot)
library(reshape)
library(sjPlot)
library(caret)
library(ROCR)
library(randomForest)
library(tree)

# ------------------------------------- Load Data
german_credit <- read.table("http://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data")
german_credit

colnames(german_credit) <- c("chk_acct", "duration", "credit_his", "purpose", 
                             "amount", "saving_acct", "present_emp", "installment_rate", "sex", "other_debtor", 
                             "present_resid", "property", "age", "other_install", "housing", "n_credits", 
                             "job", "n_people", "telephone", "foreign", "response")

# Change response type from 1=default risk and 2=no-default risk to 1=default risk and 0=no-default risk
german_credit$response <- german_credit$response - 1 
# Change data type to binary
german_credit$response <- as.factor(german_credit$response)

# ------------------------------------- Exploratory Data Analysis (EDA)
vtable::st(german_credit)

# Investigate age variable (numerical variable)
ggplot(melt(german_credit[,c(13,21)]), aes(x = variable, y = value, fill = response)) + 
  geom_boxplot()+ xlab("response") + ylab("age")

# Investigate chk_acct variable (categorical variable)
ggplot(german_credit, aes(chk_acct, ..count..)) + 
  geom_bar(aes(fill = response), position = "dodge") 

# ------------------------------------- Create train and test data split
df <- german_credit
data <- sort(sample(nrow(df), nrow(df)*.7))
set.seed(123)
train <- df[data,]
test <- df[-data,]

# ------------------------------------- Modelling

# ------------------------------------- Modelling: Logistic Regression
# All-in model
m1 <- glm(response ~ ., data = train, family=binomial)  

sjPlot::tab_model(m1, auto.label = FALSE)

# 2 variables model
m2 <- glm(response ~ chk_acct  
          + age, data = train, family=binomial)

sjPlot::tab_model(m1, m2)

# Predict on test data set
predicted_test <- predict(m2, newdata=test, type="response")
results_test <- ifelse(predicted_test > 0.5,1,0)
vec_obs <- as.factor(test$response)
results_Test_kat <- as.factor(results_test)
table(results_Test_kat, test$response)

# Performance
cm <- caret::confusionMatrix(data=results_Test_kat, reference=vec_obs)
accuracy_logReg <- round(cm$overall[1],2)
accuracy_logReg
# ------------------------------------- Modelling: Random Forest 
m3 <- randomForest::randomForest(response ~., data = train, ntree=100, importance=T, proximity=T)
m3

# Predict on test data set
rf_predicted_test <- predict(m3, newdata=test, type="class")
table(rf_predicted_test, test$response)

# Performance
cm_rf <- confusionMatrix(data=rf_predicted_test, reference=test$response)
accuracy_tree <- round(cm_rf$overall[1],2)
accuracy_tree

# ------------------------------------- Modelling: Tree
m4 <- tree(response ~ ., data=train, method="class")
summary(m4)
plot(m4)
text(m4, pretty=0,cex=0.5)

# Predict on test data set
tree_predicted_test <- predict(m4, newdata=test, type="class")
table(tree_predicted_test, test$response)

prune_tree <- prune.misclass(m3, best=7)
prune_tree_predicted_test <- predict(prune_tree, newdata=test, type="class")
table(prune_tree_predicted_test, test$Creditability)

cm_tree <- confusionMatrix(data=tree_predicted_test, reference=test$response)
accuracy_tree <- round(cm_tree$overall[1],2)
accuracy_tree
