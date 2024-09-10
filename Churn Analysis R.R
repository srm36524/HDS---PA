# Churn Analysis ML Project for Harvard Data Science by SRM

# Install necessary packages (if not already installed)
# Install necessary packages (if not already installed)
if (!requireNamespace("dplyr", quietly = TRUE)) install.packages("dplyr")
if (!requireNamespace("tidyverse", quietly = TRUE)) install.packages("tidyverse")
if (!requireNamespace("caret", quietly = TRUE)) install.packages("caret")
if (!requireNamespace("randomForest", quietly = TRUE)) install.packages("randomForest")
if (!requireNamespace("e1071", quietly = TRUE)) install.packages("e1071")
if (!requireNamespace("ggplot2", quietly = TRUE)) install.packages("ggplot2")
if (!requireNamespace("gridExtra", quietly = TRUE)) install.packages("gridExtra")
if (!requireNamespace("rpart", quietly = TRUE)) install.packages("rpart")
if (!requireNamespace("rpart.plot", quietly = TRUE)) install.packages("rpart.plot")
if (!requireNamespace("corrplot", quietly = TRUE)) install.packages("corrplot")
if (!requireNamespace("pROC", quietly = TRUE)) install.packages("pROC")

# Load libraries
library(tidyverse)
library(caret)
library(randomForest)
library(e1071)
library(ggplot2)
library(gridExtra)
library(corrplot)
library(rpart)
library(rpart.plot)
library(pROC)
library(dplyr)

# Load the dataset (use the raw URL from GitHub)
data <- read.csv("https://raw.githubusercontent.com/srm36524/HDS---PA/main/Customer%20Churn.csv", header = TRUE)

# Check the structure of the data
str(data)
summary(data)

# Check for missing values and removing Age column since it is a duplicate variable
sum(is.na(data))
data <- data[, !names(data) %in% c("Age")]

# Subset data for EDA (selecting the variables affecting the churn)
subset_data <- data %>%
  select(CallFailure, SubscriptionLength, ChargeAmount,
         SecondsofUse, Frequencyofuse, FrequencyofSMS, DistinctCalledNumbers, AgeGroup)

# Summary statistics and visualizations for EDA
summary(subset_data)
pairs(subset_data)
cor_matrix <- cor(subset_data, use = "complete.obs")
corrplot(cor_matrix, method = "circle", type = "upper", tl.cex = 0.7)

# Factorize relevant columns for modeling
data$Complains <- factor(data$Complains, levels = c(0, 1))
data$ChargeAmount <- factor(data$ChargeAmount, levels = 0:10)
data$AgeGroup <- factor(data$AgeGroup, levels = 1:5)
data$TariffPlan <- factor(data$TariffPlan, levels = c(1, 2))
data$Status <- factor(data$Status, levels = c(1, 2))
data$Churn <- factor(data$Churn, levels = c(0, 1))

# Set seed for reproducibility
set.seed(123)

# Split data into training (80%) and testing (20%) sets
trainIndex <- createDataPartition(data$Churn, p = 0.8, list = FALSE)
traindata <- data[trainIndex, ]
testdata <- data[-trainIndex, ]

# Build Random Forest model
model_rf <- randomForest(Churn ~ ., data = traindata)
predictions_rf <- predict(model_rf, newdata = testdata)
cm_rf <- confusionMatrix(predictions_rf, testdata$Churn)

# Build SVM model
model_svm <- svm(Churn ~ ., data = traindata, probability = TRUE)
predictions_svm <- predict(model_svm, newdata = testdata, probability = TRUE)
cm_svm <- confusionMatrix(predictions_svm, testdata$Churn)

# Build Decision Tree model
model_tree <- rpart(Churn ~ ., data = traindata, method = "class")
predictions_tree <- predict(model_tree, newdata = testdata, type = "class")
cm_tree <- confusionMatrix(predictions_tree, testdata$Churn)

# Print confusion matrix tables
print("Random Forest Confusion Matrix:")
print(cm_rf$table)

print("SVM Confusion Matrix:")
print(cm_svm$table)

print("Decision Tree Confusion Matrix:")
print(cm_tree$table)

# Function to calculate metrics from confusion matrix
calculate_metrics <- function(cm) {
  cm_table <- cm$table
  TP <- cm_table[2, 2]  # True Positives
  FP <- cm_table[1, 2]  # False Positives
  TN <- cm_table[1, 1]  # True Negatives
  FN <- cm_table[2, 1]  # False Negatives
  
  precision <- TP / (TP + FP)
  recall <- TP / (TP + FN)
  f1_score <- 2 * (precision * recall) / (precision + recall)
  accuracy <- (TP + TN) / (TP + TN + FP + FN)
  
  return(list(
    Accuracy = accuracy,
    Precision = precision,
    Recall = recall,
    F1_Score = f1_score
  ))
}

# Calculate metrics for each model
metrics_rf <- calculate_metrics(cm_rf)
metrics_svm <- calculate_metrics(cm_svm)
metrics_tree <- calculate_metrics(cm_tree)

# Create a data frame for plotting
metrics_df <- data.frame(
  Model = c("Random Forest", "SVM", "Decision Tree"),
  Accuracy = c(metrics_rf$Accuracy, metrics_svm$Accuracy, metrics_tree$Accuracy),
  Precision = c(metrics_rf$Precision, metrics_svm$Precision, metrics_tree$Precision),
  Recall = c(metrics_rf$Recall, metrics_svm$Recall, metrics_tree$Recall),
  F1_Score = c(metrics_rf$F1_Score, metrics_svm$F1_Score, metrics_tree$F1_Score)
)

# Print the metrics table
print(metrics_df)

# Convert data frame to long format for ggplot
metrics_long <- metrics_df %>%
  pivot_longer(cols = -Model, names_to = "Metric", values_to = "Value")

# Plot the metrics using ggplot2
ggplot(metrics_long, aes(x = Model, y = Value, fill = Metric)) +
  geom_bar(stat = "identity", position = position_dodge()) +
  theme_minimal() +
  labs(title = "Comparison of Model Performance Metrics",
       x = "Model",
       y = "Value") +
  scale_fill_manual(values = c("Accuracy" = "skyblue", "Precision" = "orange", "Recall" = "green", "F1_Score" = "purple")) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Calculate AUC for Random Forest
roc_rf <- roc(testdata$Churn, as.numeric(predictions_rf))
auc_rf <- auc(roc_rf)

# Get predicted probabilities for SVM
probabilities_svm <- attr(predict(model_svm, newdata = testdata, probability = TRUE), "probabilities")
scores_svm <- probabilities_svm[, "1"]

# Create ROC curve and calculate AUC for SVM
roc_svm <- roc(testdata$Churn, scores_svm)
auc_svm <- auc(roc_svm)

# Calculate AUC for Decision Tree
scores_tree <- predict(model_tree, newdata = testdata, type = "prob")[, "1"]
roc_tree <- roc(testdata$Churn, scores_tree)
auc_tree <- auc(roc_tree)

# Create data frame for AUC comparison
auc_table <- data.frame(
  Model = c("Random Forest", "SVM", "Decision Tree"),
  AUC = c(auc_rf, auc_svm, auc_tree)
)

# Print AUC table
cat("\nComparison of Model AUC:\n")
print(auc_table)

# Plot ROC curves
plot(roc_rf, col = "blue", main = "ROC Curves Comparison", lwd = 2)
plot(roc_svm, col = "red", add = TRUE, lwd = 2)
plot(roc_tree, col = "green", add = TRUE, lwd = 2)

# Add a legend
legend("bottomright", legend = c("Random Forest", "SVM", "Decision Tree"),
       col = c("blue", "red", "green"), lwd = 2)

# Set up plot margins to ensure the title does not overlap
par(mar = c(5, 4, 4, 2) + 0.1)  # Default margins usually work well

# Create bar plot for AUC comparison
barplot(auc_table$AUC, names.arg = auc_table$Model,
        main = "AUC Comparison",
        col = rainbow(nrow(auc_table)),
        ylim = c(0, max(auc_table$AUC) + 0.1),  # Extend y-axis to avoid clipping
        xlab = "Model",  # Label for x-axis
        ylab = "AUC")    # Label for y-axis


