

library(tidyverse)
library(tidyquant)
library(dplyr)
library(caret)
library(h2o)


# Step 1: Load the training & test dataset
# Load the dataset

dataset <- read.csv("data/product_backorders.csv") 
glimpse(dataset)

dataset_h2o <- h2o.importFile("data/product_backorders.csv")

# Set the seed for reproducibility
set.seed(123)

# Specify the response variable
response <- "went_on_backorder"

# Split the dataset into train and test sets
train_indices <- createDataPartition(dataset[, response], p = 0.7, list = FALSE)
train_data <- dataset[train_indices, ]
test_data <- dataset[-train_indices, ]

# Check the dimensions of the train and test sets
dim(train_data)
dim(test_data)

# Step 2: Specify the response and predictor variables
response_var <- "went_on_backorder"
predictor_vars <- setdiff(colnames(train_data), response_var)

# Step 3: Run AutoML specifying the stopping criterion
h2o.init()  
h2o.no_progress()  # Turn off progress bars for notebook readability

train_h2o <- as.h2o(train_data)  # Convert the training data to H2O frame
test_h2o <- as.h2o(test_data)    # Convert the test data to H2O frame

y = response_var
x = setdiff(names(train_h2o), response_var)

aml <- h2o.automl(y = y, x = x,
                  training_frame = dataset_h2o,
                  max_models = 10,
                  seed = 1)

# Step 4: View the leaderboard
my_lb <- aml@leaderboard
print(my_lb)

# Step 5: Predict using the Leader Model
leader <- aml@leader
predictions <- h2o.predict(leader, test_h2o)

typeof(predictions)

predictions_tbl <- predictions %>% as_tibble()
predictions_tbl
# Step 6: Save the Leader Model
h2o.saveModel(leader, path = "data")

# Shut down the H2O cluster
h2o.shutdown()
