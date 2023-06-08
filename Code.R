
# Step 1: Load the training & test dataset

data_path = ("data/product_backorders.csv")

train_data <- read.csv(data_path)
train_data

test_data <- read.csv(data_path)
test_data

# Step 2: Specify the response and predictor variables
response_var <- "went_on_backorder"
predictor_vars <- setdiff(colnames(train_data), response_var)

# Step 3: Run AutoML specifying the stopping criterion
library(h2o)
h2o.init()  

train_h2o <- as.h2o(train_data)  # Convert the training data to H2O frame
test_h2o <- as.h2o(test_data)    # Convert the test data to H2O frame

aml <- h2o.automl(
  x = predictor_vars,
  y = response_var,
  training_frame = train_h2o,
  stopping_metric = "AUC",  # Stopping criterion based on AUC
  max_runtime_secs = 3600  # Maximum allowed runtime in seconds
)

# Step 4: View the leaderboard
lb <- aml@leaderboard
print(lb)

# Step 5: Predict using the Leader Model
leader <- aml@leader
predictions <- h2o.predict(leader, test_h2o)

# Step 6: Save the Leader Model
h2o.saveModel(leader, path = "data")

# Shut down the H2O cluster
h2o.shutdown()
