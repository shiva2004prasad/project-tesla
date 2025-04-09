# project-tesla
Functionality
Import Libraries:

pandas: A powerful library for data analysis and manipulation.

os: Allows interaction with the operating system, used here to check if a file exists.

Defining the Function (read_csv_file):

Purpose: Reads a CSV file chunk by chunk (default chunk size: 100,000 rows).

Returns: A combined DataFrame containing all the data from the file.

Steps in the Function:
File Check:

The program checks if the file exists in the specified path (os.path.exists(file_path)).

If the file doesn't exist, it prints an error message (❌ File not found) and returns None.

Reading the CSV File:

Uses pd.read_csv with the chunksize argument to read large files in chunks, preventing memory overflow.

Each chunk is appended to the total_data DataFrame using pd.concat. ignore_index=True ensures consistent indexing.

Success Message:

Prints the total number of rows in the combined DataFrame after reading all chunks.

Error Handling:

Handles specific errors:

EmptyDataError: Occurs if the file is empty.

ParserError: Happens if the file format is invalid or has unexpected content.

Generic Exception Handling: Catches any other errors and prints the exception message.

Returns the combined DataFrame (total_data) or None if an error occurs.

Example Usage
The script includes an example to demonstrate how to use the function:

Define the File Path:

Specifies the file name (tesla_stock_data_2000_2025.csv).

Call the Function:

Reads the CSV file using the read_csv_file function and stores the output in the variable df.

Display Results:

If the function successfully reads the file (df is not None), it prints the first few rows of the dataset (df.head()).
1. Libraries Used
pandas: For data manipulation and analysis.

os: For file-related operations (like checking if the file exists).

2. Functionality
a. read_csv_file
Purpose: Reads a CSV file in chunks to handle large datasets efficiently.

Steps:

Checks if the file exists (os.path.exists).

Reads the file in chunks (pd.read_csv with chunksize).

Concatenates the chunks into a single DataFrame (pd.concat).

Output: Returns the combined DataFrame or None if errors occur.

Error Handling:

File not found.

Empty file.

Parsing errors or other unexpected issues.

b. preprocess_data
Purpose: Cleans and preprocesses the stock dataset.

Steps:

Identifies and handles the date column (Date or fallback to Price).

Converts the date column to datetime format (pd.to_datetime).

Removes rows with missing or invalid dates.

Ensures consistency by renaming the date column to Date.

Converts key numeric columns (Open, High, Low, Close, Volume) to numeric format.

Fills missing columns with NaN if they are not present in the dataset.

Removes rows with missing values in numeric columns.

Output: Returns a clean and preprocessed DataFrame.

c. calculate_daily_returns
Purpose: Calculates the percentage change in the Close column for daily returns.

Steps:

Adds a new column Daily Return by calculating the percentage change (.pct_change).

d. calculate_moving_average
Purpose: Calculates a moving average (default 20 days) for the Close column.

Steps:

Adds a new column (MA_20 by default) using the rolling mean (.rolling(window=...).mean()).

e. calculate_volatility
Purpose: Calculates rolling volatility (default 20 days) based on daily returns.

Steps:

Adds a new column (Volatility_20 by default) using the rolling standard deviation (.rolling(window=...).std()).

3. Main Script
Step 1: Read CSV

Reads the tesla_stock_data_2000_2025.csv file using the read_csv_file function.

Step 2: Preprocess

Cleans and preprocesses the data with preprocess_data.

Step 3: Calculations

Calculates:

Daily returns (calculate_daily_returns).

20-day moving averages (calculate_moving_average).

20-day rolling volatility (calculate_volatility).

Step 4: Display

Prints the last 5 rows of processed data.

Displays selected columns: Date, Close, Daily Return, MA_20, and Volatility_20.

4. Export
Saves the processed dataset to a new CSV file (processed_tesla_stock_data.csv).

5. Benefits
Memory Efficiency: Processes large files in chunks.

Data Cleaning: Ensures the dataset is clean and standardized for analysis.

Feature Engineering: Adds valuable metrics (Daily Return, Moving Average, Volatility).

Error Handling: Robust handling of common issues like missing files, empty columns, or parsing errors.

Flexibility: Can adapt to datasets with missing or differently named columns.
1. Libraries Used
pandas: For data manipulation and preprocessing.

numpy: For numerical operations (e.g., calculating RMSE).

sklearn: For machine learning tasks, including:

train_test_split: Splits the dataset into training and testing sets.

LinearRegression: Implements a linear regression model.

mean_squared_error and r2_score: Evaluate the model's performance.

2. Functionality
a. prepare_features
Purpose: Prepares the feature matrix (X) and target variable (y) for regression modeling.

Steps:

Copy and Sort Data:

Makes a copy of the input DataFrame (df) and sorts it by the Date column to ensure chronological order.

Create Target Variable (Next_Close):

Shifts the Close column one row down to use the next day's Close price as the target (df['Next_Close'] = df['Close'].shift(-1)).

Drop Rows:

Drops the last row (which will have a NaN in Next_Close due to the shift).

Feature Selection:

Extracts key features (Open, High, Low, Close, and Volume) into X.

Sets the target variable (Next_Close) as y.

b. train_model
Purpose: Splits the data, trains a linear regression model, evaluates its performance, and returns the trained model.

Steps:

Split Data:

Splits X and y into training and testing sets using an 80-20 split (train_test_split).

Train Model:

Initializes a linear regression model and trains it on the training data (model.fit(X_train, y_train)).

Make Predictions:

Predicts values for the test set (model.predict(X_test)).

Evaluate Performance:

Calculates:

RMSE (Root Mean Squared Error): Measures average error magnitude.

R² Score: Indicates how well the model explains variance in the target variable.

Prints the performance metrics.

Main Script
Load Data:

Reads the preprocessed stock data CSV file (processed_tesla_stock_data.csv) into a DataFrame (df).

Prepare Features:

Calls prepare_features to create the feature matrix (X) and target variable (y).

Train Model:

Trains and evaluates the regression model using train_model.

3. Output
Model Performance Metrics:

RMSE: Indicates the average prediction error in the target variable (lower is better).

R² Score: Represents the proportion of variance explained by the model (higher is better, with 1.0 being perfect).
Program Workflow:
Input Dataset: The program uses a CSV file (processed_tesla_stock_data.csv) containing Tesla stock data (like Date, Open, High, Low, Close, Volume).

Feature Preparation (prepare_features function):

Sort Data: The data is sorted by Date for sequential processing.

Create Target Variable: A new column, Next_Close, is added. It represents the closing price for the next trading day.

Remove NaN values: Any rows with missing Next_Close values (at the end of the dataset) are removed.

Select Features: The features used for prediction include Open, High, Low, Close, and Volume.

Return: The features (X) and the target variable (y) are returned for model training.

Model Training and Evaluation (train_model function):

Split Data: The dataset is split into training (80%) and testing (20%) sets using train_test_split.

Train Linear Regression Model: A linear regression model is fitted on the training data (X_train, y_train).

Make Predictions: Predictions (y_pred) are generated for the test set (X_test).

Evaluate Model:

RMSE (Root Mean Squared Error): Measures how close predictions are to actual values.

R² (R-squared): Shows how well the model explains the variation in the data.

Visualization of Results (plot_results function):

Actual vs Predicted Scatter Plot: Displays the relationship between the true closing prices (y_test) and the predicted ones (y_pred).

Residual Distribution Plot: Shows the errors (difference between actual and predicted values) to assess model performance.

Display Predictions (show_prediction_table function):

A table of actual vs predicted values is displayed to understand how well the model is performing on the test set.

Execution (if __name__ == "__main__" block):

The CSV file is read into a DataFrame (df).

The program sequentially:

Prepares the data.

Trains the model.

Displays a table of predictions.

Plots the evaluation visuals.
Interactive CLI Menu:

The program presents users with options to load data, preprocess it, train a model, visualize results, and exit.

It ensures that tasks are executed in the correct order by performing checks (e.g., making sure data is loaded before preprocessing).

Chunk-Based Data Loading:

Efficient Data Handling: It reads large CSV files in chunks of 100,000 rows using the read_csv_file function. This prevents memory overload for large datasets.

Data Preprocessing:

Cleans and prepares the input data for machine learning:

Converts the Price column to datetime format and renames it to Date.

Ensures numeric columns (Open, High, Low, Close, Volume) are properly cast as numeric data types.

Drops rows with missing or invalid values.

Feature Engineering:

Prepares the features (Open, High, Low, Close, Volume) and target variable (Next_Close).

The target (Next_Close) is derived by shifting the Close column by one day to represent the next trading day's closing price.

Linear Regression Model:

Trains the model using the scikit-learn library.

Splits data into training and test sets (80% training, 20% testing).

Evaluates performance using metrics:

RMSE (Root Mean Squared Error): Indicates prediction error.

R² (R-squared): Measures how well the model explains the variance in the data.

Visualization:

Generates plots to evaluate model performance:

Scatterplot for Actual vs. Predicted Values.

Histogram of Residuals (Prediction Errors).

Prediction Table:

Displays a table showing a sample of Actual vs. Predicted values.

Functions in Detail:
read_csv_file(file_path, chunksize):

Efficiently loads data from a CSV file in chunks.

Returns a combined DataFrame.

preprocess_data(df):

Handles missing/invalid data.

Converts columns to appropriate data types (e.g., datetime, numeric).

Prepares the DataFrame for feature engineering.

prepare_features(df):

Adds a Next_Close column (the target variable).

Selects relevant features (Open, High, Low, Close, Volume).

Returns the feature matrix (X), target vector (y), and processed DataFrame.

train_model(X, y):

Splits data into training and test sets.

Trains a linear regression model.

Evaluates and prints performance metrics (RMSE, R²).

show_prediction_table(y_test, y_pred):

Displays a preview of actual vs predicted closing prices in tabular format.

plot_results(y_test, y_pred):

Creates visualizations:

Scatterplot comparing actual and predicted values.

Histogram of residuals (prediction errors).

main():

Implements the CLI with options:

1: Load the dataset.

2: Preprocess the data.

3: Train the model.

4: Show predictions in a table.

5: Plot the results.

0: Exit the program.

Ensures steps are executed in order by validating prerequisites (e.g., data must be loaded before preprocessing).

Flow of Execution:
Step 1: Start the program, load data using the CLI option.

Step 2: Preprocess the data (cleaning and preparing features/target).

Step 3: Train the linear regression model and evaluate its performance.

Step 4: View predictions in a table format.

Step 5: Visualize results (actual vs predicted values, residuals).

Step 6: Exit the program when done.
Purpose of the Program:
The program is a Tesla Stock Analysis CLI Application that allows users to:

Load Tesla stock data from a CSV file.

Preprocess and clean the data.

Train a Linear Regression model to predict the next day's closing price based on past stock data.

Evaluate the model's performance using metrics and visualizations.

View predictions in a tabular format and visualize results.

New Features in This Version:
User Guide:

At the beginning, the program shows a user guide with installation instructions and tips for working with the CLI.

Provides clarity about the data requirements (columns like Date, Open, High, Low, Close, and Volume).

Error Handling:

Improved handling of missing or misnamed columns in the CSV file.

Adds custom messages and exceptions for preprocessing failures (e.g., missing Date column).

Command-Line Interaction:

A detailed menu guides users through the workflow step-by-step:

Option 1: Load the dataset (reads large files in chunks for efficiency).

Option 2: Preprocess and clean the data (converts formats, removes invalid rows).

Option 3: Train the model and evaluate metrics like RMSE and R².

Option 4: Display a table of actual vs. predicted values.

Option 5: Generate and display visualizations of model performance.

Option 0: Exit the program.

Enhanced Functionality:

Handles missing column names or irregular column naming conventions (e.g., Price Date instead of Date).

Provides step-by-step validation to ensure that users follow the correct sequence of actions (e.g., loading data before preprocessing).

Key Functions:
show_user_guide():

Displays usage instructions and program requirements.

Includes tips for successful execution and data preparation.

read_csv_file(file_path, chunksize):

Reads CSV data in chunks and returns a combined DataFrame.

Handles cases where the file does not exist and provides feedback.

preprocess_data(df):

Cleans and prepares the stock data:

Converts Price Date or similar columns to a proper Date format.

Converts numeric columns (Open, High, Low, Close, Volume) to numeric types, removing invalid data.

Handles missing or misnamed columns by raising exceptions.

prepare_features(df):

Creates a new column, Next_Close, representing the target variable (next day's closing price).

Drops rows with missing values and extracts features (X) and labels (y).

train_model(X, y):

Splits the data into training and test sets.

Trains a Linear Regression model and evaluates it using:

RMSE (Root Mean Squared Error): Measures prediction error.

R² (R-squared): Indicates how well the model explains data variance.

Returns the trained model and predictions.

show_prediction_table(y_test, y_pred):

Displays a tabular comparison of actual vs. predicted closing prices.

plot_results(y_test, y_pred):

Generates two plots:

Scatterplot: Shows actual vs. predicted values.

Histogram: Displays the residuals (errors) of predictions.

main():

Implements the interactive CLI menu.

Validates user input and ensures tasks are completed in order.

Provides meaningful feedback for invalid choices or missing steps.
