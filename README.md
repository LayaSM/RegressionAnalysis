# House Prices Regression Analysis

## **Project Overview**
This project aims to predict house prices based on various features using regression models. By preprocessing data, visualizing trends, and implementing machine learning models, we strive to build an accurate and interpretable predictive framework.

## **Dataset**
- **House Prices Dataset**: [Kaggle Competition Link](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data?select=train.csv)

## **Steps in the Project**

### **1. Understanding the Dataset**
- Loaded the dataset and examined its structure using shape, data types, and summary statistics.
- Identified categorical and continuous columns to tailor preprocessing steps accordingly.

### **2. Data Preprocessing**
- **Null Values**: Handled missing data by dropping columns with excessive null values and filling others with appropriate defaults.
- **Categorical Features**:
  - Extracted categorical columns and created a separate dataframe for analysis.
  - Applied Label Encoding to convert categorical values to numeric representations.
- **Continuous Features**:
  - Filtered continuous columns and filled missing values with zeros.
  - Checked for remaining nulls to ensure data integrity.

### **3. Data Visualization**
- **Correlation Heatmap**: Visualized correlations among numerical features to identify influential variables.
- **Feature Relationships**: Created scatter plots and regression lines to examine key trends.

### **4. Modeling**
#### **Linear Regression**
- Split data into training and testing sets.
- Trained a Linear Regression model on the dataset and evaluated it using the Coefficient of Determination ($R^2$) metric.
- Performed iterative training to obtain average model performance over 1,000 iterations.

#### **Experiment 2**
- Incorporated encoded categorical variables into the dataset for enhanced prediction.
- Evaluated the updated Linear Regression model using the same methodology.

#### **Experiment 3**
- Implemented a Random Forest Regressor for a robust non-linear model.
- Evaluated the model's performance using $R^2$, Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE).

### **5. Model Evaluation**
- Compared Linear Regression and Random Forest models on key metrics:
  - **MAE**: Measures the average magnitude of errors.
  - **MSE and RMSE**: Indicate model precision and penalty for larger errors.
  - **$R^2$**: Demonstrates the proportion of variance explained by the model.

## **Dependencies**
- **Python Libraries**:
  - numpy
  - pandas
  - seaborn
  - matplotlib
  - scikit-learn

## **How to Run**
1. Download the dataset from Kaggle and place it in the project directory.
2. Install the required libraries:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn
   ```
3. Run the Python script to preprocess data, visualize trends, and build regression models.
4. Review the output metrics and plots to gain insights into model performance.

## **Key Insights**
- The correlation heatmap highlights relationships among numerical features and the target variable.
- Linear Regression provides a baseline for performance comparison.
- Random Forest Regressor outperforms Linear Regression in capturing non-linear relationships.

## **Future Scope**
- Incorporate feature engineering to enhance prediction accuracy.
- Experiment with advanced models like Gradient Boosting and XGBoost.
- Perform hyperparameter tuning to optimize models further.

---
This project serves as a foundational exercise in regression analysis, showcasing the power of data science and machine learning in predictive modeling for real-world applications.

