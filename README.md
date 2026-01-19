# Impact of Digitalization ML

## Overview
This project analyzes survey data from small businesses in Bengal to predict digital tool adoption intent using machine learning. The analysis examines how digital adoption correlates with business characteristics and identifies key factors influencing the adoption of digital tools like UPI payments and online marketplaces.

## Project Structure
- `bengalSurvey.csv` - Survey data from small business owners covering digital adoption practices and future intentions
- `digital_adoption_analysis.ipynb` - Jupyter notebook containing the complete analysis pipeline
- `README.md` - Project documentation

## Dataset
The survey dataset contains responses from small business owners with the following features:
- **Metadata**: Timestamp, Location, Business Name
- **Likert Scale Questions** (1-5 scale): Statements about digital tool attitudes and experiences
- **Binary Questions** (Yes/No):
  - Shop accepts UPI/mobile payments
  - Use of online marketplaces for sales
- **Target Variable**: "I plan to use more digital tools in the next 12 months" (converted to binary: ≥4 = Plans to Adopt, <4 = No Plans)

## Analysis Pipeline

### 1. Data Preprocessing
- Load survey data from CSV
- Convert Likert scale responses (Strongly Agree, Agree, Neutral, Disagree, Strongly Disagree) to numeric values (5-1)
- Convert binary Yes/No responses to numeric (1/0)
- Handle missing values with sensible defaults

### 2. Exploratory Data Analysis
- Display dataset structure and first few rows
- Compute and visualize correlation matrix between features
- Identify relationships between variables using heatmap

### 3. Feature Engineering
- Remove metadata columns (Timestamp, Location, Business Name)
- Create binary target variable based on adoption intent threshold (score ≥4)
- Prepare feature matrix (X) and target vector (y)

### 4. Model Training & Evaluation
Train and compare multiple machine learning models:
- **Logistic Regression**: Linear classification baseline
- **Random Forest**: Ensemble method for feature importance analysis
- **Gradient Boosting**: Sequential ensemble for improved performance
- **Support Vector Machine (SVM)**: Non-linear classification

Evaluation metrics:
- Accuracy: Overall correctness
- Precision: True positives among predicted positives
- Recall: True positives among actual positives
- F1-Score: Harmonic mean of precision and recall

### 5. Feature Importance Analysis
Extract and visualize feature importances from the Random Forest model to identify which factors most strongly predict digital adoption intent.

## Key Findings
The analysis reveals a significant class imbalance in the target variable - the majority of survey respondents are not planning to adopt more digital tools in the next 12 months. This imbalance affects model performance and indicates the need for techniques such as:
- Class weights adjustment
- Stratified sampling
- SMOTE (Synthetic Minority Over-sampling Technique)
- Modified evaluation metrics for imbalanced datasets

## Technologies Used
- **Python 3.x**
- **pandas**: Data manipulation and analysis
- **scikit-learn**: Machine learning models and evaluation metrics
- **numpy**: Numerical computations
- **matplotlib & seaborn**: Data visualization

## Requirements
Install dependencies using:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

## Usage
1. Ensure `bengalSurvey.csv` is in the project directory
2. Open `digital_adoption_analysis.ipynb` in Jupyter Notebook
3. Run cells sequentially to execute the complete analysis pipeline
4. Review correlation heatmap and feature importances to understand key drivers of adoption

## Future Improvements
- Address class imbalance using techniques like SMOTE or class weights
- Perform hyperparameter tuning for optimal model performance
- Implement cross-validation for more robust performance estimates
- Create separate models for different business segments
- Add business impact analysis based on adoption predictions
- Conduct statistical significance testing

## Notes
- The current models show high accuracy but this is largely due to majority class bias
- Focus should be on recall/precision for the minority class (adoption intent = 1)
- Survey sample size and representativeness should be considered when generalizing findings
