# OIBSIP_DataScience_task5
OIBSIP Internship – Data Science Task 5:Sales Prediction using Python
# OIBSIP Machine Learning Task 5: Sales Prediction with Advertising Data

# Objective
This task is part of the Oasis Infobyte Internship Program (OIBSIP).  
The goal of Task 5 is to Sales Prediction using Python: Sales prediction means predicting how much of a product people will buy based on factors
such as the amount you spend to advertise your product, the segment of people you
advertise for, or the platform you are advertising on about your product.


Typically, a product and service-based business always need their Data Scientist to predict
their future sales with every step they take to manipulate the cost of advertising their
product. So let’s start the task of sales prediction with machine learning using Python 
The project demonstrates data preprocessing, exploratory data analysis, model training, evaluation, hyperparameter tuning, and prediction on new data.

#Steps Performed
1. **Data Loading** – Imported dataset `Advertising.csv`.  
2. **Data Cleaning** – Checked missing values, duplicates, and unnecessary columns.  
3. **Exploratory Data Analysis (EDA)** – Generated summary statistics and visualizations:
   - Feature distributions  
   - Sales distribution  
   - Scatter plots (TV/Radio/Newspaper vs Sales)  
   - Correlation heatmap  
4. **Data Preparation** – Defined features and target, split into train/test, and scaled features where necessary.  
5. **Model Training** – Trained multiple models:
   - Linear Regression, Ridge, Lasso  
   - Random Forest, Gradient Boosting  
   - Support Vector Regression (SVR)  
6. **Model Evaluation** – Compared models using R², RMSE, MAE, CV scores.  
7. **Hyperparameter Tuning** – Used GridSearchCV for best model optimization.  
8. **Prediction on New Data** – Predicted sales for sample advertising budgets.  
9. **Comprehensive Report** – Summarized results and insights.  

# Tools Used
- **Python 3.x**  
- **Libraries:** Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn  
- **Dataset:** *Advertising.csv* (classic dataset)  
- **Editor:** VS Code / Jupyter Notebook  

#Outcome
- Developed a full ML pipeline for sales prediction.  
- Identified **TV advertising** as the most impactful feature.  
- Achieved strong model performance with **Random Forest / Gradient Boosting**.  
- Built reusable script to generate predictions for any ad budget scenario.  

# Key Visualizations
- Distribution of TV, Radio, Newspaper spend.  
- Sales distribution.  
- Correlation heatmap.  
- Actual vs Predicted Sales (best model).  
- Residual plots & feature importance.  
- Model comparison bar charts (R², RMSE).  

# How to Run
1. Place dataset inside `datasets/` folder:
2. Run the script:
```bash
python task_5.py
