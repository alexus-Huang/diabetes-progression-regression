# Diabetes Progression Prediction

A multiple linear regression project predicting diabetes progression one year after baseline, 
using scikit-learn's built-in diabetes dataset. This project focuses on diagnosing multicollinearity 
between correlated medical features and evaluating a model on data that arrives pre-standardized.

## Dataset
[scikit-learn Diabetes Dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset) 
— 442 patients, 10 baseline variables (age, sex, BMI, blood pressure, and six blood serum 
measurements), predicting a quantitative measure of disease progression. Loaded directly via 
`sklearn.datasets.load_diabetes()`.

Note: unlike the other projects in this series, this dataset arrives already standardized 
(mean ≈ 0, std ≈ 1 for every feature) — there is no separate scaling step required.

## What's covered
- Exploratory data analysis across all 10 features
- Multiple linear regression using scikit-learn
- Diagnosing multicollinearity between correlated features (s1/total cholesterol and 
  s2/LDL) using a correlation matrix
- Model evaluation (MSE, MAE, RMSE) compared against the target's standard deviation
- Actual vs. predicted visualization
- Predictions on new, hypothetical patient input

## Key finding
BMI and s5 (log of serum triglycerides) were the strongest predictors of disease progression, 
consistent with their clear upward trends in EDA. However, s1 (total cholesterol) received a 
surprisingly large coefficient despite showing no visible correlation with the target on its 
own — this was explained by a strong correlation (0.897) between s1 and s2 (LDL), a case of 
multicollinearity where the model struggles to separate the individual effects of two closely 
related features.

Final validation RMSE was 55.75, compared to a target standard deviation of 77.01 — indicating 
the model captures real signal beyond simply predicting the average.

## Setup
\`\`\`bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
\`\`\`

Open `notebooks/diabetes_regression.ipynb` in Jupyter or VS Code.

## What I learned
- Some datasets arrive pre-processed (scaled/standardized), which changes how you interpret 
  both the raw values and the resulting coefficients, a coefficient's meaning depends on 
  the units its feature is measured in.
- Multicollinearity can distort individual coefficients even when the model's overall 
  predictions remain reasonable, checking a correlation matrix between features (not just 
  against the target) is an important diagnostic step.
- Comparing validation RMSE against the target's standard deviation is a quick, useful sanity 
  check for whether a model is learning real patterns or just predicting the average.