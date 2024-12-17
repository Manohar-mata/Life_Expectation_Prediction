# Report: Life Expectancy Prediction

## 1. Introduction

This report documents the analysis and modeling of life expectancy data using Ordinary Least Squares (OLS) regression and Logit regression. The goal is to predict life expectancy and categorize it as 'High' or 'Low' based on various factors.  The analysis was performed in a Google Colab environment.

## 2. Methodology

**2.1 Data Loading and Preprocessing:**

* The dataset, 'updated_life_expectancy_data.csv', was loaded into a Pandas DataFrame.

* Missing value handling: Rows with missing values were removed.  A more robust approach would involve imputation techniques in a real-world scenario.  

* Data Transformation:  Float values in the 'percentage expenditure' column were converted to absolute values and then to integers.

* Zero Value Handling: Zeros in specific columns (identified in the code) were replaced with the median of the respective column, excluding zero values. This mitigates the effect of outliers. Found the variables which are skewed and normally distributed by using Shapiro-Wilk test for normality and set the thresholds to 0.05 for normally distributed data and 0.5 for skewed data.

* Feature Engineering: Two new binary features, 'Status_Developed' and 'Status_Developing', were created from the 'Status' column using lambda functions.  The original 'Status' column was subsequently dropped.

* Data Splitting: The dataset was split into training and testing sets (80/20 split) with a random state of 42, ensuring reproducibility.

**2.2 Exploratory Data Analysis (EDA):**

* **Univariate Analysis:** Histograms, boxplots, and bar charts were used to visualize the distributions of numerical and categorical variables, respectively.  The code included loops to iterate through the relevant columns for visualization.

* **Bivariate Analysis:**  The `create_high_correlation_pivot_table` function was used to identify and display pairs of highly correlated numerical variables (correlation coefficient >= 0.7).  A heatmap visualization highlighted these correlations.  This helps in identifying potential multicollinearity issues. Conducted pairplots to visualize the relationships between numerical variables.

* **Normality Tests:** Shapiro-Wilk tests and skewness measures were used to assess normality for the specified columns. Histograms and Q-Q plots were generated to provide visual confirmation of the normality test results.


**2.3 Model Building:**

* **OLS Regression:** An OLS regression model was fitted using the `statsmodels` library. A constant was added to the independent variables to account for the intercept term.

* **Logit Regression:** A Logit regression model was built using the `statsmodels` library. Predictions were generated, and a threshold of 0.5 was used to classify life expectancy into 'High' or 'Low' categories.


**2.4 Model Evaluation:**

* **OLS Regression:** The model was evaluated using RMSE and R-squared, which quantify model fit and variance explained, respectively.

* **Logit Regression:**  Evaluation metrics included: accuracy, confusion matrix, classification report (precision, recall, F1-score), and the Receiver Operating Characteristic (ROC) curve with its Area Under the Curve (AUC).


## 3. Findings

**3.1 OLS Regression:**

* The OLS models performance can be assessed using the reported RMSE and R-squared values.

                       OLS Regression Results                            
==============================================================================

Dep. Variable:       Life expectancy    R-squared:                       0.839

Model:                            OLS   Adj. R-squared:                  0.837

Method:                 Least Squares   F-statistic:                     424.0

Date:                Tue, 17 Dec 2024   Prob (F-statistic):               0.00

Time:                        01:32:16   Log-Likelihood:                -4419.4

No. Observations:                1649   AIC:                             8881.

Df Residuals:                    1628   BIC:                             8994.

Df Model:                          20                                         

Covariance Type:            nonrobust                                         

=================================================================================================

* The model summary shows the significance of various predictors. Like coefficient interpretations and influences on life expectancy from the summary.


                                      coef    std err          t      P>|t|      [0.025      0.975]

---------------------------------------------------------------------------------------------------

const                             207.9528     30.752      6.762      0.000     147.635     268.271

Year                               -0.1288      0.023     -5.592      0.000      -0.174      -0.084

Adult Mortality                    -0.0161      0.001    -17.108      0.000      -0.018      -0.014

infant deaths                       0.0903      0.010      8.622      0.000       0.070       0.111

Alcohol                            -0.1362      0.033     -4.069      0.000      -0.202      -0.071

percentage expenditure              0.0003      0.000      1.753      0.080   -3.73e-05       0.001

Hepatitis B                        -0.0034      0.004     -0.756      0.450      -0.012       0.005

Measles                         -1.024e-05   1.07e-05     -0.960      0.337   -3.12e-05    1.07e-05

 BMI                                0.0316      0.006      5.318      0.000       0.020       0.043

under-five deaths                  -0.0677      0.008     -8.922      0.000      -0.083      -0.053

Polio                               0.0055      0.005      1.071      0.284      -0.005       0.016

**3.2 Logit Regression:**

* Model accuracy: 0.8629472407519709

* Confusion matrix: (include confusion matrix from output).  Analyze the number of true positives, true negatives, false positives, and false negatives. 
the confusion matrix breakdown:

True Positives (TP) = 690

The model correctly predicted 'High' life expectancy.
True Negatives (TN) = 733

The model correctly predicted 'Low' life expectancy.
False Positives (FP) = 133

The model incorrectly predicted 'High' life expectancy when it was actually 'Low'.
False Negatives (FN) = 93

The model incorrectly predicted 'Low' life expectancy when it was actually 'High'.

* Classification report: 

 precision    recall  f1-score   support

           0       0.89      0.85      0.87       866

           1       0.84      0.88      0.86       783

    accuracy                           0.86      1649

   macro avg       0.86      0.86      0.86      1649

weighted avg       0.86      0.86      0.86      1649

* ROC AUC: [0.83] - The ROC curve shows the trade-off between the True Positive Rate (TPR) and False Positive Rate (FPR) for your logistic regression model. The Area Under the Curve (AUC = 0.83) indicates strong model performance, significantly better than random guessing. The curve bending towards the top-left corner reflects the model's good ability to distinguish between 'High' and 'Low' life expectancy.  An AUC of 0.5 suggests no discrimination; higher values indicate better discrimination.

* Significant features: 


 Logit Regression Results                              

====================================================================================

Dep. Variable:     Life_expectancy_category   No. Observations:                 1649

Model:                                Logit   Df Residuals:                     1628

Method:                                 MLE   Df Model:                           20

Date:                      Tue, 17 Dec 2024   Pseudo R-squ.:                  0.5752

Time:                              01:34:16   Log-Likelihood:                -484.61

converged:                             True   LL-Null:                       -1140.9

Covariance Type:                  nonrobust   LLR p-value:                5.952e-266


                                      coef    std err          z      P>|z|      [0.025      0.975]


const                              25.1104   3.74e+07   6.72e-07      1.000   -7.33e+07    7.33e+07

Year                               -0.0221      0.021     -1.036      0.300      -0.064       0.020

Adult Mortality                    -0.0127      0.001     -9.880      0.000      -0.015      -0.010

infant deaths                       0.0517      0.043      1.201      0.230      -0.033       0.136

Alcohol                            -0.0661      0.033     -2.020      0.043      -0.130      -0.002

percentage expenditure              0.0014      0.000      3.435      0.001       0.001       0.002

Hepatitis B                        -0.0035      0.004     -0.781      0.435      -0.012       0.005

Measles                          3.563e-05   1.62e-05      2.204      0.028    3.94e-06    6.73e-05

BMI                               -0.0058      0.006     -1.025      0.305      -0.017       0.005

under-five deaths                  -0.0482      0.036     -1.352      0.177      -0.118       0.022




## 4. Insights

 OLS Regression Insights:
 
 Based on the OLS regression results, several variables significantly influence life expectancy.
 For instance, a negative coefficient for 'Adult Mortality' suggests that higher adult mortality rates are associated with lower life expectancy, which aligns with expectations.  Similarly, 'under-five deaths' shows a strong negative correlation.  The positive coefficient of 'percentage expenditure' on health indicates that increased spending on healthcare corresponds to a higher life expectancy.  Interpret other significant variables in a similar fashion.  The R-squared value suggests the model explains a substantial portion of the variance in life expectancy. The RMSE indicates the average difference between predicted and actual life expectancies.  A lower RMSE indicates a better fit.

 Logit Regression Insights:
 The Logit model effectively classifies life expectancy into 'High' and 'Low' categories. The accuracy score and the AUC of the ROC curve provide strong indications of the model's performance.  Examine the confusion matrix to understand the model's strengths and weaknesses in making specific predictions. The classification report provides more insight into precision, recall, and F1-scores for both classes.  Analyze the significant predictors as we did in OLS; however, the coefficients are log-odds, requiring interpretation within a logistic context.  For example, a positive coefficient means that an increase in the predictor is associated with a higher *probability* of a high life expectancy.  Noteworthy is the fact that the Logit model does not explicitly model the magnitude of life expectancy but categorizes it.

 Comparing Models:
 The OLS model provides continuous predictions of life expectancy, enabling a more granular understanding of the magnitude of the effect of different features.  The Logit model offers a binary classification, which may be suitable when the primary goal is to determine the category ('High' or 'Low') rather than the precise numerical value of life expectancy.  Depending on the specific application, one model might be preferable over the other.

 Significant Variable Effects:
 Compare the effects of significant variables across both models. For example, 'Adult Mortality' negatively impacts both models. Does it have a stronger effect in one model over the other?  This comparison can reveal nuanced aspects of the variables' influence on different aspects of life expectancy prediction (numeric vs. binary).  

 Overall:
 The combination of these models provides a comprehensive picture of the factors affecting life expectancy. The OLS model gives a measure of magnitude whereas the logit model provides probabilities. The visualizations and evaluation metrics help quantify the model performance.  Further analysis using domain knowledge and context could help explain the observed relationships and improve model predictions further.


## 5. Limitations

* Missing data handling: Removing rows with missing values may introduce bias.  Imputation methods should be considered.

* Data Transformations:  The chosen transformation for zero values (replacing them with the median) might not be optimal in all cases and different treatment may be required.

* Multicollinearity: The presence of highly correlated features may influence model coefficients.


## 6. Further Work

* Investigate more advanced imputation methods.
eling techniques.

* Implement more robust feature selection methods to improve model parsimony.

* Use cross-validation to better assess model generalization.
