# Wine-Quality-Prediction-models
Machine learning project using the UCI Wine Quality dataset to predict wine quality scores through classification and regression models.


CS 4120 – MACHINE LEARNING, DATA MINING PROJECT PROPOSAL

AUTHORS:
•	CHIZURUM EWELIKE
•	LAYO FALOSEYI

WINE QUALITY

Section 1.0: Overview

Expert tasters often judge wine quality, but their evaluations can be subjective and time-consuming. By building machine learning models, we can create a reproducible, data-driven way to evaluate wine quality, helping wine producers maintain consistency, supporting automated quality control, and benefiting consumers who want reliable indicators of wine quality.

Section 2.0: Dataset Description
This project uses the Wine Quality Dataset from the UCI Machine Learning Repository (Cortez et al., 2009), released under a Creative Commons license. The Wine Quality dataset consists of 6,497 Portuguese “Vinho Verde” wine samples, including 1,599 red and 4,898 white wines, each described by 11 physicochemical input variables such as acidity, residual sugar, pH, sulphates, and alcohol. The target variable is a sensory quality score between 0 and 10, rated by wine experts. No missing values are reported, and there are no sensitive attributes since the data is purely chemical and sensory.
Section 3.0: Tasks
This project will address a classification and a regression problem using the Wine Quality dataset.
•	Classification Task: The wine quality score (0–10) will be converted into a binary label: High Quality (≥7) vs. Low Quality (<7). This derivation follows standard practice in prior studies, balancing interpretability with the class distribution.
•	Regression Task: The original quality score (0–10) will be predicted directly as a continuous variable, based on expert sensory ratings.
Both tasks are feasible because the dataset provides a numerical target that can be framed as a categorical outcome or a continuous score. This allows us to explore and compare supervised learning approaches on the same dataset. We frame this as <P, T, E>: Performance (P) = Accuracy/F1 & MAE/RMSE, Task (T) = predict label and score, Experience (E) = the wine physicochemical measurements.
Section 4.0: Metrics Plan
We will use standard metrics for classification and regression tasks to evaluate model performance. We will use a stratified train/validation/test split with feature scaling for models that require it and k-fold cross-validation for model selection; final metrics will be reported on a held-out test set.
•	Classification: Models will be assessed using Accuracy (overall proportion of correct predictions) and F1-score (harmonic mean of precision and recall, which is helpful for imbalanced classes). We will primarily use Accuracy and F1, since the classes are imbalanced. ROC-AUC may also be reported as a supplementary metric.
•	Regression: Models will be evaluated using Mean Absolute Error (MAE), which captures average prediction error in the same units as the target score, and Root Mean Squared Error (RMSE), which penalizes larger errors more heavily.

Section 5.0: Baseline Plan (Classical ML Models)
We will implement at least two baseline models for each task from the required set of classical machine learning methods.
•	Classification: We will use Logistic Regression and a Decision Tree Classifier. Logistic Regression provides a strong linear baseline, while Decision Trees can capture nonlinear relationships and feature interactions.
•	Regression: We will use Linear Regression and a Decision Tree Regressor. Linear Regression offers an interpretable baseline, and Decision Trees provide flexibility in modelling complex variable effects.
These baselines will be trained and evaluated at the Midpoint stage, as the foundation for comparing neural network models in the Final report. We will implement a small multilayer perceptron (2 hidden layers, 32–64 units, ReLU) with Adam optimizer and early stopping on validation loss. Inputs will be standardized.
Section 6.0: Reproducibility Plan
We will use a GitHub repository with a clear folder structure and a requirements.txt file that pins exact dependency versions to ensure reproducibility. In addition, we will use MLflow tracking to log model parameters, metrics, and outputs across experiments. This setup will allow our results to be consistently reproduced from the repository.


Section 7.0: Tables
Table 1 – Dataset Snapshot 
Rows	Columns	Target Descriptions	% Missing (top 5)	Class distribution
6,497 total (1,599 red + 4,898 white)

	12 (11 features + 1 target)	Classification: High (≥7) vs Low (<7) Quality; Regression: Quality Score (0–10)	0% across all columns	Low (<7): 80.3%; High (≥7): 19.7%


Table 2 – Planned Models and Metrics
Task	Baseline models	Metrics (Midpoint)	Metrics (Final)
Classification	Logistic Regression; Decision Tree	Accuracy; F1	Accuracy; F1
Regression	Linear Regression; Decision Tree Regressor	MAE; RMSE	MAE; RMSE







Section 8.0: References
•	Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. (2009). Modelling wine preferences by data mining from physicochemical properties. Decision Support Systems, 47(4), 547–553. https://www.sciencedirect.com/

•	UCI Machine Learning Repository. (2009). Wine Quality Data Set. University of California, Irvine. http://archive.ics.uci.edu/ml/datasets/Wine+Quality

•	GitHub Repository: https://github.com/LAYO200/Wine-Quality-Prediction-models.git

## MIDPOINT REPORT
Section 1.0: Overview
Expert tasters often judge wine quality, but their evaluations can be subjective and time-consuming. We build machine learning models to provide a reproducible, data-driven evaluation of wine quality, aiming to support consistent production decisions and offer reliable indicators for consumers.
Section 2.0: Dataset Description
We use the Wine Quality dataset from the UCI Machine Learning Repository (Cortez et al., 2009), containing 6,497 Portuguese Vinho Verde wine samples (1,599 red; 4,898 white). Each sample has 11 physicochemical features (e.g., acidity, residual sugar, pH, sulphates, alcohol) and a target quality score in the range of [0, 10] based on expert ratings.
Cleaning notes applied in code:
•	Concatenated red and white subsets; added a categorical type column.
•	Coerced all feature columns to numeric; dropped rows that became NaN after coercion.
•	Cast quality to int; ensured features are float64 for stable MLflow schema.
•	Derived binary label target_cls = 1 if quality ≥ 7, else 0.
Section 3.0: Tasks
•	Classification: Predict High Quality (≥7) vs. Low Quality (<7).
•	Regression: Predict the original quality score (0–10).
This dual-task design enables a side-by-side comparison of linear and nonlinear approaches on the same dataset.
Section 4.0: Metrics Plan
•	Classification metrics: Accuracy and F1 score (to account for class imbalance). ROC-AUC may be reported as supplementary.
•	Regression metrics: Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).
Model selection change: Our proposal planned a k-fold CV. For the midpoint, we switched to a fixed stratified train/validation/test split (seed = 42) to simplify iteration and runtime. We select the best classification model based on validation F1 and the best regression model based on validation RMSE, and we report the test metrics once. We will consider reinstating k-fold and small hyperparameter grids for the final.
Section 5.0: Baseline Plan (Classical ML Models)
•	Classification baselines: Logistic Regression (with standardization) and Decision Tree Classifier (no scaling).
•	Regression baselines: Linear Regression (with standardization) and Decision Tree Regressor (no scaling).
Preprocessing is managed with scikit-learn Pipelines and a ColumnTransformer for scaling numeric features. We track parameters, metrics, figures, and serialized models in MLflow (with an input example for schema stability).

Section 6.0: Exploratory Data Analysis (EDA)
 PLOT1
Interpretation: The distribution is imbalanced (≈ 81% class 0 vs 19% class 1). Due to this skew, accuracy alone can be misleading; therefore, we report F1 to reflect minority-class performance and use stratified splits.
 PLOT2
Interpretation: Alcohol and sulphates show positive correlation with quality; volatile acidity tends to be negatively related. Several features are moderately correlated with each other (e.g., total sulphur dioxide and free sulphur dioxide), which motivates the use of regularized/linear baselines and tree-based models that handle interactions.
Section 7.0: Results
Table 1 — Classification metrics (Val Accuracy/F1; Test Accuracy/F1)
TABLE 1 – CLASSIFICATION METRICS (VAL/TEST)
Model	Val_Accuracy	Val_F1	Test_Accuracy	Test_F1
Logistic Regression	0.830827 	0.430380       	0.813910 	0.369427
Decision Tree Classifier	0.793233 	0.471154      	0.789474 	0.469194
Interpretation: Decision Tree Classifier achieves a higher F1 (Val 0.471, Test 0.469) than Logistic Regression (Val 0.430, Test 0.369), so it’s the best classifier by the primary metric under imbalance. Logistic regression has higher accuracy, but the performance of the minority class is weaker; hence, the preference for F1.
PLOT3
Interpretation: With TP=99, FN=103, FP=121, TN=741, errors cluster near the 6/7 threshold. False negatives (missed high-quality wines) are common when alcohol levels are moderate, but acidity/sulphates are favourable. False positives often have high alcohol but an unfavourable acidity profile—alcohol alone is not decisive. Precision and recall for class 1 are balanced (≈0.45–0.49), resulting in an F1 ≈ score of approximately 0.47.
Table 2 — Regression metrics (Val MAE/RMSE; Test MAE/RMSE)
Table 2 – Regression Metrics (Val/Test)
Model	Val_Mae	Val_Rsme	Test_Mae	Test_Rmse
LinearRegression	0.573437  	0.735946  	0.546002   	0.701434
Decision Tree Regressor	0.657895  	0.959715  	0.662594   	0.960204
Interpretation: Linear Regression is best (Val MAE 0.573 / RMSE 0.736; Test MAE 0.546 / RMSE 0.701) vs. Decision Tree Regressor (Val 0.658 / 0.960; Test 0.663 / 0.960), indicating the target follows largely linear/monotonic trends that the linear model captures well.
PLOT4
Interpretation: Residuals exhibit a slight downward trend (under-prediction at low predicted scores and over-prediction at high), accompanied by mild heteroskedasticity (a wider spread at higher predicted values). Consider interaction terms or tree ensembles (and for the NN, nonlinearity) to reduce high-end bias.
Section 8.0: Results Summary:
•	Classification: We select the Decision Tree as the top classifier, achieving an F1 score of 0.471 (Val) and 0.469 (Test), over Logistic Regression, which yields an F1 score of 0.430 (Val) and 0.369 (Test). Because the target is imbalanced, we prioritize F1 over accuracy (even though LR’s accuracy is higher). The tree’s win tells us we’re benefiting from nonlinear interactions (e.g., alcohol × acidity × sulphates) that a linear boundary can’t capture.
•	Regression. We select Linear Regression as the best regressor (Val MAE 0.573 / RMSE 0.736; Test MAE 0.546 / RMSE 0.701), beating the Decision Tree Regressor (Val 0.658 / 0.960; Test 0.663 / 0.960). These results indicate largely monotonic/linear trends, and the linear model fits better.
•	Class imbalance. We’re working with a skewed label (≈ 81% class 0 vs 19% class 1). Accuracy can appear strong while minority performance lags, so we report F1 to reflect the quality of the positive class.
•	Confusion matrix (best classifier on test). From Plot 3 (Decision Tree), we observe TP = 99, FN = 103, FP = 121, and TN = 741. Errors cluster at the 6/7 cutoff: we miss borderline 7s (false negatives, FN) and incorrectly promote some 6s (false positives, FP). This pattern highlights the limitations of shallow splits and single-feature cues (e.g., alcohol) in the absence of richer interactions.
•	Residuals (best regressor on test). From Plot 4, we observe a slight downward trend in residuals (under-prediction at low predicted scores and over-prediction at high) and mild heteroskedasticity (a wider spread at higher predicted values).
•	Next steps. For classification, we’ll try class weights, probability calibration, and tree ensembles (RF/GBDT). For regression, we’ll use linear regression as a strong baseline and explore interaction terms or an ensemble mitigator to reduce high-end bias. For the final, we’ll consider reinstating k-fold CV or running a small hyperparameter sweep.
Section 9.0: Neural Network Plan
We will implement a small MLP suitable for tabular data (e.g., 2 hidden layers with 32–64 units, ReLU activations), trained with the Adam optimizer and early stopping based on the validation loss. Inputs will be standardized. We will compare the NN against the classical baselines using the same splits and primary metrics (Acc/F1 for classification; MAE/RMSE for Regression).
