# Breast_cancer_prediction-
ML model's to detect breast cancer 


**PROJECT OVERVIEW** 

Breast cancer is one of the most common cancers affecting women worldwide. Early and accurate diagnosis is crucial for effective treatment.
This project implements machine learning models to predict whether a tumor is Malignant (M) or Benign (B) based on features extracted from cell nuclei obtained from breast mass images.

The models included in this project are:
Logistic Regression – A simple linear model for binary classification.
Random Forest Classifier – An ensemble tree-based method that improves accuracy by combining multiple decision trees.
XGBoost Classifier – An optimized gradient boosting algorithm known for high performance in structured data.

**Libraries Used**
pandas – Data manipulation
numpy – Numerical computations
matplotlib & seaborn – Data visualization
scikit-learn – Machine learning algorithms and evaluation
xgboost – Gradient boosting classifier

**DATA SET**

The dataset used in this project is the Breast Cancer Wisconsin (Diagnostic) Dataset, which contains features computed from digitized images of fine needle aspirates (FNA) of breast masses. 
These features describe characteristics of the cell nuclei present in the images.

The dataset used in this project is the **Breast Cancer Wisconsin (Diagnostic) Dataset** from Kaggle.

- **Source / Download**: [Breast Cancer Wisconsin (Diagnostic) Data Set on Kaggle]. [https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data?utm_]

Description: The dataset includes 30 features such as radius, texture, perimeter, area, smoothness, compactness, concavity, and symmetry, along with a target variable indicating whether the tumor is malignant (M) or benign (B).
License: Public Domain
Features:
Radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension (mean, standard error, worst).

 **DATA PREPROCESSING**
Data preprocessing steps include:
Drop irrelevant or constant columns – Columns with the same value in all rows do not contribute to model learning.
Handle missing values – Replacing NaN values in numeric columns with column mean.
Handle infinite values – Replace inf and -inf with NaN, then impute.
Encode target – Convert diagnosis from categorical (M/B) to numeric (1/0).
Feature scaling – StandardScaler is applied to numeric features for models sensitive to scale, such as Logistic Regression.

**EXPLORATORY DATA ANALYSIS** 
Correlation heatmap to visualize relationships between features.
Descriptive statistics to understand data distribution.
Visualization of distributions for key features.
This helps to identify multicollinearity, feature relevance, and data quality issues.

**TRAIN- TEST SPLIT**
The dataset is split into 80% training and 20% testing using stratification to maintain class balance.
Random state is set for reproducibility

**MODEL TRAINING**
Three models are trained:
Logistic Regression
Good baseline for binary classification.

**Scaled features improve performance**
Random Forest
Ensemble of decision trees to reduce overfitting.
Provides feature importance scores.
XGBoost
Gradient boosting framework optimized for speed and accuracy.
Often achieves higher performance on tabular data.
**EVALUATION METRICS**
Accuracy
Confusion Matrix
Classification Report (Precision, Recall, F1-score)

**MODEL COMPARISON**
Model performances are visualized using a bar chart for easy comparison.
The best-performing model is highlighted based on test set accuracy.

**FEATURE IMPORTANCE**
For tree-based models (Random Forest, XGBoost), feature importance plots show which features most influence the predictions.
This provides insights into critical factors for breast cancer diagnosis, e.g., mean radius, perimeter, concave points.

**RESULTS**
Best Model: XGBoost (highest accuracy)
Key Features: Features with the largest contribution to the model are identified through feature importance plots.
Confusion matrices and classification reports show how well the models distinguish Malignant vs. Benign tumors.

**FUTURE SCOPE**
1 Hyperparameter Tuning:
Improve model performance by fine-tuning parameters using GridSearchCV or RandomizedSearchCV.
Example: Optimize number of trees in Random Forest or learning rate in XGBoost.

2 Cross-Validation:
Implement k-fold cross-validation to get a more robust estimate of model accuracy.
Reduces variance from a single train-test split.

3 ROC-AUC and Threshold Analysis:
Evaluate models using ROC curves and AUC scores to understand trade-offs between sensitivity and specificity.
Helps optimize classification thresholds for better Malignant detection.

4 Feature Engineering:
Create additional features or transformations (ratios, normalized metrics) to capture hidden patterns in the data.
Could improve model accuracy and interpretability.

5 Ensemble Methods:
Combine multiple models (stacking, voting classifiers) to improve predictive performance.
Example: Logistic Regression + XGBoost + Random Forest ensemble.
