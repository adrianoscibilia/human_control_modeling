import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np
import shap
import time
import xgboost
from sklearn.datasets import fetch_california_housing


# train an XGBoost model
housing = fetch_california_housing()
X = housing.data
y = housing.target
model = xgboost.XGBRegressor().fit(X, y)

# explain the model's predictions using SHAP
# (same syntax works for LightGBM, CatBoost, scikit-learn, transformers, Spark, etc.)
# explainer = shap.Explainer(model)
# shap_values = explainer(X)
explainer = shap.KernelExplainer(model.predict, X)
shap_values = explainer.shap_values(X)

# visualize the first prediction's explanation
shap.plots.waterfall(shap_values[0])

# visualize the first prediction's explanation with a force plot
shap.plots.force(shap_values[0])

# visualize all the training set predictions
shap.plots.force(shap_values)

# create a dependence scatter plot to show the effect of a single feature across the whole dataset
shap.plots.scatter(shap_values[:,"RM"], color=shap_values)

# summarize the effects of all the features
shap.plots.beeswarm(shap_values)
shap.plots.bar(shap_values)
