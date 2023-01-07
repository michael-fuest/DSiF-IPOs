import pandas as pd
import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf

#Defining path and filenames
path = './data/'
largefile = 'master_data.csv'
df = pd.read_csv(path + largefile, index_col= 'ipoDate')

#Dropping Symbol column
df.drop(columns=['Symbol', 'index'], inplace=True, axis = 1)

#One hot encoding sector and weekday cols
categorical_cols = ['sector', 'IPO_weekday'] 
df = pd.get_dummies(df, columns = categorical_cols)

#Normalizing numerical variables for ML modeling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(df)
df = pd.DataFrame(normalized_data, columns=df.columns)

#Logistic Regression Modeling
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn import metrics

X = df.drop(columns= ['intra_day_up', 'intra_week_up', 'intra_month_up', 'firstday_volume', 'inweek_volume', 'inmonth_volume'])
y_d = np.array(df.intra_day_up)
y_w = np.array(df.intra_week_up)
y_m = np.array(df.intra_month_up)

#Dealing with class imbalances by oversampling minority class (underperforming IPOs)
sm = SMOTE(random_state=27)
X_d_bal, y_d_bal = sm.fit_resample(X, y_d)
X_w_bal, y_w_bal = sm.fit_resample(X, y_w)
X_m_bal, y_m_bal = sm.fit_resample(X, y_m)

#Model accuracy does not improve substantially with SMOTE, but recall and balanced acc does.
# setting up stratified testing and training sets
X_d_train, X_d_test, y_d_train, y_d_test = train_test_split(X_d_bal, y_d_bal, test_size=0.2, random_state=0, stratify = y_d_bal)
X_w_train, X_w_test, y_w_train, y_w_test = train_test_split(X_w_bal, y_w_bal, test_size=0.2, random_state=0, stratify = y_w_bal)
X_m_train, X_m_test, y_m_train, y_m_test = train_test_split(X_m_bal, y_m_bal, test_size=0.2, random_state=0, stratify = y_m_bal)

#Fitting Logistic Regression Models
logreg_d = LogisticRegression()
logreg_w = LogisticRegression()
logreg_m = LogisticRegression()

logreg_d.fit(X_d_train, y_d_train)
logreg_w.fit(X_w_train, y_w_train)
logreg_m.fit(X_m_train, y_m_train)

log_pred_d = logreg_d.predict(X_d_test)
log_pred_w = logreg_w.predict(X_w_test)
log_pred_m = logreg_m.predict(X_m_test)

from sklearn.model_selection import RandomizedSearchCV
#Creating grid search for Random Forest Hyperparameter Tuning
#Taken from https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1500, num = 10)]
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 200, num = 10)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

from sklearn.ensemble import RandomForestClassifier
# Use the random grid to search for best hyperparameters
# First create the base models to tune for each predicted label
rf_d = RandomForestClassifier()
rf_w = RandomForestClassifier()
rf_m = RandomForestClassifier()

# Random search of parameters, using 10 fold cross validation, 
# search across 50 different combinations, and use all available cores
rf_random_d = RandomizedSearchCV(estimator = rf_d, param_distributions = random_grid, n_iter = 50, cv = 10, random_state=42, n_jobs = -1)
rf_random_w = RandomizedSearchCV(estimator = rf_w, param_distributions = random_grid, n_iter = 50, cv = 10, random_state=42, n_jobs = -1)
rf_random_m = RandomizedSearchCV(estimator = rf_m, param_distributions = random_grid, n_iter = 50, cv = 10, random_state=42, n_jobs = -1)

print('Starting randomized grid search for random forest classifier...')

# Fit all random search models
rf_random_d.fit(X_d_train, y_d_train)
rf_random_w.fit(X_w_train, y_w_train)
rf_random_m.fit(X_m_train, y_m_train)

print('Randomized Search completed for random forest model.')

#Evaluating accuracy of rf models
best_random_d = rf_random_d.best_estimator_
best_random_w = rf_random_w.best_estimator_
best_random_m = rf_random_m.best_estimator_

#Creating random model predictions
rf_predictions_d = best_random_d.predict(X_d_test)
rf_predictions_w = best_random_d.predict(X_w_test)
rf_predictions_m = best_random_d.predict(X_m_test)

from xgboost import XGBClassifier

#Creating XGB models for each predicted label
xgb_d = XGBClassifier(objective="binary:logistic")
xgb_w = XGBClassifier(objective="binary:logistic")
xgb_m = XGBClassifier(objective="binary:logistic")

#Defining search grid for randomized search
param_grid = {
    "max_depth": [int(x) for x in np.linspace(10, 110, num = 20)],
    "learning_rate": [0.1, 0.01, 0.05, 0.2, 0.25, 0.3],
    "gamma": [0, 0.1, 0.25, 0.5, 1],
    "reg_lambda": [0, 1, 2, 5, 10],
    "scale_pos_weight": [1, 2, 3, 4, 5],
    "subsample": [0.8],
    "colsample_bytree": [0.5],
}

#Fitting randomized search to XGB models
xgb_random_d = RandomizedSearchCV(estimator = xgb_d, param_distributions = param_grid, n_iter = 50, cv = 10, random_state=42, n_jobs = -1)
xgb_random_w = RandomizedSearchCV(estimator = xgb_w, param_distributions = param_grid, n_iter = 50, cv = 10, random_state=42, n_jobs = -1)
xgb_random_m = RandomizedSearchCV(estimator = xgb_m, param_distributions = param_grid, n_iter = 50, cv = 10, random_state=42, n_jobs = -1)

print('Starting randomized grid search for XGboost classifier...')

#Fitting search to training data
xgb_random_d.fit(X_d_train, y_d_train)

xgb_random_w.fit(X_w_train, y_w_train)
xgb_random_m.fit(X_m_train, y_m_train)

print('Randomized Search completed for XGBoost model.')

#Evaluating accuracy of xgb models
best_random_d = xgb_random_d.best_estimator_
best_random_w = xgb_random_w.best_estimator_
best_random_m = xgb_random_m.best_estimator_

#Creating random model predictions
xgb_predictions_d = best_random_d.predict(X_d_test)
xgb_predictions_w = best_random_d.predict(X_w_test)
xgb_predictions_m = best_random_d.predict(X_m_test)

#Looking at feature importances across models
#Feature importance for rf models
feature_names = [col for col in X.columns]
rf_d_importance = pd.Series(rf_random_d.best_estimator_.feature_importances_, index = feature_names)
rf_w_importance = pd.Series(rf_random_w.best_estimator_.feature_importances_, index = feature_names)
rf_m_importance = pd.Series(rf_random_m.best_estimator_.feature_importances_, index = feature_names)

#Feature importance for xgb models
xgb_d_importance = pd.Series(xgb_random_d.best_estimator_.feature_importances_, index = feature_names)
xgb_w_importance = pd.Series(xgb_random_w.best_estimator_.feature_importances_, index = feature_names)
xgb_m_importance = pd.Series(xgb_random_m.best_estimator_.feature_importances_, index = feature_names)

#Looking at most important features in logistic regression model
log_d_importance = pd.Series(abs(logreg_d.coef_[0]), index = X.columns)
log_w_importance = pd.Series(abs(logreg_w.coef_[0]), index = X.columns)
log_m_importance = pd.Series(abs(logreg_m.coef_[0]), index = X.columns)


#Result aggregation
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score

#RF results
rf_results = pd.DataFrame({'pred_label':[], 'accuracy': [], 'balanced_accuracy': [], 'top_5_features': []})
rf_results['model'] = ['RFModel']*3
rf_results.accuracy = [accuracy_score(y_d_test, rf_predictions_d), accuracy_score(y_w_test, rf_predictions_w), accuracy_score(y_m_test, rf_predictions_m)]
rf_results.balanced_accuracy = [balanced_accuracy_score(y_d_test, rf_predictions_d), balanced_accuracy_score(y_w_test, rf_predictions_w), balanced_accuracy_score(y_m_test, rf_predictions_m)]
rf_results.top_5_features = [rf_d_importance.nlargest(5).index.tolist(), rf_w_importance.nlargest(5).index.tolist(), rf_m_importance.nlargest(5).index.tolist()]
rf_results.pred_label = ['intra day underperformance', 'in-week underperformance', 'in-month underperformance']


#XGB results
xgb_results = pd.DataFrame({'pred_label':[], 'accuracy': [], 'balanced_accuracy': [], 'top_5_features': []})
xgb_results['model'] = ['XGBoost']*3
xgb_results.accuracy = [accuracy_score(y_d_test, xgb_predictions_d), accuracy_score(y_w_test, xgb_predictions_w), accuracy_score(y_m_test, xgb_predictions_m)]
xgb_results.balanced_accuracy = [balanced_accuracy_score(y_d_test, xgb_predictions_d), balanced_accuracy_score(y_w_test, xgb_predictions_w), balanced_accuracy_score(y_m_test, xgb_predictions_m)]
xgb_results.top_5_features = [xgb_d_importance.nlargest(5).index.tolist(), xgb_w_importance.nlargest(5).index.tolist(), xgb_m_importance.nlargest(5).index.tolist()]
xgb_results.pred_label = ['intra day underperformance', 'in-week underperformance', 'in-month underperformance']


#Logreg results
log_results = pd.DataFrame({'pred_label':[], 'accuracy': [], 'balanced_accuracy': [], 'top_5_features': []})
log_results['model'] = ['Logreg']*3
log_results.accuracy = [accuracy_score(y_d_test, log_pred_d), accuracy_score(y_w_test, log_pred_w), accuracy_score(y_m_test, log_pred_m)]
log_results.balanced_accuracy = [balanced_accuracy_score(y_d_test, log_pred_d), balanced_accuracy_score(y_w_test, log_pred_w), balanced_accuracy_score(y_m_test, log_pred_m)]
log_results.top_5_features = [log_d_importance.nlargest(5).index.tolist(), log_w_importance.nlargest(5).index.tolist(), log_m_importance.nlargest(5).index.tolist()]
log_results.pred_label = ['intra day underperformance', 'in-week underperformance', 'in-month underperformance']



print(pd.concat([rf_results, xgb_results, log_results], axis = 0))