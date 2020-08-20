#libraries 
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import auc, roc_curve
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import RandomizedSearchCV

#importing data
raw = pd.read_csv("Stats.csv")

#removing NAs
raw = raw.dropna()

#ploting class balance
plt.figure(figsize=(8, 8))
sns.countplot('winner', data=raw)
plt.title('Class')
plt.show()


#lable encoding for catagorical data
bin_cols = raw.nunique()[raw.nunique() == 2].keys().tolist()
le = LabelEncoder()
for i in bin_cols :
    raw[i] = le.fit_transform(raw[i])

#droping unwanted coulmns 
raw = raw.drop(["match_id", "player_id"],axis = 1)

#X- indipendent variables, y- dependent variable
X = raw.iloc[:, :-1].values
y = raw.iloc[:, -1].values


#PCA for dimensonality reduction
pca = PCA(n_components=2)
X = pca.fit_transform(X)

pca.explained_variance_ratio_



# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)


#implimenting SVC
svm = SVC()
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)
acc = accuracy_score(y_pred,y_test)
print("Accuracy :", acc)
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
score_auc = auc(false_positive_rate, true_positive_rate)
print("AUC Score =", score_auc)
score_f1 = f1_score(y_test, y_pred)
print("F1 Score :", score_f1)
score_kappa = cohen_kappa_score(y_pred,y_test)
print("Cohens Kappa :", score_kappa)

#seting hyper parameters
smc_params = {'C': range(1, 10, 1), 'gamma': np.arange(0.1, 1, 0.1), 'kernel': ['rbf', 'poly']}

#random search cv for hyper paramter tuning
random_search = RandomizedSearchCV(estimator = svm, param_distributions = svm_params, n_iter = 10, cv = 5, verbose=2, random_state=42, n_jobs = -1)
random_search.fit(X_train, y_train)

#checking the best hyper paramters
print('\n Best estimator:')
print(random_search.best_estimator_)
print('\n Best hyperparameters:')
print(random_search.best_params_)

#implimenting Gaussian naive bayes
gaussian = GaussianNB()
gaussian.fit(X_train, y_train)
# Predicting the Test set results
y_pred = gaussian.predict(X_test)
acc = accuracy_score(y_pred,y_test)
print("Accuracy :", acc)
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
score_auc = auc(false_positive_rate, true_positive_rate)
print("AUC Score =", score_auc)
score_f1 = f1_score(y_test, y_pred)
print("F1 Score :", score_f1)
score_kappa = cohen_kappa_score(y_pred,y_test)
print("Cohens Kappa :", score_kappa)



#implimenting random forest
from sklearn.ensemble import RandomForestClassifier
randomf = RandomForestClassifier(n_estimators = 690, min_samples_split = 6, min_samples_leaf = 1, max_features 'auto', max_depth 10)
randomf.fit(X_train, y_train)
# Predicting the Test set results
y_pred = randomf.predict(X_test)
acc = accuracy_score(y_pred,y_test)
print("Accuracy :", acc)
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
score_auc = auc(false_positive_rate, true_positive_rate)
print("AUC Score =", score_auc)
score_f1 = f1_score(y_test, y_pred)
print("F1 Score :", score_f1)
score_kappa = cohen_kappa_score(y_pred,y_test)
print("Cohens Kappa :", score_kappa)

#seting range for hyper parameter
randomf_params = {'n_estimators': range(200, 2000, 10),
               'max_features': ['auto', 'sqrt'],
               'max_depth': range(10, 110, 10),
               'min_samples_split': range(2, 10, 1),
               'min_samples_leaf': [1, 2, 4]}

#random search cv for hyper paramter tuning
random_search = RandomizedSearchCV(estimator = randomf, param_distributions = random_params, n_iter = 50, cv = 10, verbose=2, random_state=42, n_jobs = -1)
random_search.fit(X_train, y_train)


print('\n Best estimator:')
print(random_search.best_estimator_)
print('\n Best hyperparameters:')
print(random_search.best_params_)


#implimenting logistic regression
from sklearn.linear_model import LogisticRegression
logi = LogisticRegression(penalty = 'l1', C = 1.0)
logi.fit(X_train, y_train)
y_pred = logi.predict(X_test)
acc = accuracy_score(y_pred,y_test)
print("Accuracy :", acc)
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
score_auc = auc(false_positive_rate, true_positive_rate)
print("AUC Score =", score_auc)
score_f1 = f1_score(y_test, y_pred)
print("F1 Score :", score_f1)
score_kappa = cohen_kappa_score(y_pred,y_test)
print("Cohens Kappa :", score_kappa)

#seting range for hyper pramter tuning
logi_param={"C":np.logspace(-3,3,7), "penalty":["l1","l2"],}

#random search cv for hyper paramter tuning
random_search = RandomizedSearchCV(lr, param_distributions=logi_param, n_iter=50, scoring='accuracy', n_jobs= -1, verbose=3)

random_search.fit(X_train, y_train)


print('\n Best estimator:')
print(random_search.best_estimator_)
print('\n Best hyperparameters:')
print(random_search.best_params_)

