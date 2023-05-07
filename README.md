# machine-learning

# numpy
import numpy
print('numpy: %s' % numpy.__version__)
# matplotlib
import matplotlib
print('matplotlib: %s' % matplotlib.__version__)
# pandas
import pandas
print('pandas: %s' % pandas.__version__)
# scikit-learn
import sklearn
print('sklearn: %s' % sklearn.__version__)

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

df = pd.read_csv('/content/MALE_VFL.csv')
df.shape
df.head(10)
df.describe()

df['Visceral_Fat_Volume_Litres']=pd.cut(df['Visceral_Fat_Volume_Litres'],bins=[0,3,99],labels=['0','1'])

df.isna().sum()
df.dropna(subset=['Visceral_Fat_Volume_Litres'], axis=0, inplace=True)

del df['SUBJECT_ID']
del df['SEX']

df.dropna(thresh=0.8*len(df),axis=1,inplace=True)

import numpy as np
numeric = df.select_dtypes(include=np.number)
numeric_columns = numeric.columns
df[numeric_columns] = df[numeric_columns].fillna(df.mean())

X=df.drop(columns=['Visceral_Fat_Volume_Litres'])
X.head()

y = df['Visceral_Fat_Volume_Litres'].values
y[0:]

df.isna().sum()
df.isna().sum()/len(df)*100
df.dtypes
numeric = df.select_dtypes(include=np.number)

numeric_columns = numeric.columns
df.head(10)
df[numeric_columns] = df[numeric_columns].interpolate(method ='linear', limit_direction ='forward')

import sklearn 
from sklearn.datasets import load_files
import pandas as pd 
import matplotlib.pyplot as plt 

# Create the dataframe 
a = ['COMPUTER_USE_TIME_PER_DAY_HOURS', 'AGE_years', 'HEIGHT_cm']

# Box Plot 
import seaborn as sns 
df.boxplot(a)

Q1 = np.percentile(df[a], 25,  interpolation = 'midpoint')   
Q3 = np.percentile(df[a], 75, interpolation = 'midpoint')  
IQR = Q3 - Q1  
print("Old Shape: ", df.shape)  

# Upper bound 
upper = np.where(df[a] >= (Q3+1.5*IQR)) 

# Lower bound 
lower = np.where(df[a] <= (Q1-1.5*IQR)) 

''' Removing the Outliers ''' 
df.drop(upper[0], inplace = True) 
df.drop(lower[0], inplace = True) 
print("New Shape: ", df.shape)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_df = scaler.fit_transform(X)
print(scaled_df)

import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)

# split X and y into training and testing sets
from sklearn.model_selection import train_test_split

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) 

# Create Decision Tree Classifier object
clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

print('Accuracy:', metrics.accuracy_score(y_test, y_pred))

df_d=pd.DataFrame({'Actual':y_test, 'Predicted':y_pred})
df_d

# Create Decision Tree Classifier object
clf = DecisionTreeClassifier(criterion = 'entropy', max_depth = 3)

# Train Decision Tree Classifer
clf = clf.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)
print('Accuracy:', metrics.accuracy_score(y_test, y_pred))
from sklearn.model_selection import GridSearchCV 

#create new a DT model 
DT = DecisionTreeClassifier() 

#create a dictionary of all values we want to test for max depth 
param_grid = {'max_depth': np.arange(1, 25)} 

#use gridsearch to test all values for max depth 
DT_gscv = GridSearchCV(DT, param_grid, cv=5) 

#fit model to data 
DT_gscv.fit(X, y) 
DT_gscv.best_params_ 
DT_gscv.best_score_ 
from sklearn.neighbors import KNeighborsClassifier 

# Create KNN classifier 
knn = KNeighborsClassifier(n_neighbors = 3) 

# Fit the classifier to the data 
knn= knn.fit(X_train,y_train) 

#show first 5 model predictions on the test data 
knn.predict(X_test)[0:5] 

#check accuracy of our model on the test data 
knn.score(X_test, y_test) 
y_pred1 = knn.predict(X_test)
print('Accuracy:', metrics.accuracy_score(y_test, y_pred1))

# Calculating error for K values between 1 and 40 
error = [] 

import numpy as np 
import matplotlib.pyplot as plt 

# Calculating error for K values between 1 and 40 
for i in range(1, 40): 
    knn = KNeighborsClassifier(n_neighbors=i) 
    knn.fit(X_train, y_train) 
    pred_i = knn.predict(X_test) 
    error.append(np.mean(pred_i != y_test)) 

plt.figure(figsize=(12, 6)) 
plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o', 
         markerfacecolor='blue', markersize=10) 

plt.title('Error Rate K Value') 
plt.xlabel('K Value') 
plt.ylabel('Mean Error') 

from sklearn.model_selection import cross_val_score 
import numpy as np 

#create a new KNN model 
knn_cv = KNeighborsClassifier(n_neighbors=2) 

#train model with cv of 5  
cv_scores = cross_val_score(knn_cv, X, y, cv=5) 

#print each cv score (accuracy) and average them 
print(cv_scores) 
print('cv_scores mean:{}'.format(np.mean(cv_scores))) 

#create new a knn model 
knn2 = KNeighborsClassifier() 

#create a dictionary of all values we want to test for n_neighbors 
param_grid = {'n_neighbors': np.arange(1, 25)} 

#use gridsearch to test all values for n_neighbors 
knn_gscv = GridSearchCV(knn2, param_grid, cv=5) 

#fit model to data 
knn_gscv.fit(X, y) 

#check top performing n_neighbors value 
knn_gscv.best_params_ 

#check mean score for the top performing value of n_neighbors 
knn_gscv.best_score_ 

from sklearn.metrics import accuracy_score 
print('Accuracy: {:.2f}'.format(accuracy_score(y_test, pred_i))) 

#Naive Bayes
from sklearn.naive_bayes import GaussianNB 
nb = GaussianNB() 
nb=nb.fit(X_train, y_train) 
print("Naive Bayes score: ",nb.score(X_test, y_test)) 

from sklearn.metrics import accuracy_score 
y_pred2 = nb.predict(X_test)
print('Accuracy:', metrics.accuracy_score(y_test, y_pred2))

#create a dictionary of all values we want to test for n_neighbors 
param_grid = {'var_smoothing': np.arange(1, 25)} 

#use gridsearch to test all values for n_neighbors 
nb_gscv = GridSearchCV(nb, param_grid, cv=5) 

#fit model to data 
nb_gscv.fit(X, y)

nb_gscv.best_params_ 
nb_gscv.best_score_ 

#ANN
from sklearn.neural_network import MLPClassifier 
# Import necessary modules 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_squared_error 
from math import sqrt 
from sklearn.metrics import r2_score
df.describe().transpose() 

#Scale the Train and Test Data 
from sklearn.preprocessing import StandardScaler 
scaler = StandardScaler() 

# Fit only to the training data 
scaler.fit(X_train) 
StandardScaler(copy=True, with_mean=True, with_std=True) 

# Now apply the transformations to the data: 
X_train = scaler.transform(X_train) 
X_test = scaler.transform(X_test)

%%time 
from sklearn.model_selection import GridSearchCV 

params = {'activation': ['relu', 'tanh', 'logistic', 'identity'], 
          'hidden_layer_sizes': [(13,), (50,100,), (50,75,100,)], 
          'solver': ['adam', 'sgd', 'lbfgs'], 
          'learning_rate' : ['constant', 'adaptive', 'invscaling'], 
          'max_iter': [500] 
         } 

mlp_classif_grid = GridSearchCV(MLPClassifier(random_state=123), param_grid=params, n_jobs=-1, cv=5, verbose=5) 
mlp_classif_grid.fit(X_train,y_train) 

print('Train Accuracy : %.3f'%mlp_classif_grid.best_estimator_.score(X_train, y_train)) 
print('Test Accuracy : %.3f'%mlp_classif_grid.best_estimator_.score(X_test, y_test)) 
print('Best Accuracy Through Grid Search : %.3f'%mlp_classif_grid.best_score_) 
print('Best Parameters : ',mlp_classif_grid.best_params_) 

mlp = MLPClassifier(activation= 'relu', hidden_layer_sizes= (13,), learning_rate='constant', solver='lbfgs', max_iter=500) 
mlp = mlp.fit(X_train,y_train)

y_pred3 = mlp.predict(X_test)
print('Accuracy:', metrics.accuracy_score(y_test, y_pred3))
predictions = mlp.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix 
#Confusion Matrix for ANN
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions)) 

#Confusion Matrix for DT
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred)) 

#Confusion Matrix for KNN
pred = knn.predict(X_test)
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred)) 

#Confusion matrix for NB
predict = nb.predict(X_test)
print(confusion_matrix(y_test,predict))
print(classification_report(y_test,predict)) 

DT_best = DT_gscv.best_estimator_
knn_best = knn_gscv.best_estimator_
nb_best = nb_gscv.best_estimator_
mlp_best = mlp_classif_grid.best_estimator_

from sklearn.metrics import RocCurveDisplay 
DT.fit(X_train, y_train)
DT_disp = RocCurveDisplay.from_estimator(DT, X_test, y_test) 
nb_disp = RocCurveDisplay.from_estimator(nb, X_test, y_test) 
knn_disp = RocCurveDisplay.from_estimator(knn, X_test, y_test) 
mlp_disp = RocCurveDisplay.from_estimator(mlp, X_test, y_test) 

print('knn: {}'.format(knn_best.score(X_test, y_test))) 
print('DT: {}'.format(DT_best.score(X_test, y_test))) 
print('nb: {}'.format(nb_best.score(X_test, y_test)))
print('mlp: {}'.format(mlp_best.score(X_test, y_test)))

from sklearn.ensemble import VotingClassifier 

#create a dictionary of our models 
estimators=[('knn', knn_best), ('DT', DT_best), ('nb', nb_best), ('mlp', mlp_best)] 

#create our voting classifier, inputting our models 
ensemble = VotingClassifier(estimators, voting='hard') 

#fit model to training data 
ensemble.fit(X_train, y_train) 

#test our model on the test data 
ensemble.score(X_test, y_test) 

# install pycaret
!pip install pycaret

from pycaret.classification import *
s = setup(df, target = 'Visceral_Fat_Volume_Litres')
# compare all models
best_model = compare_models(sort='F1')

# tune the best model
tuned_model_knn = tune_model(knn)
tuned_model_nb = tune_model(nb)
tuned_model_DT = tune_model(DT)

# Analyse the AUC Plot
plot_model(tuned_model_knn, plot = 'auc')
# Analyse the Confusion Matrix
plot_model(tuned_model_knn, plot = 'confusion_matrix')

