
import pandas as pd
import numpy as np
from pandas.core.computation.expressions import evaluate
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn import model_selection
import sklearn.model_selection
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


seed = 520

def create_model_random_forest():
    # You can find the full list of parameters here:
    # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    #model = RandomForestClassifier(n_estimators=100,
    #                               min_samples_split=5,
    #                               random_state=seed,
    #                               n_jobs=-1)

    #model = RandomForestClassifier(
    #    max_depth=110,
    #    bootstrap=True,
    #    max_features='auto',
    #    n_estimators=500,
    #    criterion='gini',
    #)

    model = RandomForestClassifier(
        max_depth=None,
        n_estimators=500,
        min_samples_split=5,
    )


    return model

def create_scaler_standard():
    return StandardScaler()

create_model = create_model_random_forest

create_scaler = create_scaler_standard

np.set_printoptions(precision=3)

print('Load the data')
# This is a classification problem with 2 target classes.
training_data = pd.read_csv('data/labeled.csv')
X = training_data.drop('y', axis=1)
num_missing = (X[['x1','x2','x3','x4','x5','x6','x7','x8','x9','x10','x11','x12','x13','x14','x15','x16','x17','x18','x19','x20','x21']]=='None').sum()

#As we can see, only x1, x4 and x8 have missing data.
#print(num_missing)

# Here we will replace the missing data with '-1' values.
X[['x1','x4','x8']] = X[['x1','x4','x8']].replace('None', -1)
X = X.to_numpy()
y = training_data['y']
y = y.to_numpy()


count_class_0, count_class_1 = training_data['y'].value_counts()

print("Class Zero:", count_class_0)
print("Class One:", count_class_1)

df_class_0 = training_data[training_data['y'] == 0]
df_class_1 = training_data[training_data['y'] == 1]

df_class_1_over = df_class_1.sample(count_class_0, replace=True)
df_test_over = pd.concat([df_class_0, df_class_1_over], axis=0)

print('Random over-sampling:')
print(df_test_over['y'].value_counts())

y_oversampled = df_test_over['y']
X_oversampled = df_test_over.drop('y',axis=1)
X_oversampled[['x1','x4','x8']] = X_oversampled[['x1','x4','x8']].replace('None', -1)
X_oversampled = X_oversampled.to_numpy()
y_oversampled = y_oversampled.to_numpy()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


print('Features:')
print(X)

print('Targets:')
print(y)

'''
print("Let's look for some poggers hyperparameters")
from sklearn.model_selection import GridSearchCV
# Create the parameter grid based on the results of random search
param_grid = {
    'bootstrap': [True],
    'max_depth': [110, 200, None],
    'max_features': ['auto', 'sqrt', 'log2'],
    'criterion' :['gini', 'entropy'],
    'n_estimators': [100, 200, 300, 500, 1000]
    #'max_depth': [110, 200, None],
    #'min_samples_split': [2, 5, 10],
    #'n_estimators': [100, 200, 300, 500, 1000]
}
# Create a based model
rf = RandomForestClassifier()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid,
                          cv = 3, n_jobs = -1, verbose = 2)

# Fit the grid search to the data
grid_search.fit(X_train, y_train)
print("best params baby")
print(grid_search.best_params_)
best_grid = grid_search.best_estimator_
print("best grid", best_grid)

'''


print('Train the model and predict')
scaler = create_scaler()
model = create_model()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
model.fit(X_train_scaled, y_train)
y_train_hat = model.predict(X_train_scaled)
y_test_hat = model.predict(X_test_scaled)

print('Model evaluation (train):')
print('Accuracy:')
print(metrics.accuracy_score(y_test, y_test_hat))

print('Classification report:')
print(metrics.classification_report(y_test, y_test_hat))


print('Confusion matrix')
df = pd.DataFrame({'y_Actual': y_test, 'y_Predicted': y_test_hat})
confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])
print(confusion_matrix)

print('Cross-validation')
np.random.seed(seed)
y_cv_hat = np.zeros(y.shape)

kfold = model_selection.KFold(n_splits=5, shuffle=True, random_state=seed)

# Cross-validation
for cv_train, cv_test in kfold.split(X, y):
    # Train classifier on training data, predict test data

    # Scaling train and test data
    # Train scaler on training set only
    scaler.fit(X[cv_train])
    X_cv_train = scaler.transform(X[cv_train])
    X_cv_test = scaler.transform(X[cv_test])

    model = create_model()
    model.fit(X_cv_train, y[cv_train])
    y_cv_hat[cv_test] = model.predict(X_cv_test)

print('Model evaluation (CV):')
print('Accuracy:')
print(metrics.accuracy_score(y, y_cv_hat))

print('Classification report:')
print(metrics.classification_report(y, y_cv_hat))

df = pd.DataFrame({'y_Actual': y, 'y_Predicted': y_cv_hat})
confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])
print(confusion_matrix)