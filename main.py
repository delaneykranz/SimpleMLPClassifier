# Delaney Kranz
# Final Project: Multi-Layer Perceptron Classifier
import numpy as np
from numpy import nan
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from sklearn import model_selection
from sklearn.preprocessing import RobustScaler

'''For the scaler, I tried out all of the sklearn.preprocessing scalers and 
saw that the RobustScaler resulted in the best consistent performance.

sklearn documentation states that RobustScaler is pretty good at handling outliers.
I think this is good in the case of our data, where we have missing values.
'''
create_scaler = RobustScaler()




mlp = MLPClassifier(max_iter=300)

'''
parameter_space = {
    'hidden_layer_sizes': [(50,50,50), (150,100), (50,100,50), (100,)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.005, 0.05],
    'learning_rate': ['constant','adaptive'],
}

# This is a classification problem with 2 target classes.
train = pd.read_csv('data/labeled.csv')

X = train.drop('y',axis=1)
num_missing = (X[['x1','x2','x3','x4','x5','x6','x7','x8','x9','x10','x11','x12','x13','x14','x15','x16','x17','x18','x19','x20','x21']]=='None').sum()

#As we can see, only x1, x4 and x8 have missing data.
print(num_missing)

# Here we will replace the missing data with 'None' values.
X[['x1','x4','x8']] = X[['x1','x4','x8']].replace('None', -1)
X = X.to_numpy()
y = train['y']
y = y.to_numpy()

from sklearn.model_selection import GridSearchCV

clf = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=5)
clf.fit(X, y)

# Best parameter set
print('Best parameters found:\n', clf.best_params_)

# All results
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

'''

seed = 520

def create_model_mlpclassifier():
    model = MLPClassifier(hidden_layer_sizes=(50,100, 50), activation='relu', max_iter=300, learning_rate='constant', solver='adam', random_state=seed)
    return model

create_model = create_model_mlpclassifier

np.set_printoptions(precision=3)

print('Load the data')
# This is a classification problem with 2 target classes.
train = pd.read_csv('data/labeled.csv')
#print(train)

count_class_0, count_class_1 = train['y'].value_counts()

print("Class Zero:", count_class_0)
print("Class One:", count_class_1)

df_class_0 = train[train['y'] == 0]
df_class_1 = train[train['y'] == 1]


#Random under-sampling feels illegal, so I don't want to do this.
'''
df_class_0_under = df_class_0.sample(count_class_1)
df_test_under = pd.concat([df_class_0_under, df_class_1], axis=0)

print('Random under-sampling:')
print(df_test_under['y'].value_counts())
'''
df_class_1_over = df_class_1.sample(count_class_0, replace=True)
df_test_over = pd.concat([df_class_0, df_class_1_over], axis=0)

print('Random over-sampling:')
print(df_test_over['y'].value_counts())

df_test_over.to_csv('oversampling.csv')

y_oversampled = df_test_over['y']
X_oversampled = df_test_over.drop('y',axis=1)
X_oversampled[['x1','x4','x8']] = X_oversampled[['x1','x4','x8']].replace('None', -1)
X_oversampled = X_oversampled.to_numpy()
y_oversampled = y_oversampled.to_numpy()

brain = pd.read_csv('data/labeled.csv')
X = brain.drop('y',axis=1)
num_missing = (X[['x1','x2','x3','x4','x5','x6','x7','x8','x9','x10','x11','x12','x13','x14','x15','x16','x17','x18','x19','x20','x21']]=='None').sum()

#As we can see, only x1, x4 and x8 have missing data.
#print(num_missing)

# Here we will replace the missing data with '-1' values.
X[['x1','x4','x8']] = X[['x1','x4','x8']].replace('None', -1)
X = X.to_numpy()
y = brain['y']
y = y.to_numpy()

print('Features:')
#print(X)

print('Targets:')
#print(y)

print('Train the model and predict')
scaler = RobustScaler()
model = create_model()
model.fit(X_oversampled, y_oversampled)
#model.fit(X, y)
y_hat = model.predict(X)
print(np.count_nonzero(y_hat))
print(np.shape(y_hat))

pd.DataFrame(y_hat).to_csv('y_hat.csv')

print('Model evaluation (train)')
print('Accuracy:')
print(metrics.accuracy_score(y, y_hat))
print('Classification report:')
print(metrics.classification_report(y, y_hat))

print('Confusion matrix (train)')


print('Confusion matrix')
df = pd.DataFrame({'y_Actual': y, 'y_Predicted': y_hat})
confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])
print(confusion_matrix)

print('Cross-validation')
np.random.seed(seed)
y_prob = np.zeros(y.shape)
y_hat = np.zeros(y.shape)

kfold = model_selection.KFold(n_splits=5, shuffle=True, random_state=seed)

# Cross-validation
for train, test in kfold.split(X, y):
    # Train classifier on training data, predict test data

    # Scaling train and test data
    # Train scaler on training set only
    scaler.fit(X[train])
    X_train = scaler.transform(X[train])
    X_test = scaler.transform(X[test])

    #X_resampled, y_resampled = ros.fit_resample(X_train, y[train])

    model = create_model()
    #model.fit(X_resampled, y_resampled)
    model.fit(X_train, y[train])
    y_prob[test] = model.predict_proba(X_test)[:, 1]
    y_hat[test] = model.predict(X_test)

print('Model evaluation (CV)')
print('Accuracy:')
print(metrics.accuracy_score(y, y_hat))
print('Classification report:')
print(metrics.classification_report(y, y_hat))


print('Confusion matrix')
df = pd.DataFrame({'y_Actual': y, 'y_Predicted': y_hat})
confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])
print(confusion_matrix)


