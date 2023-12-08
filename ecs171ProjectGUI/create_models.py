import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, confusion_matrix, multilabel_confusion_matrix
from sklearn.metrics import mean_squared_error, accuracy_score, r2_score
from sklearn.preprocessing import OneHotEncoder

#Model 1
filePath = 'California_Fire_Incidents.csv'
dataset = pd.read_csv(filePath)
dataset.fillna(0, inplace=True)
#eliminate non zero injury rows, just train on injuries 
X = dataset.loc[dataset['Injuries'] != 0, ['CrewsInvolved', 'PersonnelInvolved']]
y = dataset.loc[dataset['Injuries'] != 0, ['Injuries']]
#scale inputs for training 
scaler = MinMaxScaler()
print("X: ", X)
print("X.colums: ", X.columns)
X = scaler.fit_transform(X)
counties = dataset.loc[dataset['Injuries'] != 0, ['Counties']]

encoder = OneHotEncoder()
encoded_counties = encoder.fit_transform(counties.values.reshape(-1, 1))
X = np.hstack((encoded_counties.toarray(), X))

#train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
mlp = MLPClassifier(solver = 'sgd', random_state = 42, activation = 'identity', learning_rate_init = 0.4, batch_size = 20, hidden_layer_sizes = (3, 2), max_iter = 600)

n_splits=10
# step 1: randomize the dataset and create k equal size partitions
kf = KFold(n_splits=n_splits)

acc = 0
mse = 0

i = 0 #keep track of batch number
# step 5: iterate k times with a different testing subset
for train_indices, test_indices in kf.split(X):

    # step 2-3: use k-1/k^th partition for the training/testing model
    start_train, stop_train = train_indices[0], train_indices[-1]+1
    start_test, stop_test = test_indices[0], test_indices[-1]+1
    
    # perform the training similar to Q1
    #this was based on the requirements in Q1
    mlp = MLPClassifier(solver = 'sgd', random_state = 42, activation = 'logistic', learning_rate_init = 0.3, batch_size = 15, hidden_layer_sizes = (8, 5), max_iter = 600)
    mlp.fit(X[start_train:stop_train], y[start_train:stop_train])
    pred = mlp.predict(X[start_test:stop_test])
    
    # step 4: record the evaluating scores
    i+=1
    acc += accuracy_score(y[start_test:stop_test], pred)
    mse += mean_squared_error(y[start_test:stop_test], pred)
    
    print("\nAccuracy for batch ", i, " : ", accuracy_score(y[start_test:stop_test], pred))
    print("Mean Square Error for batch ", i, " : ", mean_squared_error(y[start_test:stop_test], pred))
    print("R-Squared: ", r2_score(y[start_test:stop_test], pred))

joblib.dump(encoder, 'county_encoder.gz')
joblib.dump(scaler, 'model1_x_scaler.gz')
joblib.dump(mlp, 'mlp.gz')




#Model 2
from sklearn.linear_model import LinearRegression

wf = pd.read_csv(filePath)

wf['Month'] = wf['Started'].str[5:7].astype(int)

temp_df = pd.read_csv('temperature_data.csv')


temp_df['Date'] = temp_df['Date'].apply(str)
temp_df['Month'] = temp_df['Date'].str[4:6].astype(int)
temp_df['ArchiveYear'] = temp_df['Date'].str[0:4].astype(int)

temp_df = temp_df.drop(['Anomaly', 'Date'], axis=1)

df_freq = wf[['Month', 'ArchiveYear']]

train, test = train_test_split(df_freq, test_size=0.5, random_state=10)
def make_model(i):
  i_1 = i - 1

  if (i_1 < 1):
    i_1 += 12


  n1_train = pd.DataFrame(train[train['Month'] == i_1]['ArchiveYear'].value_counts().sort_index(axis=0))

  n1_train.columns = ['n1']

  t1 = temp_df[temp_df['Month'] == i_1][['Value', 'ArchiveYear']].sort_index(axis=0)

  t1.columns = ['t1', 'ArchiveYear']

  x_train = t1.set_index('ArchiveYear').join(n1_train, on='ArchiveYear')


  x_train.fillna(0, inplace=True)

  n1_test = pd.DataFrame(test[test['Month'] == i_1]['ArchiveYear'].value_counts().sort_index(axis=0))

  n1_test.columns = ['n1']

  x_test = t1.set_index('ArchiveYear').join(n1_test, on='ArchiveYear')

  x_test.fillna(0, inplace=True)

  x_scaler = MinMaxScaler(feature_range=(0, 1))
  x_train_rescaled = x_scaler.fit_transform(x_train)
  x_train_rescaled = pd.DataFrame(data = x_train_rescaled, columns = x_train.columns)

  x_test_rescaled = x_scaler.transform(x_test)
  x_test_rescaled = pd.DataFrame(data = x_test_rescaled, columns = x_test.columns)


  y_train = pd.DataFrame(train[train['Month'] == i]['ArchiveYear'].value_counts().sort_index(axis=0))

  y_test = pd.DataFrame(test[test['Month'] == i]['ArchiveYear'].value_counts().sort_index(axis=0))


  for j in list(df_freq['ArchiveYear'].unique()):
    y_zero = pd.DataFrame({"ArchiveYear": 0}, index=([j]))
    if (j not in list(y_train.index)):
      y_train = pd.concat([y_train, y_zero])
    if (j not in list(y_test.index)):
      y_test = pd.concat([y_test, y_zero])

  y_train = y_train.sort_index(axis=0)
  y_test = y_test.sort_index(axis=0)

  scaler = MinMaxScaler(feature_range=(0, 1))
  y_train_rescaled = scaler.fit_transform(y_train)
  y_train_rescaled = pd.DataFrame(data = y_train_rescaled, columns = y_train.columns)

  y_test_rescaled = scaler.transform(y_test)
  y_test_rescaled = pd.DataFrame(data = y_test_rescaled, columns = y_test.columns)

  lr = LinearRegression()
  lr.fit(x_train_rescaled, y_train_rescaled)


  model = pd.DataFrame({"Month": [i], "lr": [lr], "x_scaler": [x_scaler], "y_scaler": [scaler]})

  return model


lr = pd.DataFrame({"Month": [], "lr": [], "x_scaler": [], "y_scaler": []})
for i in range(1, 13):
  lr = pd.concat([lr, make_model(i)], axis=0)

print("Model2 created")

joblib.dump(lr, 'model2.gz')

