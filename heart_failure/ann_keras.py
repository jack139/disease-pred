# import liberaries
import pandas as pd
import numpy as np  
import keras

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score

np.random.seed(0)


df_heart = pd.read_csv('../datasets/heart_failure_clinical_records_dataset.csv')

df_heart = df_heart[df_heart['ejection_fraction']<70]

# 只使用了 ejection_fraction, serum_creatinine, time
x = df_heart.iloc[:, [4,7,11]].values
y = df_heart.iloc[:,-1].values

# Splitting the dataset into training set and test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state =0)

# Feature Scaling
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


# Initialising the ANN
ann = keras.models.Sequential()
# Adding the input layer and the first hidden layer
ann.add(keras.layers.Dense(units = 7, activation = 'relu'))
# Adding the second hidden layer
ann.add(keras.layers.Dense(units = 7, activation = 'relu'))
# Adding the third hidden layer
ann.add(keras.layers.Dense(units = 7, activation = 'relu'))
# Adding the fourth hidden layer
ann.add(keras.layers.Dense(units = 7, activation = 'relu'))
 #Adding the output layer
ann.add(keras.layers.Dense(units = 1, activation = 'sigmoid'))

# Compiling the ANN
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy' , metrics = ['accuracy'] )

# Training the ANN on the training set
ann.fit(x_train, y_train, batch_size = 32, epochs = 100)


# Predicting the test set results
y_pred = ann.predict(x_test)
y_pred = (y_pred > 0.5)
np.set_printoptions()

# Making the confusion matrix, calculating accuracy_score 

# confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix")
print(cm)
print()

# accuracy
ac = accuracy_score(y_test, y_pred)
print("Accuracy")
print(ac)
