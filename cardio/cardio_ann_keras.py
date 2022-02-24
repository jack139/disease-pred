import time
import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from keras.layers import Dense, BatchNormalization, Dropout, LSTM
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras import callbacks
from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report, accuracy_score, f1_score


def check_zipcode(code):
    for i in zipcode.keys():
        if code >= int(zipcode[i][0]) and code <= int(zipcode[i][1]):
            return i
    return ''

#loading data
data = pd.read_csv("../datasets/cardio.csv")
#data.head()
#data.info()
#data.describe().T

# 暂不处理 zipcode
data.drop(["id", "age_days"],axis=1,inplace=True)

# Removing Outliers
#data = data[data.height <= 200]
#data = data[data.height >= 120]
#
#data = data[data.weight <= 160]
#
#data = data[data.ap_hi.between(0,500)]
#
#data = data[data.ap_lo.between(0,2000)]


#assigning values to features as X and target as y
X=data.drop(["cardio"],axis=1)
y=data["cardio"]


#Set up a standard scaler for the features
col_names = list(X.columns)
s_scaler = preprocessing.StandardScaler()
X_df= s_scaler.fit_transform(X)
X_df = pd.DataFrame(X_df, columns=col_names)   
X_df.describe().T

#spliting test and training sets
X_train, X_test, y_train, y_test = train_test_split(X_df.values, y, test_size=0.3, random_state=7)


if __name__ == '__main__':

    #MODEL BUILDING

    early_stopping = callbacks.EarlyStopping(
        min_delta=0.001, # minimium amount of change to count as an improvement
        patience=20, # how many epochs to wait before stopping
        restore_best_weights=True)

    # Initialising the NN
    model = Sequential()

    # layers
    model.add(Dense(units = 64, activation = 'relu', input_dim = 11))
    model.add(BatchNormalization())
    model.add(Dense(units = 32, activation = 'relu'))
    model.add(BatchNormalization())
    model.add(Dense(units = 16, activation = 'relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(Dense(units = 1, activation = 'sigmoid'))

    # Compiling the ANN
    model.compile(
        optimizer = Adam(learning_rate=1e-4),
        loss = 'binary_crossentropy', 
        metrics = ['accuracy']
    )

    # Train the ANN
    history = model.fit(X_train, y_train, 
        batch_size = 128, 
        epochs = 200,
        callbacks=[early_stopping], 
        validation_split=0.2
    )

    # valuated result
    val_accuracy = np.mean(history.history['val_accuracy'])
    print("\n%s: %.2f%%" % ('val_accuracy', val_accuracy*100))

    # Predicting the test set results
    y_pred = model.predict(X_test)
    y_pred = (y_pred > 0.5)
    np.set_printoptions()
    print(classification_report(y_test, y_pred))
