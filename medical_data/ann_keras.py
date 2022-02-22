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

from uszip import zipcode

def check_zipcode(code):
    for i in zipcode.keys():
        if code >= int(zipcode[i][0]) and code <= int(zipcode[i][1]):
            return i
    return ''

#loading data
data = pd.read_csv("../datasets/Medical_data.csv")
#data.head()
#data.info()
#data.describe().T

# 暂不处理 zipcode
data.drop(["id", 
    #"zipcode",
    #"employment_status",
    #"education",
    #"ancestry",
    #"military_service",
],axis=1,inplace=True)

#data = data[data['ejection_fraction']<70]

#assigning values to features as X and target as y
X=data.drop(["disease"],axis=1)
y=data["disease"]

# 转换zip
X['zipcode'] = X['zipcode'].apply(check_zipcode)
l_zipcode = preprocessing.LabelEncoder()
X['zipcode']=l_zipcode.fit_transform(X['zipcode'])

# 生日转为年龄
X['dob'] = time.localtime().tm_year - pd.DatetimeIndex(X['dob']).year

#转换 字符 转 数字 label
l_gender = preprocessing.LabelEncoder()
X['gender']=l_gender.fit_transform(X['gender'])

l_employment_status = preprocessing.LabelEncoder()
X['employment_status']=l_employment_status.fit_transform(X['employment_status'])

l_education = preprocessing.LabelEncoder()
X['education']=l_education.fit_transform(X['education'])

l_marital_status = preprocessing.LabelEncoder()
X['marital_status']=l_marital_status.fit_transform(X['marital_status'])

l_ancestry = preprocessing.LabelEncoder()
X['ancestry']=l_ancestry.fit_transform(X['ancestry'])

l_military_service = preprocessing.LabelEncoder()
X['military_service']=l_military_service.fit_transform(X['military_service'])

# 转换 y 的 label
l_disease = preprocessing.LabelEncoder()
y=l_disease.fit_transform(y)


#Set up a standard scaler for the features
col_names = list(X.columns)
s_scaler = preprocessing.StandardScaler()
X_df= s_scaler.fit_transform(X)
X_df = pd.DataFrame(X_df, columns=col_names)   
X_df.describe().T

#spliting test and training sets
#X_train, X_test, y_train, y_test = train_test_split(X_df.values, y, test_size=0.2, random_state=7)
X_train, y_train = X_df.values, y


if __name__ == '__main__':

    #MODEL BUILDING

    early_stopping = callbacks.EarlyStopping(
        min_delta=0.001, # minimium amount of change to count as an improvement
        patience=20, # how many epochs to wait before stopping
        restore_best_weights=True)

    # Initialising the NN
    model = Sequential()

    # layers
    model.add(Dense(units = 256, activation = 'relu', input_dim = 12))
    model.add(Dense(units = 128, activation = 'relu'))
    #model.add(Dropout(0.25))
    model.add(Dense(units = 64, activation = 'relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(units = 13, activation = 'softmax'))

    # Compiling the ANN
    model.compile(
        optimizer = Adam(learning_rate=1e-2), 
        loss = 'sparse_categorical_crossentropy', 
        metrics = ['sparse_categorical_accuracy']
    )

    # Train the ANN
    history = model.fit(X_train, y_train, 
        batch_size = 32, 
        epochs = 100,
        callbacks=[early_stopping], 
        validation_split=0.1
    )

    # valuated result
    val_accuracy = np.mean(history.history['val_sparse_categorical_accuracy'])
    print("\n%s: %.2f%%" % ('val_accuracy', val_accuracy*100))


    # Predicting the test set results
    #y_pred = model.predict(X_test)
    #y_pred = np.argmax(y_pred, axis=1)
    #np.set_printoptions()
    #print(classification_report(y_test, y_pred))
