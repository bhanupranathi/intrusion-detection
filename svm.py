# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

''' np.set_printoptions(threshold=np.inf)
np.set_printoptions(edgeitems=3,infstr='inf',linewidth=75, nanstr='nan', precision=8,suppress=False, threshold=1000, formatter=None)'''

# importing csv file
dataFrame = pd.read_csv('Train_data.csv')

# seperating dependant and independant variables
y = dataFrame.iloc[:, -1].values
x = dataFrame.iloc[:, :41].values



# performing preprocessing if data values are missing
# filling the missing values
'''from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputer = imputer.fit(x[:,1])
x[:,1] = imputer.transform(x[:,1])
x
x[:,1]
protocol=[]
for protocol1 in x[:,1]:
    protocol.append(protocol1)
protocolset=set(protocol)
print(protocol)'''


# Performing LabelEncoding and OneHotEncoding  on Second Column
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(x[:, 1])
onehot_encoder1 = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder1.fit_transform(integer_encoded)



# Creating new DataFrames
df1 = pd.DataFrame({'duration': x[:, 0]})
df2 = pd.DataFrame({'icmp': onehot_encoded[:, 0], 'tcp': onehot_encoded[:, 1], 'udp': onehot_encoded[:, 2]})
df3 = pd.DataFrame({'service': x[:, 2], 'flag': x[:, 3], 'src_bytes': x[:, 4], 'dst_bytes': x[:, 5], 'land': x[:, 6],
                    'wrong_fragment': x[:, 7], 'urgent': x[:, 8], 'hot': x[:, 9], 'num_failed_logins': x[:, 10],
                    'logged_in': x[:, 11], 'num_compromised': x[:, 12], 'root_shell': x[:, 13],
                    'su_attempted': x[:, 14], 'num_root': x[:, 15], 'num_file_creations': x[:, 16],
                    'num_shells': x[:, 17], 'num_access_files': x[:, 18], 'num_outbound_cmds': x[:, 19],
                    'is_host_login': x[:, 20], 'is_guest_login': x[:, 21], 'count': x[:, 22], 'srv_count': x[:, 23],
                    'serror_rate': x[:, 24], 'srv_serror_rate': x[:, 25], 'rerror_rate': x[:, 26],
                    'srv_rerror_rate': x[:, 27], 'same_srv_rate': x[:, 28], 'diff_srv_rate': x[:, 29],
                    'srv_diff_host_rate': x[:, 30], 'dst_host_count': x[:, 31], 'dst_host_srv_count': x[:, 32],
                    'dst_host_same_srv_rate': x[:, 33], 'dst_host_diff_srv_rate': x[:, 34],
                    'dst_host_same_src_port_rate': x[:, 35], 'dst_host_srv_diff_host_rate': x[:, 36],
                    'dst_host_serror_rate': x[:, 37], 'dst_host_srv_serror_rate': x[:, 38],
                    'dst_host_rerror_rate': x[:, 39], 'dst_host_srv_rerror_rate': x[:, 40]})


# Concatenate all data frames into single DataFrame
X = pd.concat([df1, df2, df3], axis=1)

#view data frame
X.head()

# performing splitting of test and train data
from sklearn.model_selection import train_test_split


# specify test_size value for splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.0001, shuffle=False)


# performing FeatureScaling
from sklearn.preprocessing import StandardScaler

sc_x = StandardScaler()
X_train = sc_x.fit_transform(X_train)
X_test = sc_x.fit_transform(X_test)



# Support Vector Machine  Algorithm
from sklearn.svm import SVC


#creating classifier
classifier = SVC(kernel='linear', random_state=0)
classifier.fit(X_train, y_train)

# predicting
predicted = classifier.predict(X_test)

# calculating the Accuracy
from sklearn.metrics import accuracy_score

print('Accuracy Score :', accuracy_score(y_test, predicted))


# to view the ndarray object of numpy module
y_test_df = pd.DataFrame(y_test)
predicted_df = pd.DataFrame(predicted)
