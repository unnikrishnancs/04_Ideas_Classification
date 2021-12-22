from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
from UtilityFunctions import *

#to display all rows and cols
#pd.set_option('display.max_columns',10)
#pd.set_option('display.max_rows',None)


#$$$$$$$$$$$$$$$$$$$$$$$$
#import data from csv file
#$$$$$$$$$$$$$$$$$$$$$$$$$

data=pd.read_csv("bangalore_rentdtls_updt.csv",na_values=["-"])

print("-------------Top 5 rows---------------")
print()
print(data.head())
print()
print("-------------row count, no. of cols, column data types, non-null values, distri. of datatypes, memory usage---------------")
print()
data.info()
print()
print("-------------count of null values in each column---------------")
print()
print(data.isnull().sum())
print()



#$$$$$$$$$$$$$$$$$$$$$$$$
#plot data (diff. dataset used)
#$$$$$$$$$$$$$$$$$$$$$$$$

#plot_data()


#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#-------PRE-PROCESSING------------
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

print("--------------data pre-processing (drop nulls, handle categorical val. etc) ---------------")
print()
data_features_final,data_bkup=data_preprocessing(data)
print(data_features_final)
print()
print("--------------(After data pre-processing) df.info ---------------")
print()
data_features_final.info()
print()
print("-------------(After data pre-processing) summary of numeric columns---------------")
print()
print(data_features_final.describe())
print()

'''
#===================================
#scale the feature vector X (inputs)
#===================================
print("------------Scale input features-----------")
data_features_final=scale_features(data_features_final)
print("After Scaling")
print(data_features_final.head())
'''

#============================================
# Convert target class to numeric (OUTPUT / TARGET)
#============================================

print("----------Convert label to numeric---------------")
print()
#use LabelEncoder..returns Series object
data_out=convert_labels_to_num(data_bkup,"LE")

#use LabelBinarizer...returns numpy array
#data_out=convert_labels_to_num(data,"LB")

#===========================================================
#join all the columsn into one (i:e input features + class))
#===========================================================

print("-------------Final dataset before training----------------")
print()

#using LabelEncoder used
data_final=pd.concat([data_features_final,pd.Series(data_out)],axis=1) 
print(data_final)

#using LabelBinarizer used
#data_final=pd.concat([data_features_final,pd.DataFrame(data_out)],axis=1) # after using LabelBinarizer (DataFrame object)
#print(data_final[:6,:])

#print(data_final.info())
print()


#$$$$$$$$$$$$$$$$$$$$$
#-----Define X and y
#$$$$$$$$$$$$$$$$$$$$$

print("----------------Define X and y---------------")
print()

X=data_final.iloc[:,:-1]
print(X.head())
print()

y=data_final.iloc[:,-1]
print(y[:5])
print()

#data_final.to_csv("exported_data_final.csv")

#$$$$$$$$$$$$$$$$$$$$$$$
#-----train model-------
#$$$$$$$$$$$$$$$$$$$$$#$

train_x,test_x,train_y,test_y=train_test_split(X,y,test_size=0.2,random_state=2)

print("-----------Training the model-------------")
model=LogisticRegression()
model.fit(train_x,train_y)
print()

#$$$$$$$$$$$$$$$$
#---predict------
#$$$$$$$$$$$$$$$$

#run prediction on TRAIN data
predict_train_y=model.predict(train_x)

#run prediction on TEST data
predict_test_y=model.predict(test_x)

print("-----------Prediction completed on Train and Test data-------------")
print()


#$$$$$$$$$$$$$$$$$$$$$$
#------metrics---------
#$$$$$$$$$$$$$$$$$$$$$$

# TRAIN DATA
print("---------------Metrics for TRAIN data------------")
print()
print(confusion_matrix(train_y,predict_train_y))
print()
print("Accuracy Score : %.2f "%accuracy_score(train_y,predict_train_y))
print()

# TEST DATA
print("----------------Metrics for TEST data----------------")
print()
print(confusion_matrix(test_y,predict_test_y))
print()
print("Accuracy Score : %.2f "%accuracy_score(test_y,predict_test_y))
print()


#$$$$$$$$$$$$$$$$$$$$$$
#------Predict NEW data---------
#$$$$$$$$$$$$$$$$$$$$$$

# ex. pass predict_NEW_y=model.predict(NEW)
# where NEW will have a row of form "Locality, MinPrice, MaxPrice, AvgRent"...so you have to preprocee this row

#feature scaling (run model WITH / WITHOUT feature scaling)
# 
# ================Logistic Regression =====================
# WITHOUT scaling : train acc - 45 % , test acc - 41 %
# WITH scaling : train acc - 99 % test acc - 41 %
# WITHOUT scaling but WITH labelbinarizer : train acc - 71 % test acc - 67 %
# WITH scaling and WITH labelbinarizer : train acc - 100 % test acc -85 %
# 
# ================ SVC =====================
# WITH scaling and WITH labelbinarizer : train acc - 81 % test acc - 70 %
#
# 

# to do
# ...inverse transofmr test data and see which are new data 
# ...//try to visualize data
# ....modularize code...func to pre-process, train , pass new data (Ex. loc,minrent,max rent) and pre-process and predict

