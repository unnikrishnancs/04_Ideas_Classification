from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import numpy as np
from UtilityFunctions import *

#to display all rows and cols
#pd.set_option('display.max_columns',10)
pd.set_option('display.max_rows',None)


#$$$$$$$$$$$$$$$$$$$$$$$$
#import data from csv file
#$$$$$$$$$$$$$$$$$$$$$$$$$

data=pd.read_csv("bangalore_rentdtls_updt.csv",na_values=["-"])
#print(data.head())
data.info() #column data types and non-null values
print()
print("-------------count of null values in each column---------------")
print(data.isnull().sum())
#print(data.describe()) #summary of numeric columans

#$$$$$$$$$$$$$$$$$$$$$$$$
#plot data (diff. dataset used)
#$$$$$$$$$$$$$$$$$$$$$$$$

#plot_data()



#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#-------PRE-PROCESSING------------
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

#print("--------------data_newdf...after pre-processing (except MinPrice,MaxPrice,AvgRent---------------")
data_features_final,data_bkup=data_preprocessing(data)
print(data_features_final)
print()

'''
#to make it concat friendly
print("--------df_price...Price details after re-indexing------")
df_price=data[["MinPrice","MaxPrice","AvgRent"]]
df_price=df_price.reset_index(drop=True) # see what happens if you dont use this
print(df_price)
print()

#join input features
print("---------------Input features concatenated---------------")
data_features_final=pd.concat([data_newdf,df_price],axis=1)
print(data_features_final)
print()
'''


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

#use LabelEncoder
data_out=convert_labels_to_num(data_bkup,"LE")

#use LabelBinarizer
#data_out=convert_labels_to_num(data,"LB")


#===========================================================
#join all the columsn into one (i:e input features + class))
#===========================================================

print("-------------Final dataset before training----------------")

#If LabelEncoder used
data_final=pd.concat([data_features_final,pd.Series(data_out)],axis=1) 
print(data_final)

#If LabelBinarizer used
#data_final=pd.concat([data_features_final,pd.DataFrame(data_out)],axis=1) # after using LabelBinarizer (DataFrame object)
#print(data_final[:6,:])

#print(data_final.info())
print()


#$$$$$$$$$$$$$$$$$$$$$
#-----Define X and y
#$$$$$$$$$$$$$$$$$$$$$

print("----------------Define X and y---------------")

X=data_final.iloc[:,:-1]
print(X.head())

y=data_final.iloc[:,-1]
print(y[:5])
print()

#print(data_final.describe())
#print(data_final[0].value_counts())
#data_final.to_csv("exported_data_final.csv")

#$$$$$$$$$$$$$$$$$$$$$$$
#-----train model-------
#$$$$$$$$$$$$$$$$$$$$$#$

train_x,test_x,train_y,test_y=train_test_split(X,y,test_size=0.2,random_state=2)
#print(type(train_x),type(train_y))

print("-----------Train the model-------------")
model=LogisticRegression()
#model=SVC()
model.fit(train_x,train_y)

#$$$$$$$$$$$$$$$$
#---predict------
#$$$$$$$$$$$$$$$$

#run prediction on TRAIN data
predict_train_y=model.predict(train_x)

#run prediction on TEST data
predict_test_y=model.predict(test_x)


#$$$$$$$$$$$$$$$$$$$$$$
#------metrics---------
#$$$$$$$$$$$$$$$$$$$$$$

# TRAIN DATA
print("---------------Metrics for TRAIN data------------")
print(confusion_matrix(train_y,predict_train_y))
print()
print("Accuracy Score : %.2f "%accuracy_score(train_y,predict_train_y))
print()

# TEST DATA
print("----------------Metrics for TEST data----------------")
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



