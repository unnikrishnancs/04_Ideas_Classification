from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import numpy as np
from UtilityFunctions import *

#to display all rows and cols
#pd.set_option('display.max_columns',10)
#pd.set_option('display.max_rows',None)

#$$$$$$$$$$$$$$$$$$$$$$$$$
#$$$$$$$$$$$$$$$$$$$$$$$$
#import data from csv file
#$$$$$$$$$$$$$$$$$$$$$$$$$
#$$$$$$$$$$$$$$$$$$$$$$$$$

data=pd.read_csv("bangalore_rentdtls_updt.csv",na_values=["-"])
#print(data.head())
data.info() #column data types and non-null values
#print(data.describe()) #summary of numeric columans

#plot the data......DIFFETENT DATASET USED
#plot_data()


#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#-------------------------PRE-PROCESSING--------------------------
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

data_newdf=data_preprocessing(data)

#to make it concat friendly
df_price=data[["MinPrice","MaxPrice","AvgRent"]]
df_price=df_price.reset_index(drop=True) # see what happens if you dont use this

#join input features
print("Input features concatenated")
data_features_final=pd.concat([data_newdf,df_price],axis=1)
print(data_features_final)
print()


'''
#===================================
#scale the feature vector X (inputs)
#===================================
ss=StandardScaler()
data_features_final=pd.DataFrame(data=ss.fit_transform(data_features_final))
print("After Scaling")
print(data_features_final.head())


#============================================
# Handle categorical values (OUTPUT / TARGET)
#============================================
'''

''' 
print("WITHOUT LabelBinarizer")
le=LabelEncoder()
le.fit(data["HouseType"])
print(le.classes_)
data_out=le.transform(data["HouseType"])
#print(data_out[:6])
print(data_out)
print()


print("WITH LabelBinarizer")
lb=LabelBinarizer()
lb.fit(data["HouseType"])
print(lb.classes_)
data_out=lb.transform(data["HouseType"])
print(type(data_out), data_out[:6])
#print(data_out)
print()
'''




'''
data_out=convert_labels_to_num(data)

#join all the columsn into one (i:e input features + class))
print("Final dataset before training")
#data_final=pd.concat([data_features_final,pd.Series(data_out)],axis=1) # before using LabelBinarizer
data_final=pd.concat([data_features_final,pd.DataFrame(data_out)],axis=1) # after using LabelBinarizer
print(data_final)
#print(data_final.info())
print()
'''






'''
#$$$$$$$$$$$$$$$$$$$$$
#$$$$$$$$$$$$$$$$$$$$$
#-----Define X and y
#$$$$$$$$$$$$$$$$$$$$$
#$$$$$$$$$$$$$$$$$$$$$

X=data_final.iloc[:,:-1]
print(X.head())

y=data_final.iloc[:,-1]
print(y[:5])
print()

#print(data_final.describe())
#print(data_final[0].value_counts())
data_final.to_csv("exported_data_final.csv")

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#--------------train model-----------------
#$$$$$$$$$$$$$$$$$$$$$#$$$$$$$$$$$$$$$$$$$$$
#$$$$$$$$$$$$$$$$$$$$$#$$$$$$$$$$$$$$$$$$$$$

train_x,test_x,train_y,test_y=train_test_split(X,y,test_size=0.2,random_state=2)

print(type(train_x),type(train_y))
#model=LogisticRegression()
model=SVC()
model.fit(train_x,train_y)



#$$$$$$$$$$$$$$$$$$$$$$
#$$$$$$$$$$$$$$$$$$$$$$
#-------predict--------
#$$$$$$$$$$$$$$$$$$$$$$
#$$$$$$$$$$$$$$$$$$$$$$

#print("Test Data")
#print(test_x)
predict_train_y=model.predict(train_x)
predict_test_y=model.predict(test_x)


#$$$$$$$$$$$$$$$$$$$$$$
#$$$$$$$$$$$$$$$$$$$$$$
#------metrics---------
#$$$$$$$$$$$$$$$$$$$$$$
#$$$$$$$$$$$$$$$$$$$$$$

# TRAIN DATA
print("confusion matrix for TRAIN data")
print(confusion_matrix(train_y,predict_train_y))
print()

print("accuracy score for TRAIN data")
print("Accuracy Score : %.2f "%accuracy_score(train_y,predict_train_y))
print()
print()

# TEST DATA
print("confusion matrix for TEST data")
print(confusion_matrix(test_y,predict_test_y))
print()

print("accuracy score for TEST data")
print("Accuracy Score : %.2f "%accuracy_score(test_y,predict_test_y))

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
'''



