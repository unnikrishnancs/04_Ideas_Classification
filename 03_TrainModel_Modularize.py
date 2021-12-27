
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
from UtilityFunctions import *
from sklearn.svm import SVC

#to display all rows and cols
#pd.set_option('display.max_columns',10)
#pd.set_option('display.max_rows',None)


#$$$$$$$$$$$$$$$$$$$$$$$$
#import data from csv file
#$$$$$$$$$$$$$$$$$$$$$$$$$

data=pd.read_csv("__data__/bangalore_rentdtls_updt.csv",na_values=["-"])

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

#msg=""
#msg="With Feature Scaling; With LabelEncoder"
#msg="With Feature Scaling; With LabelBinarizer"
feat_scale="N" # 'Y' for feature scaling and 'N' to avoid feature scaling
label_method="LB" # 'LE' for LabelEncoding and 'LB' for LabelBinarizer
# LB not working ...wip

inp_feat,labels,msg=data_preprocessing(data, feat_scale, label_method)


'''
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

print(data_features_final.columns)

msg=""
#msg="With Feature Scaling; "

if msg=="With Feature Scaling; ":
	#===================================
	#scale the feature vector X (inputs)
	#===================================
	print("------------Scale input features-----------")
	data_features_final=scale_features(data_features_final)
	print("After Scaling")
	print()
	print(data_features_final.head())
else:
	msg="No Feature Scaling; "

#============================================
# Convert target class to numeric (OUTPUT / TARGET)
#============================================

print("----------Convert label to numeric---------------")
print()
method="LE"
if method=="LE":
	#use LabelEncoder..returns Series object
	data_out=convert_labels_to_num(data_bkup)
	msg+="With LabelEncoder; "
elif method=="LB":
	#use LabelBinarizer...returns numpy array
	data_out=convert_labels_to_num(data_bkup,"LB")
	msg+="With LabelBinarizer; "
'''


#===========================================================
#join all the columsn into one (i:e input features + class))
#===========================================================

print("-------------Final dataset before training----------------")
print()

if label_method=="LE":
	data_final=pd.concat([inp_feat,pd.Series(labels)],axis=1) 
	print(data_final.iloc[:6])
elif label_method=="LB":
	data_final=pd.concat([inp_feat,pd.DataFrame(labels)],axis=1)
	#print(data_final.iloc[:6,:])
	print(data_final.iloc[:6])

#print(data_final.info())
print()

#$$$$$$$$$$$$$$$$$$$$$
#-----Define X and y
#$$$$$$$$$$$$$$$$$$$$$

print("----------------Define X and y---------------")
print()

if label_method=="LE":
	print("X (first 5 rows)")
	X=data_final.iloc[:,:-1]
	print(X.head())
	print()

	print("y (first 5 rows)...label encoded")
	y=data_final.iloc[:,-1]
	print(y[:5])
	print()
elif label_method=="LB":
	print("X (first 5 rows)")
	X=data_final.iloc[:,:-3]
	print(X.head())
	print()

	print("y (first 5 rows)...label binarizer")
	y=data_final.iloc[:,58:61]
	print(y[:5])
	print()

#data_final.to_csv("exported_data_final.csv")

#$$$$$$$$$$$$$$$$$$$$$$$
#-----train model-------
#$$$$$$$$$$$$$$$$$$$$$#$

train_x,test_x,train_y,test_y=train_test_split(X,y,test_size=0.2,random_state=2)

print("-----------Training the model-------------")

#LogisticRegression
#model=LogisticRegression()

#svm
model=SVC()

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

calc_metrics("Train", train_y, predict_train_y, msg)

calc_metrics("Test", test_y, predict_test_y, msg)


#$$$$$$$$$$$$$$$$$$$$$$
#---Predict NEW data---
#$$$$$$$$$$$$$$$$$$$$$$

loca="Yelahanka"
min_price=15000
max_price=25000
avg_rent=20000

print("-----------New Data to be predicted--------- \n")
predict_newdata(model,loca,min_price,max_price,avg_rent)


