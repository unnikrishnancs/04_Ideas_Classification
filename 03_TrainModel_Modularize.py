
from sklearn.model_selection import train_test_split
import numpy as np
from UtilityFunctions import *
from sklearn.svm import SVC
import sys

#to display all rows and cols
#pd.set_option('display.max_columns',10)
#pd.set_option('display.max_rows',None)



#$$$$$$$$$$$$$$$$$$$$$$$$
#import data from csv file
#$$$$$$$$$$$$$$$$$$$$$$$$$

data=pd.read_csv("data/bangalore_rentdtls_updt.csv",na_values=["-"])

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

feat_scale="N" # 'Y' for feature scaling and 'N' to avoid feature scaling
label_method="LE" # 'LE' for LabelEncoding and 'LB' for LabelBinarizer

if label_method=="LB":
	print("LabelBinarizer not working as expected currently...try with LabelEncoder (LE)")
	print()
	sys.exit()

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#-------PRE-PROCESSING------------
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

inp_feat,labels,msg,l2n=data_preprocessing(data, feat_scale, label_method)

#join all the columsn into one (i:e input features + class))

print("-------------Final dataset before training----------------")
print()

if label_method=="LE":
	data_final=pd.concat([inp_feat,pd.Series(labels)],axis=1) 
	print(data_final.iloc[:6])
elif label_method=="LB":
	pass
	'''
	data_final=pd.concat([inp_feat,pd.DataFrame(labels)],axis=1)
	#print(data_final.iloc[:6,:])
	print(data_final.iloc[:6])
	'''

print()
print("--------------(After data pre-processing) data_final.info ---------------")
print()
data_final.info()
print()
print("-------------(After data pre-processing) summary of numeric columns---------------")
print()
print(data_final.describe())
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
	pass	
	'''
	print("X (first 5 rows)")
	X=data_final.iloc[:,:-3]
	print(X.head())
	print()

	print("y (first 5 rows)...label binarizer")
	y=data_final.iloc[:,58:61]
	print(y[:5])
	print()
	'''

#data_final.to_csv("exported_data_final.csv")
	

#$$$$$$$$$$$$$$$$$$$$$$$
#-----train model-------
#$$$$$$$$$$$$$$$$$$$$$#$

print("-----------Training the model-------------")

#split dataset into Train and Test
train_x,test_x,train_y,test_y=train_test_split(X,y,test_size=0.2,random_state=2)

#initialize SVM model for classification
model=SVC()

#train SVM on train set
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

loca="Kalyan Nagar"

'''
#2BHK
min_price=8500
max_price=20000
avg_rent=11642.86

#1BHK
min_price=5000
max_price=14000
avg_rent=9000
'''

#3BHK
min_price=20000
max_price=45000
avg_rent=35000

min_price=20000
max_price=30000
avg_rent=0

print("-----------New Data to be predicted--------- \n")
predict_newdata(model,loca,min_price,max_price,avg_rent,l2n)


