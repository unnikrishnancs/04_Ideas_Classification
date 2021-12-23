from sklearn.metrics import confusion_matrix, accuracy_score
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

#===========================================================
#join all the columsn into one (i:e input features + class))
#===========================================================

print("-------------Final dataset before training----------------")
print()

if method=="LE":
	data_final=pd.concat([data_features_final,pd.Series(data_out)],axis=1) 
	print(data_final.iloc[:6])
elif method=="LB":
	data_final=pd.concat([data_features_final,pd.DataFrame(data_out)],axis=1)
	print(data_final.iloc[:6,:])

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

# TRAIN DATA
print("---------------Metrics for TRAIN data: [ ",msg," ]" )
print()
print(confusion_matrix(train_y,predict_train_y))
print()
print("Accuracy Score : %.2f "%accuracy_score(train_y,predict_train_y))
print()

# TEST DATA
print("---------------Metrics for TEST data: [ ",msg," ]" )
print()
print(confusion_matrix(test_y,predict_test_y))
print()
print("Accuracy Score : %.2f "%accuracy_score(test_y,predict_test_y))
print()


#$$$$$$$$$$$$$$$$$$$$$$
#---Predict NEW data---
#$$$$$$$$$$$$$$$$$$$$$$
# ex. pass predict_NEW_y=model.predict(NEW)
# where NEW will have a row of form "Locality, MinPrice, MaxPrice, AvgRent"...so you have to preprocee this row

try:
	#enter input
	loc=input("Enter locality: ")
	minp=input("Enter Min. Price: ")
	maxp=input("Enter Max. Price: ")
	avg=input("Enter Avg. Rent: ")
	print()
	inp={"loc":loc,"minp":minp,"maxp":maxp,"avg":avg}
	print("You entered -> ",inp)
	
	#pre-process data
	proc,_=data_preprocessing(pd.DataFrame(data=inp.values,columns=["loc","minp","maxp","avg"]))
	print(proc)
	
	#predict the data

except  Exception as ex:
	print("Error occured :", ex)



