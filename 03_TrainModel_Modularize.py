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

try:
	inp={'Locality_BTM Layout':0, 'Locality_Bagaluru Near Yelahanka':0,
       'Locality_Banashankari':0, 'Locality_Banaswadi':0, 'Locality_Battarahalli':0,
       'Locality_Begur':0, 'Locality_Bellandur':0, 'Locality_Bommanahalli':0,
       'Locality_Brookefield':0, 'Locality_Budigere Cross':0,
       'Locality_CV Raman Nagar':0, 'Locality_Chandapura':0,
       'Locality_Dasarahalli on Tumkur Road':0,
       'Locality_Electronic City Phase 1':0, 'Locality_Electronics City':0,
       'Locality_Gottigere':0, 'Locality_HSR Layout':1, 'Locality_Harlur':0,
       'Locality_Hebbal':0, 'Locality_Hennur':0, 'Locality_Horamavu':0,
       'Locality_Hosa Road':0, 'Locality_Hoskote':0, 'Locality_Hulimavu':0,
       'Locality_Indira Nagar':0, 'Locality_J. P. Nagar':0,
       'Locality_JP Nagar Phase 7':0, 'Locality_Jakkur':0, 'Locality_Jayanagar':0,
       'Locality_Jigani':0, 'Locality_Kalyan Nagar':0,
       'Locality_Kannur on Thanisandra Main Road':0, 'Locality_Kasavanahalli':0,
       'Locality_Koramangala':0, 'Locality_Krishnarajapura':0,
       'Locality_Kumbalgodu':0, 'Locality_Mahadevapura':0, 'Locality_Marathahalli':0,
       'Locality_Marsur':0, 'Locality_Murugeshpalya':0, 'Locality_Nagarbhavi':0,
       'Locality_Narayanapura on Hennur Main Road':0, 'Locality_RR Nagar':0,
       'Locality_Rajajinagar':0, 'Locality_Ramamurthy Nagar':0,
       'Locality_Sarjapur':0, 'Locality_Sarjapur Road Post Railway Crossing':0,
       'Locality_Subramanyapura':0, 'Locality_Talaghattapura':0,
       'Locality_Thanisandra':0, 'Locality_Varthur':0, 'Locality_Vidyaranyapura':0,
       'Locality_Whitefield':0, 'Locality_Whitefield Hope Farm Junction':0,
       'Locality_Yelahanka':0, 'MinPrice':15000, 'MaxPrice':25000, 'AvgRent':20000}       
	
	#create datafrane
	inp_df=pd.DataFrame(data=[inp])
	print(inp_df)
	
	#predict the data
	print("House Type Prediction -> :",model.predict(inp_df))

except  Exception as ex:
	print("Error occured :", ex)



