import pandas as pd
from sklearn.preprocessing  import OneHotEncoder, LabelEncoder,LabelBinarizer
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

#convert labels/target to numeric
def convert_labels_to_num(data,method="LE"):
	if method=="LB":				
		lb=LabelBinarizer()		
		lb.fit(data["HouseType"])		
		data=lb.transform(data["HouseType"])
	elif method=="LE":		
		le=LabelEncoder()
		le.fit(data["HouseType"])
		data=le.transform(data["HouseType"])
	return data


#import dataset
data=pd.read_csv("rentdtls.csv",na_values=["-"])
print(data.head())

#==========================================
# Data pre-processing
#==========================================

#Handle NaN value
data=data.dropna()
data.info()
data_bkup=data
	
#Handle categorical value (INPUT FEATURES)
ohe=OneHotEncoder()
ohe.fit(data[["Locality"]])
data_new=ohe.transform(data[["Locality"]]).toarray()
data_newdf=pd.DataFrame(data=data_new,columns=ohe.get_feature_names_out())	
#remaining columns
df_price=data[["MinPrice","MaxPrice","AvgRent"]]
#to make it concat friendly		
df_price=df_price.reset_index(drop=True) 

#join input features
inp_feat=pd.concat([data_newdf,df_price],axis=1)

print("Input Features after pre-processing : \n", inp_feat)

# LE for LabelEncoder and LB for LabelBinarizer
label_method="LB"

#Convert label to numeric
if label_method=="LE":
	labels=convert_labels_to_num(data_bkup)
	data_final=pd.concat([inp_feat,pd.Series(labels)],axis=1) 
elif label_method=="LB":
	labels=convert_labels_to_num(data_bkup,"LB")
	data_final=pd.concat([inp_feat,pd.DataFrame(labels)],axis=1)

#Define X and y
if label_method=="LE":
	X=data_final.iloc[:,:-1]
	y=data_final.iloc[:,-1]
elif label_method=="LB":
	X=data_final.iloc[:,:-3]
	y=data_final.iloc[:,58:61]
	
#split data into Train and Test
train_x,test_x,train_y,test_y=train_test_split(X,y,test_size=0.2,random_state=2)

#Train the model
model=SVC()
model.fit(train_x,train_y)

#Predict TEST data
predict_test_y=model.predict(test_x)

#print metrics
print(confusion_matrix(predict_test_y,test_y))
print()
print("Accuracy Score : %.2f "%accuracy_score(predict_test_y,test_y))
print()


