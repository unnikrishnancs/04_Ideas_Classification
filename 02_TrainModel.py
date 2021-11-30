import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing  import OneHotEncoder, LabelEncoder,LabelBinarizer
#from sklearn.model_selection import train_test_model


#import data from csv file
data=pd.read_csv("bangalore_rentdtls_updt.csv",na_values=["-"])
print(data.head())

#column data types and non-null values
data.info()

# ???????? TRY TO FIND SOLN....handle "L" (for Lakhs in "4.5 L") in MinPrice and MaxPrice columsn ....FOR NOW, DELETE IT MANUALLY
#
#drop rows having "L" (lakh) in either minprice, maxprice or avg rent
#print("---&&&&&&&&&&--")
#print(data.query("'*L*' in MaxPrice"))
#print(data["MaxPrice"].str.contains("L"))
#print("---&&&&&&&&&&--")
#
#

#==============
#pre-processing 
#==============

#rent cols to int
for col in data.select_dtypes("object"):
	#if col in ("MinPrice","MaxPrice","AvgRent"):
	if col in ("MinPrice"):
		print(col, data[col].dtype)
		data[col]=data[col].astype(int)

data.info()


#summary of numeric columans
print(data.describe())

#handle outliers

#handle values like "8 L" for 800000

#handle nan values...ANY OTHER WAY TO HANDLE NaNs
data=data.dropna()
data.info()
print()

#========================================================================
#convert categorical to numeric (INPUT FEATURES)...locality and housetype
#========================================================================

# NOTE : *** this conversion to be put into NEW module / class and call it ***



ohe=OneHotEncoder()

#fit
ohe.fit(data[["Locality"]])
print("Identified Categories")
print(ohe.categories_)
print()
print("Feature Names")
print(ohe.get_feature_names_out())
print()

#transform into one hot encoding
data_new=ohe.transform(data[["Locality"]]).toarray()
data_newdf=pd.DataFrame(data=data_new,columns=ohe.get_feature_names_out())
print(data_newdf.head())
print()

#join input features
print("Input features concatenated")
data_features_final=pd.concat([data_newdf,data[["MinPrice","MaxPrice","AvgRent"]]],axis=1)
print(data_features_final.head())

#=================================================
#convert categorical to numeric (LABELS / CLASSES)
#=================================================

le=LabelEncoder()
le.fit(data["HouseType"])
print(le.classes_)
data_out=le.transform(data["HouseType"])
print(data_out[:10])



#.fit(data[["Locality","HouseType"]])

#try with and without LabelBinarizer

#feature scaling (run model WITH / WITHOUT feature scaling)


#join all the columsn into one (i:e input features + class))


#split data into Train and Test set


#train model


#predict


#metrixs



