import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing  import OneHotEncoder, LabelEncoder,LabelBinarizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np


if __name__=="__main__":
	print("This program (UtilityFunctions.py) was called directly from python command prompt: python <file.py>")
	print()
else:
	print("This program (UtilityFunctions.py) was called by another program")
	print()

		
def data_preprocessing(data,feat_scale, label_method):
	msg=""
	
	#Handle NaN value
	data=data.dropna()
	#data.info()
	data_bkup=data # this wiill be used later for converting labels to numeric
	print()
	
	#Handle categorical value (INPUT FEATURES)
	ohe=OneHotEncoder()
	ohe.fit(data[["Locality"]])
	#print("-----------Identified Categories: \n",ohe.categories_,"\n Feature Names : \n",ohe.get_feature_names_out())
	data_new=ohe.transform(data[["Locality"]]).toarray()
	data_newdf=pd.DataFrame(data=data_new,columns=ohe.get_feature_names_out())	
	
	#remaining columns
	df_price=data[["MinPrice","MaxPrice","AvgRent"]]
	#to make it concat friendly		
	df_price=df_price.reset_index(drop=True) 
	
	#join input features
	inp_feat=pd.concat([data_newdf,df_price],axis=1)
	
	if feat_scale=="Y":		
		msg="With Feature Scaling; "
		#print("------------Scale input features-----------")
		#print("Before Scaling")
		#print()
		#print(inp_feat.head())
		inp_feat=scale_features(inp_feat)
		#print("After Scaling")
		#print()
		#print(inp_feat.head())
	else:
		msg="No Feature Scaling; "	
	
	print("----------Convert label to numeric---------------")
	print()
	
	if label_method=="LE":
		#use LabelEncoder..returns Series object
		labels,l2n=convert_labels_to_num(data_bkup)
		msg+="With LabelEncoder; "
	elif label_method=="LB":
		pass
		'''
		#use LabelBinarizer...returns numpy array
		labels,l2n=convert_labels_to_num(data_bkup,"LB")
		msg+="With LabelBinarizer; "
		'''
		
	return inp_feat,labels,msg,l2n
	

#scale the feature vector X (inputs)
def scale_features(data):
	ss=StandardScaler()
	data=pd.DataFrame(data=ss.fit_transform(data),columns=ss.feature_names_in_)
	return data

#convert labels/target to numeric
def convert_labels_to_num(data,method="LE"):
	if method=="LB":
		pass
		'''		
		print("-----------WITH LabelBinarizer------------")
		l2n=LabelBinarizer() #sparse_output=True
		#l2n.fit(data["HouseType"])
		l2n.fit(data["HouseType"])
		print("l2n.y_type_ : ",l2n.y_type_,", l2n.classes_ :",l2n.classes_)
		data=l2n.transform(data["HouseType"])
		print(type(data), "\n",data[:6,:])
		#print(data)	
		'''	
	elif method=="LE":
		print("-----------WITH LabelEncoder-------------")
		l2n=LabelEncoder()
		l2n.fit(data["HouseType"])
		print(l2n.classes_)
		data=l2n.transform(data["HouseType"])
		print(type(data), data[:6])
		
	print()
	return data,l2n

#plot the data	...DIFFETENT DATASET USED
def plot_data():	
	data=pd.read_csv("ForPlot_bangalore_rentdtls_updt.csv",na_values=["-"])
	loc=data.iloc[:,[0,3,4]]
	print(type(loc),loc)
	plt.scatter(loc["Locality"], loc["AvgRent"], c=loc["HouseType"])
	plt.legend()
	plt.xlabel("Locality ->")
	plt.ylabel("Avg. Rent ->")
	plt.xticks(rotation=45)
	plt.title("Average Rent across localities in Bangalore (1BHK, 2BHK, 3BHK)")
	plt.tight_layout()
	plt.show() 
	

#function to predict new data
def predict_newdata(model,loc,min_p,max_p,avg,l2n):
	try:
		# initialize one-hot vector for "Locality" (all 0s)
		inp={'Locality_BTM Layout':0, 'Locality_Bagaluru Near Yelahanka':0,
		'Locality_Banashankari':0, 'Locality_Banaswadi':0, 'Locality_Battarahalli':0,
		'Locality_Begur':0, 'Locality_Bellandur':0, 'Locality_Bommanahalli':0,
		'Locality_Brookefield':0, 'Locality_Budigere Cross':0,
		'Locality_CV Raman Nagar':0, 'Locality_Chandapura':0,
		'Locality_Dasarahalli on Tumkur Road':0,
		'Locality_Electronic City Phase 1':0, 'Locality_Electronics City':0,
		'Locality_Gottigere':0, 'Locality_HSR Layout':0, 'Locality_Harlur':0,
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
		'Locality_Yelahanka':0}        
		
		#pass NEW data for prediction...in this case ---------- locality='Yelahanka', MinPrice=15000, MaxPrice=25000, AvgRent=20000 ----------
		inp["Locality_"+loc]=1	
		inp["MinPrice"]=min_p
		inp["MaxPrice"]=max_p
		inp["AvgRent"]=avg 
		#print(inp)
		#print()
		   	
		#create datafrane
		inp_df=pd.DataFrame(data=[inp])
		print(inp_df)
		print()
		
		#predict the data
		#print("House Type Prediction (locality=",loc,", min_price=",min_p,", max_price=",max_p,", avg_rent=", avg,") :",model.predict(inp_df))
		print("House Type Prediction (locality=",loc,", min_price=",min_p,", max_price=",max_p,", avg_rent=", avg,") :",l2n.inverse_transform(model.predict(inp_df)))
		print()

	except  Exception as ex:
		print("Error occured :", ex)
		print()
	
	

def calc_metrics(type, orig,pred,msg):
	if type=="Train":
		# TRAIN DATA
		print("---------------Metrics for TRAIN data: [ msg= ",msg," ]" )
		print("Adjust the 'feat_scale' and 'label_method' variable for other options (scaling, labelencoder vs labelbinarizer)")
		print()
		print(confusion_matrix(orig,pred))
		print()
		print("Accuracy Score : %.2f "%accuracy_score(orig,pred))
		print()

	if type=="Test":
		# TEST DATA
		print("---------------Metrics for TEST data: [ msg= ",msg," ]" )
		print("Adjust the 'feat_scale' and 'label_method' variable for other options (scaling, labelencoder vs labelbinarizer)")
		print()
		print(confusion_matrix(orig,pred))
		print()
		print("Accuracy Score : %.2f "%accuracy_score(orig,pred))
		print()

