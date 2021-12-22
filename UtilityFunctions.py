import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing  import OneHotEncoder, LabelEncoder,LabelBinarizer
from sklearn.preprocessing import StandardScaler


if __name__=="__main__":
	print("This program (UtilityFunctions.py) was called directly from python command prompt: python <file.py>")
	print()
else:
	print("This program (UtilityFunctions.py) was called by another program")
	print()

def predict_newdata():
	#clean data
	#predict data
	#return <result>
	pass
	
		
def data_preprocessing(data):
	#Handle NaN value
	data=data.dropna()
	data.info()
	data_bkup=data # this wiill be used later for converting labels to numeric
	print()
	
	#Handle categorical value (INPUT FEATURES)
	ohe=OneHotEncoder()
	ohe.fit(data[["Locality"]])
	#print("-----------Identified Categories: \n",ohe.categories_,"\n Feature Names : \n",ohe.get_feature_names_out())
	data_new=ohe.transform(data[["Locality"]]).toarray()
	data_newdf=pd.DataFrame(data=data_new,columns=ohe.get_feature_names_out())	
	
	#to make it concat friendly
	#print("--------df_price...Price details after re-indexing------")
	df_price=data[["MinPrice","MaxPrice","AvgRent"]]
	df_price=df_price.reset_index(drop=True) # see what happens if you dont use this

	#join input features
	data_features_final=pd.concat([data_newdf,df_price],axis=1)		
	return data_features_final,data_bkup
	

def scale_features(data):
	ss=StandardScaler()
	data=pd.DataFrame(data=ss.fit_transform(data))
	return data

#convert labels/target to numeric
def convert_labels_to_num(data,method="LB"):
	if method=="LB":
		print("-----------WITH LabelBinarizer------------")
		lb=LabelBinarizer()
		lb.fit(data["HouseType"])
		print(lb.classes_)
		data=lb.transform(data["HouseType"])
		print(type(data), data[:6])
		#print(data)
	else :
		print("-----------WITH LabelEncoder-------------")
		le=LabelEncoder()
		le.fit(data["HouseType"])
		print(le.classes_)
		data=le.transform(data["HouseType"])
		print(type(data), data[:6])
		
	print()
	return data

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
	
	

