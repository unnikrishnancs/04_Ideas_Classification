import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing  import OneHotEncoder, LabelEncoder,LabelBinarizer


if __name__=="__main__":
	print("This program was called directly from python command prompt: python <file.py>")
else:
	print("This program was called by another program")

def predict_newdata():
	#clean data
	#predict data
	#return <result>
	pass
	
		
def data_preprocessing(data):
	print("handle nan values...ANY OTHER WAY TO HANDLE NaNs")	
	data=data.dropna()
	data.info()
	print()
	
	print("Handle categorical value (INPUT FEATURES)")
	ohe=OneHotEncoder()
	ohe.fit(data[["Locality"]])
	print("Identified Categories: \n",ohe.categories_,"\n Feature Names : \n",ohe.get_feature_names_out())
	print()	
	data_new=ohe.transform(data[["Locality"]]).toarray()
	data_newdf=pd.DataFrame(data=data_new,columns=ohe.get_feature_names_out())
	print(data_newdf)
	print()
	return data_newdf
	

#Input Features....convert categorical to numeric
def convert_catfeat_to_num():
	pass


#convert labels/target to numeric
def convert_labels_to_num(data):
	print("WITH LabelBinarizer")
	lb=LabelBinarizer()
	lb.fit(data["HouseType"])
	print(lb.classes_)
	data_out=lb.transform(data["HouseType"])
	print(type(data_out), data_out[:6])
	#print(data_out)
	print()
	return 

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
	
