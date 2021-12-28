import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np

#Open the page
#page=requests.get("https://www.makaan.com/price-trends/property-rates-for-rent-in-bangalore")
page=requests.get("https://www.makaan.com/price-trends/property-rates-for-rent-in-bangalore?page=6")

print("Response Code ==>",page.status_code)
#print("First 100 characters...\n", page.content[:100])

#parse the data
data=BeautifulSoup(page.text,"lxml")
#print(data.prettify())

tbl=data.tbody
localities=tbl.find_all("tr",itemtype="http://schema.org/Place")
#print(localities)

l_loca=list()
l_minprice=list()
l_maxprice=list()
l_avgrent=list()
l_housetype=list()

for loc in localities:
	i=1
	for col in loc:	
		if i==1:
			try:
				locality=col.a.text
			except AttributeError:
				locality="nan"
			except :
				print("'locality' something else went wrong")
			
			print("locality", locality)
			print()
		if i==2:
			try:
				#Ex. convert "5,000" to 5000
				BHK1_minPrice=(col.find("span",itemprop="minPrice").text).replace(",","")
			except AttributeError:
				BHK1_minPrice="nan"	
			except:
				print("'BHK1_minPrice' something else went wrong")
				
			print("1BHK minPrice",BHK1_minPrice)	
			
			try:	
				BHK1_maxPrice=(col.find("span",itemprop="maxPrice").text).replace(",","")
			except AttributeError:
				BHK1_maxPrice="nan"
			except:
				print("'BHK1_maxPrice' something else went wrong")
			print("1BHK maxPrice",BHK1_maxPrice)
			#convert to float
		if i==3:
			try:
				BHK1_avgrent=((col.text).strip()).replace(",","")
			except AttributeError:
				BHK1_avgrent="nan"
			except :
				print("'BHK1_avgrent' something else went wrong")
				
			print("1BHK avg rent",BHK1_avgrent)
			#convert to float
			print()
		if i==4:
			try:
				BHK2_minPrice=(col.find("span",itemprop="minPrice").text).replace(",","")
			except AttributeError:
				BHK2_minPrice="nan"
			except :
				print("'BHK2_minPrice' something else went wrong")
				
			print("2BHK minPrice",BHK2_minPrice)
			
			try:	
				BHK2_maxPrice=(col.find("span",itemprop="maxPrice").text).replace(",","")
			except AttributeError:
				BHK2_maxPrice="nan"
			except:
				print("'BHK2_maxPrice' something else went wrong")
				
			print("2BHK maxPrice",BHK2_maxPrice)
			#convert to float
		if i==5:
			try:
				BHK2_avgrent=((col.text).strip()).replace(",","")
			except AttributeError:
				BHK2_avgrent="nan"
			except :
				print("'BHK2_avgrent' something else went wrong")	
				
			print("2BHK avg rent",BHK2_avgrent)
			#convert to float
			print()
		if i==6:
			try:
				BHK3_minPrice=(col.find("span",itemprop="minPrice").text).replace(",","")
			except AttributeError:
				BHK3_minPrice="nan"
			except :
				print("'BHK3_minPrice' something else went wrong")
			
			print("3BHK minPrice",BHK3_minPrice)
			
			try:
				BHK3_maxPrice=(col.find("span",itemprop="maxPrice").text).replace(",","")
			except AttributeError:
				BHK3_maxPrice="nan"
			except:
				print("'BHK3_maxPrice' something else went wrong")
			
			print("3BHK maxPrice",BHK3_maxPrice)
			#convert to float
		if i==7:
			try:
				BHK3_avgrent=((col.text).strip()).replace(",","")
			except AttributeError:
				BHK3_avgrent="nan"
			except :
				print("'BHK3_avgrent' something else went wrong")
			
			print("3BHK avg rent",BHK3_avgrent)
			#convert to float
			print()
		
		i=i+1
		
	
	l_loca.append(locality)
	l_loca.append(locality)
	l_loca.append(locality)
	
	l_minprice.append(BHK1_minPrice)
	l_minprice.append(BHK2_minPrice)
	l_minprice.append(BHK3_minPrice)
	
	l_maxprice.append(BHK1_maxPrice)
	l_maxprice.append(BHK2_maxPrice)
	l_maxprice.append(BHK3_maxPrice)
	
	l_avgrent.append(BHK1_avgrent)
	l_avgrent.append(BHK2_avgrent)
	l_avgrent.append(BHK3_avgrent)
	
	l_housetype.append("1BHK")
	l_housetype.append("2BHK")
	l_housetype.append("3BHK")
	

#create series object from list
loca_series=pd.Series(l_loca)
minprice_series=pd.Series(l_minprice)
maxprice_series=pd.Series(l_maxprice)
avgrent_series=pd.Series(l_avgrent)
housetype_series=pd.Series(l_housetype)

#create Dataframe from series object
df=pd.concat([loca_series,minprice_series,maxprice_series,avgrent_series,housetype_series],axis=1)
df.columns=["Locality","MinPrice","MaxPrice","AvgRent","HouseType"]
print(type(df),"\n",df)

#convert to CSV file
df.to_csv("bangalore_rentdtls_pg6.csv",index=False)



