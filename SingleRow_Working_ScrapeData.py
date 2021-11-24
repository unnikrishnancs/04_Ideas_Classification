import requests
from bs4 import BeautifulSoup
import pandas as pd

#Open the page
page=requests.get("https://www.makaan.com/price-trends/property-rates-for-rent-in-bangalore")
print("Response Code ==>",page.status_code)
#print("First 100 characters...\n", page.content[:100])

#parse the data
data=BeautifulSoup(page.text,"lxml")
#print(data.prettify())

#initialize a pandas DataFrame
scraped_data=pd.DataFrame(columns=["locality","minprice","maxprice","avg_rent","house_type"])
print(scraped_data)

tbl=data.tbody
tbl_dtls=tbl.find("tr",itemtype="http://schema.org/Place")
#print(tbl_dtls)
i=1
for col in tbl_dtls:	
	if i==1:
		print("locality", col.a.text)
		print()
	if i==2:
		print("1BHK minPrice",col.find("span",itemprop="minPrice").text)
		print("1BHK maxPrice",col.find("span",itemprop="maxPrice").text)
		#convert to float
	if i==3:
		print("1BHK avg rent",col.text)
		#convert to float
		print()
	if i==4:
		print("2BHK minPrice",col.find("span",itemprop="minPrice").text)
		print("2BHK maxPrice",col.find("span",itemprop="maxPrice").text)
		#convert to float
	if i==5:
		print("2BHK avg rent",col.text)
		#convert to float
		print()
	if i==6:
		print("3BHK minPrice",col.find("span",itemprop="minPrice").text)
		print("3BHK maxPrice",col.find("span",itemprop="maxPrice").text)
		#convert to float
	if i==7:
		print("3BHK avg rent",col.text)
		#convert to float
		print()
		
	#print(col.a.span.text)
	'''
	rent=col.find_all("td",class_="caps ta-c")
	print(rent)
	for r in rent:
		print(rent.text)
	break	
	'''	
	i=i+1
	

#print(name.find("span",itemprop="name").text)


'''
#dtls=data.find(name="table",id="locality_apartment")
#Locality Name
#print(dtls.tbody.tr.td.a.text)
locality=data.find_all("td",class_="ta-l link-td-txt")
for l in locality:
	#print(l.a.text)
	scraped_data["locality"]=l.a.text	

#1 BHk / 2 BHK / 3 BHK Avg rent
#print(dtls.find("td",class_="caps ta-c").text)

avg_rent=dtls.find_all("td",class_="caps ta-c")
for a in avg_rent:
	print(a.text)

print(scraped_data.head())


#summary=article.find("div",class_="entry-content").p.text
'''



