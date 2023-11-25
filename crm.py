import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt

pd.set_option("display.max_columns", None)
#pd.set_option("display.width", 500)
#pd.set_option("display.max_rows", None)
pd.set_option("display.float_format",lambda x : "%.3f" % x)


df_= pd.read_excel(r"C:\Users\bett0\Desktop\online_retail_II.xlsx",sheet_name="Year 2009-2010")
df = df_.copy()

#veri hazırlama
print(df.head(5))
df.isnull().sum() #eksik değerler tespit edilir.
df.dropna(inplace=True) #verideki bütün eksik gözler silinir.
#df.drop(df[df["Description"].isna()].index, axis=0, inplace=True) #Description NaN olan satırları çıkardım
#df.dropna(subset=["Customer ID"], axis=0, inplace=True)

print(df.describe().T) #Quantityde negatif değerler var
df = df[~df["Invoice"].str.contains("C",na = False)]

# RFM Metriklerini hesaplama
df["TotalPrice"] = df["Quantity"] * df["Price"]
df["InvoiceDate"].max()
today_date = dt.datetime(2010,12,11)
rfm = df.groupby("Customer ID").agg({"InvoiceDate":lambda InvoiceDate : (today_date - InvoiceDate.max()).days,
                                     "Invoice": lambda Invoice : Invoice.nunique(),
                                    "TotalPrice": lambda TotalPrice: TotalPrice.sum()})
rfm.columns = ["recency","frequency","monetary"]
rfm = rfm[rfm["monetary"] > 0]

"""
print(df["TotalPrice"])
print(df.groupby("Invoice").agg({"TotalPrice":"sum"}).head())

"""
#rfm skor hesabı

rfm["recency_score"] = pd.qcut(rfm["recency"],5,labels = [5,4,3,2,1])

rfm["monetary_score"] = pd.qcut(rfm["monetary"],5,labels = [1,2,3,4,5])

rfm["frequency_score"] = pd.qcut(rfm["frequency"].rank(method="first"),5,labels = [1,2,3,4,5])

rfm["RFM_SCORE"] = (rfm["recency_score"].astype(str) + rfm["frequency_score"].astype(str))

print(rfm[rfm["RFM_SCORE"] == "55"])
print(rfm)

#segmentasyon (regex)

seg_map = {r"[1-2][1-2]" : "hipernating",
           r"[1-2][3-4]" : "at_Risk",
           r"[1-2]5" : "cant_loose",
           r"3[1-2]" : "about_to_sleep",
           r"33" : "need_attention",
           r"[3-4][4-5]" : "loyal_costumer",
           r"41" :  "promising",
           r"51" : "new_customers",
           r"[4-5][2-3]": "potential_loyalists",
           r"5[4-5]" : "champions"
           }

rfm["segment"] = rfm["RFM_SCORE"].replace(seg_map,regex = True) #birleştirilen skorları segment et

rfm[["segment","recency","frequency","monetary"]].groupby("segment").agg(["mean","count"])

print(rfm[rfm["segment"] == "need_attention"].index) #dikkate alınmazsa churne gidecek müşteriler

new_df = pd.DataFrame()
new_df["new_customer_id"] = rfm[rfm["segment"] == "new_customers"].index
new_df["new_customer_id"].astype(int)
print(new_df)

new_df.to_csv("new_customers.csv")