import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

pd.set_option("display.max_columns",20)
pd.set_option("display.max_rows",20)
pd.set_option("display.float_format",lambda x : "%.5f" %x)

df_ = pd.read_excel(r"C:\Users\bett0\Desktop\online_retail_II.xlsx",sheet_name="Year 2009-2010")
df = df_.copy()
#print(df.loc[df["Customer ID"] == 12346.00000,:])

#veri temizleme hazırlama
df.dropna(inplace=True)
df = df[~df["Invoice"].str.contains("C",na = False)]
df = df[df["Quantity"] > 0]

df["Total_Price"] = df["Price"] * df["Quantity"]
cltv_c = df.groupby("Customer ID").agg({"Invoice":lambda x:x.nunique(),
                                        "Total_Price":lambda x:x.sum()})
#average_order_value hesabı
cltv_c.columns = ["total_transaction","total_price"]
print(cltv_c.head())
cltv_c["average_order_value"] = cltv_c["total_price"]/cltv_c["total_transaction"]

#purchase frequency hesabı (müşterinin yaptığı toplam işlem sayısı / toplam müşteri sayısı)
total_customer = cltv_c.shape[0]
cltv_c["purchase_frequency"] = cltv_c["total_transaction"]/total_customer

#repeat rate- churn rate (birden fazla alışveriş yapan müşteri sayısı / tüm müşteriler)
repeat_rate = cltv_c[cltv_c["total_transaction"] > 1].shape[0] / total_customer
churn_rate = 1 - repeat_rate

#profit margin (profit_margin = total_price * 0.10)
cltv_c["profit_margin"] = cltv_c["total_price"] * 0.10

#customer value (costumer_value = average_order_value * purchase frequency)
cltv_c["customer_value"] = cltv_c["average_order_value"] * cltv_c["purchase_frequency"]

#customer lifetime value (cltv = (customer_value / churn_rate) * profit_margin)

cltv_c["cltv"] = (cltv_c["customer_value"] / churn_rate) * cltv_c["profit_margin"]

cltv_c.sort_values(by = "cltv",ascending = False).head()

cltv_c["segment"] = pd.qcut(cltv_c["cltv"], 4, labels=["D", "C","B","A"])

cltv_c.groupby("segment").agg({"count","mean","sum"})

cltv_c.to_csv("cltv_c.csv")