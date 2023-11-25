import pandas as pd
import numpy as np
import seaborn as sns
import datetime as dt
import matplotlib.pyplot as plt

pd.set_option("display.max_columns",None)

df_ = pd.read_csv(r"C:\Users\bett0\Desktop\flo_data_20k.csv")

df = df_.copy()

print(df.head(10)) #ilk on gözlem

print(df.columns) #değişken isimleri

print(df.isnull().sum()) #boş değer toplamları

print(df.info()) #değişken tipleri

df["Omnichannel"] = df["order_num_total_ever_offline"] + df["order_num_total_ever_online"] # yeni değişkene müşterinin online + offline toplam alışverişini atadık

df["TotalPrice"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]

date_columns = df.columns[df.columns.str.contains("date")]

df[date_columns] = df[date_columns].apply(pd.to_datetime)

print((df.info()))

#Alışveriş kanallarındaki müşteri sayısının, toplam alınan ürün sayısının ve toplam harcamaların dağılımına bakınız

print(df.groupby("order_channel").agg({"master_id" : "count",
                    "Omnichannel" : "sum",
                    "TotalPrice" : "sum",
                     }))

#En fazla kazancı getiren ilk 10 müşteriyi sıralayınız

print(df.sort_values(by = "TotalPrice",ascending = False)[:10])

#En fazla siparişi veren ilk 10 müşteriyi sıralayınız

print(df.sort_values(by = "Omnichannel",ascending = False).head(10))
df.reset_index(inplace = True)
#Veri ön hazırlık sürecini fonksiyonlaştırınız.

def data_prep(dataframe):

    dataframe["Omnichannel"] = dataframe["order_num_total_ever_offline"] + dataframe["order_num_total_ever_online"] # yeni değişkene müşterinin online + offline toplam alışverişini atadık

    dataframe["TotalPrice"] = dataframe["customer_value_total_ever_offline"] + dataframe["customer_value_total_ever_online"]

    date_columns = dataframe.columns[dataframe.columns.str.contains("date")]

    dataframe[date_columns] = dataframe[date_columns].apply(pd.to_datetime)





    #Müşteri özelinde Recency, Frequency ve Monetary metriklerini hesaplayınız.


    today_date = dt.datetime(2021,6,1)

    rfm = pd.DataFrame()

    rfm["master_id"] = dataframe["master_id"]

    rfm["recency"] = (today_date - dataframe["last_order_date"]).dt.days

    rfm["frequency"] = dataframe["Omnichannel"]

    rfm["monetary"] = dataframe["TotalPrice"]

    rfm = rfm[rfm["monetary"] > 0 ]

    #rfm skorları


    rfm["recency_score"] = pd.qcut(rfm["recency"],5,[5, 4, 3, 2, 1])

    rfm["monetary_score"] = pd.qcut(rfm["monetary"],5, [1,2,3,4,5],)

    rfm["frequency_score"] = pd.qcut(rfm["frequency"].rank(method="first"),5,[1,2,3,4,5])


    rfm["RF_SCORE"] =  rfm["recency_score"].astype(str) + rfm["frequency_score"].astype(str)

    #segment tanımlama

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

    rfm["segment"] = rfm["RF_SCORE"].replace(seg_map,regex = True) #birleştirilen skorları segment et

    return rfm

rfm = data_prep(df)
segmented_mean = rfm[["segment", "recency", "frequency", "monetary"]].groupby("segment").agg(["mean", "count"])

print(segmented_mean)

rfm.to_csv("rfm_flo_müşteri.csv",index = False)

new_df = rfm.merge(df,on = "master_id")

print(new_df)

#Sadıkmüşterilerinden(champions,loyal_customers)ve
#kadınkategorisindenalışverişyapankişilerözelolarakiletişimkurulacakmüşteriler.
#Bumüşterilerinidnumaralarınıcsvdosyasınakaydediniz

filtered_df = new_df[(new_df["segment"].isin(["champions", "loyal_customers"])) & (new_df["interested_in_categories_12"].str.contains("KADIN"))]
filtered_df["master_id"].to_csv("customer_id")

print(filtered_df)

# b. Erkek ve Çoçuk ürünlerinde %40'a yakın indirim planlanmaktadır. Bu indirimle ilgili kategorilerle ilgilenen geçmişte iyi müşterilerden olan ama uzun süredir
# alışveriş yapmayan ve yeni gelen müşteriler özel olarak hedef alınmak isteniliyor. Uygun profildeki müşterilerin id'lerini csv dosyasına indirim_hedef_müşteri_ids.csv
# olarak kaydediniz.

target_segments_customer_ids = rfm[rfm["segment"].isin(["cant_loose","hibernating","new_customers"])]["master_id"]
cust_ids = df[(df["master_id"].isin(target_segments_customer_ids)) & ((df["interested_in_categories_12"].str.contains("ERKEK"))|(df["interested_in_categories_12"].str.contains("COCUK")))]["master_id"]
cust_ids.to_csv("indirim_hedef_müşteri_ids.csv", index=False)

print("cust_ids")