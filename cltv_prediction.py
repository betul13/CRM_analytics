# Data prepreration
#Bir e-ticaret şirketi müşterilerini segmente ayırıp bu segmentlere göre pazarlama stratejisi geliştirmek istiyor.

#pip install lifetime

import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions
from sklearn.preprocessing import MinMaxScaler

pd.set_option("display.max_columns",None)


def outlier_thresholds(dataframe,variable):
    quartile1 = dataframe[variable].quantile(0.01) #normalde %25 lik çeyrek değer alınır. Ancak veriye bakıldığında çok fazla aykırı değer yoktur.İşlem yapılırsa değer kaybedilir.
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3-quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit,up_limit

def replace_with_thresholds(dataframe,variable):
    low_limit,up_limit = outlier_thresholds(dataframe,variable)
    dataframe.loc[(dataframe[variable] < low_limit),variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit),variable] = up_limit

#verinin okunması

df_ = pd.read_excel(r"C:\Users\bett0\Desktop\online_retail_II.xlsx",sheet_name = "Year 2010-2011")
df = df_.copy()
df.describe().T
df.head()
print(df.isnull().sum())

#veri ön işleme

df.dropna(inplace=True)
df = df[~df["Invoice"].str.contains("C",na = False)]
df = df[df["Quantity"] > 0]
df = df[df["Price"] > 0]

replace_with_thresholds(df,"Quantity") #içinde outlier_thresholds ile eşik değeri hesaplandı ve aykırı değerler eşik değeri ile değişti
replace_with_thresholds(df,"Price") #aykırı değerlerden arındık

df["TotalPrice"] = df["Quantity"] * df["Price"]

today_date = dt.datetime(2011,12,11)

#Lifetime veri yapısının hazırlanması

#recency = son satın alma üzerinden geçen zaman haftalık (kullanıcı özelinde)(son satın alma - ilk satın alma)
#T = Müşterinin yaşı (Haftalık) (Analiz tarihindem ne kadar önce ilk satın alma yapılmış)
#frequency = Tekrar eden toplam satın alma sayısı(frequency > 1)
#monetary_value : satın alma başına ortalama kazanç

cltv_df = df.groupby("Customer ID").agg({"InvoiceDate":[lambda date : (date.max()-date.min()).days,
                                                        lambda date : (today_date - date.min()).days],
                                         "Invoice" : lambda invoice : invoice.nunique(),
                                         "TotalPrice" : lambda totalprice : totalprice.sum()})

cltv_df.columns = ["recency","T","frequency","monetary"]

cltv_df.reset_index(drop = True)

cltv_df["monetary"] = cltv_df["monetary"] / cltv_df["frequency"]

cltv_df = cltv_df[cltv_df["frequency"] > 1]

cltv_df["recency"] = cltv_df["recency"] / 7

cltv_df["T"] = cltv_df["T"] / 7

# BG-NBD Modelinin Kurulması

bgf = BetaGeoFitter(penalizer_coef = 0.01)

bgf.fit(cltv_df["frequency"],
        cltv_df["recency"],
        cltv_df["T"])

#1 Hafta içerisinde en çok satın alma beklediğimiz 10 müşteri kimdir?

cltv_df["expected_purc_1_week"] = bgf.conditional_expected_number_of_purchases_up_to_time(1,
                                                        cltv_df["frequency"],
                                                        cltv_df["recency"],
                                                        cltv_df["T"])

cltv_df["expected_purc_1_month"] = bgf.predict(4,
                                                cltv_df["frequency"],
                                                cltv_df["recency"],
                                                cltv_df["T"])

cltv_df["expected_purc_3_month"] = bgf.predict(12 ,
                                                cltv_df["frequency"],
                                                cltv_df["recency"],
                                                cltv_df["T"])

# Görselleştirilmesi actual-prediction

plot_period_transactions(bgf)
plt.show()

# GAMMA-GAMMA modelinin kurulması

ggf = GammaGammaFitter(penalizer_coef=0.01)

ggf.fit(cltv_df["frequency"], cltv_df["monetary"])

cltv_df["expected_average_profit"] = ggf.conditional_expected_average_profit(cltv_df["frequency"], cltv_df["monetary"])

top_customers = cltv_df.sort_values(by="expected_average_profit", ascending=False).head(10)

print(top_customers)

# BG-NBD ve GG modeli  ile CLTV'nin hesaplanması

cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df["frequency"],
                                   cltv_df["recency"],
                                   cltv_df["T"],
                                   cltv_df["monetary"],
                                   time = 3, #3 aylık
                                   freq = "W", # Tnin frekans bilgisi
                                   discount_rate = 0.01)
cltv = cltv.reset_index()

cltv_final = cltv_df.merge(cltv,on = "Customer ID",how = "left")
print(cltv_final.sort_values(by = "clv",ascending = False).head(10))

#müşteri segmentinin oluşturulması

cltv_final["segment"] = pd.qcut(cltv_final["clv"],4,labels = ["D","C","B","A"])
print(cltv_final.sort_values(by="clv",ascending = False).head(50))

print(cltv_final.groupby("segment").agg({"count","mean","sum"}))