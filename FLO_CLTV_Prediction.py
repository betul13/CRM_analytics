import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from lifetimes.plotting import plot_period_transactions
from lifetimes import GammaGammaFitter
from lifetimes import BetaGeoFitter

pd.set_option("display.max_columns",None)

df_ = pd.read_csv(r"C:\Users\bett0\Desktop\flo_data_20k.csv")

df = df_.copy()

print(df.head(10))

def outlier_thresholds(dataframe,variable):

    quantile1 = dataframe[variable].quantile(0.01)

    quantile3= dataframe[variable].quantile(0.99)

    interquantile_range = quantile3 - quantile1

    up_limit = (quantile3 + 1.5 * interquantile_range)

    low_limit = (quantile1 - 1.5 * interquantile_range)

    return up_limit,low_limit

def replace_with_thresholds(dataframe,variable):

    up_limit,low_limit = outlier_thresholds(dataframe,variable)

    dataframe.loc[dataframe[variable] > up_limit, variable] = up_limit

    dataframe.loc[dataframe[variable] < low_limit, variable] = low_limit

    up_limit = up_limit.round()

    low_limit = low_limit.round()

#"order_num_total_ever_online", "order_num_total_ever_offline", "customer_value_total_ever_offline",
#"customer_value_total_ever_online" değişkenlerinin aykırı değerleri varsa baskılayanız.

for col in  ["order_num_total_ever_online", "order_num_total_ever_offline", "customer_value_total_ever_offline",
"customer_value_total_ever_online"] :
    replace_with_thresholds(df,col)

df["Omnichannel"] = df["order_num_total_ever_offline"] + df["order_num_total_ever_online"]

df["TotalPrice"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]

#Değişken tiplerini inceleyiniz. Tarih ifade eden değişkenlerin tipini date'e çeviriniz.

date_columns = df.columns[df.columns.str.contains("date")]
df[date_columns] = df[date_columns].apply(pd.to_datetime)

#Veri setindeki en son alışverişin yapıldığı tarihten 2 gün sonrasını analiz tarihi olarak alınız

analysis_date = dt.datetime(2021,6,1)

# customer_id, recency_cltv_weekly, T_weekly, frequency ve monetary_cltv_avg değerlerinin yer aldığı yeni bir cltv dataframe'i
#oluşturunuz. Monetary değeri satın alma başına ortalama değer olarak, recency ve tenure değerleri ise haftalık cinsten ifade edilecek.

cltv_df = pd.DataFrame({"customer_id" : df["master_id"],
                        "recency_cltv_weekly" : (df["last_order_date"] - df["first_order_date"]).dt.days // 7,
                        "T_weekly" : (analysis_date - df["first_order_date"]).dt.days // 7,
                        "frequency" : df["Omnichannel"] ,
                        "monetary_cltv_avg" : (df["TotalPrice"] / df["Omnichannel"])})

cltv_df.columns = ["customer_id","recency","T","frequency","monetary"]

cltv_df.reset_index(drop = True,inplace=True)
cltv_df = cltv_df[cltv_df["frequency"] > 1]
cltv_df["frequency"] = cltv_df["frequency"].astype("int64")
#BG/NBD modelini fit ediniz

bgf = BetaGeoFitter(penalizer_coef = 0.01)

bgf.fit(cltv_df["frequency"],
        cltv_df["recency"],
        cltv_df["T"])

#3 ay içerisinde müşterilerden beklenen satın almaları tahmin ediniz ve exp_sales_3_month olarak cltv
#dataframe'ine ekleyiniz.

cltv_df["exp_sales_3_month"] = bgf.predict(12,
                                           cltv_df["frequency"],
                                           cltv_df["recency"],
                                           cltv_df["T"])

cltv_df["exp_sales_6_month"] = bgf.predict(24,
                                           cltv_df["frequency"],
                                           cltv_df["recency"],
                                           cltv_df["T"])
cltv_df.reset_index(inplace=True)

#Gamma-Gamma modelini fit ediniz. Müşterilerin ortalama bırakacakları değeri tahminleyip exp_average_value olarak cltv
#dataframe'ine ekleyiniz

ggf = GammaGammaFitter(penalizer_coef = 0.01)

ggf.fit(cltv_df["frequency"],
          cltv_df["monetary"])


# BG-NBD ve GG modeli  ile CLTV'nin hesaplanması

cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df["frequency"],
                                   cltv_df["recency"],
                                   cltv_df["T"],
                                   cltv_df["monetary"],
                                   time = 6, #6 aylık
                                   freq = "W", # Tnin frekans bilgisi
                                   discount_rate = 0.01)

cltv_df["cltv"] = cltv

cltv_df.reset_index(inplace=True)


# CLTV değeri hesaplanmış veri çerçevesini sıralayarak en yüksek 20 kişiyi gözlemleyin

top_20 = cltv_df.sort_values(by = "cltv",ascending=False).head(20)
print(top_20)

#6 aylık CLTV'ye göre tüm müşterilerinizi 4 gruba (segmente) ayırınız ve grup isimlerini veri setine ekleyiniz.

cltv_df["segment_6_month"] = pd.qcut(cltv_df["exp_sales_6_month"], 4, ["D","C","B","A"])
print(cltv_df.head(3))