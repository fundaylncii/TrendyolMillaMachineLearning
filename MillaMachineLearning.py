##!pip install scikit-optimize
##!pip install missingno
##!pip install catboost
##!pip install lightgbm
##!pip install xgboost
from random import uniform, randint

import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from datetime import date
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.neighbors import LocalOutlierFactor, KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
import os
from skopt import gp_minimize
from skopt.space import Real
import math
import missingno as msno
import warnings
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score,GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import cross_validate

pd.set_option("display.width", 1000)
pd.set_option("display.max_columns", 500)
pd.set_option('display.max_colwidth', 500)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

########################################################################################################################
## VERİ SETİNİN OLUŞTURULMASI
########################################################################################################################

## Detail dosyarının tek bir df haline getirilmesi:
directory_path = "DetailData"
dfs = []

for filename in os.listdir(directory_path):
    if filename.endswith(".csv"):
        file_path = os.path.join(directory_path, filename)
        df = pd.read_csv(file_path)
        dfs.append(df)

detail = pd.concat(dfs, ignore_index=True)
detail.head()
detail.shape

## General dosyasının tek bir df haline getirilmesi:

directory_path = "GeneralData"
dfs = []

for filename in os.listdir(directory_path):
    if filename.endswith(".csv"):
        file_path = os.path.join(directory_path, filename)
        df = pd.read_csv(file_path)
        dfs.append(df)

general = pd.concat(dfs, ignore_index=True)
general.head()
general.shape

## detail dfinde 500 dan az sayıda veri içeren kolonların silinmesi:

detail.count()

def remove_columns_with_few_data(df, threshold=500):
    column_counts = df.count()
    columns_to_keep = column_counts[column_counts >= threshold].index
    return df[columns_to_keep]


new_detail = remove_columns_with_few_data(detail)
new_detail.count()
new_detail.shape

## merge dfin oluşturulması:

milla_df = pd.merge(general, new_detail, on = "ID", how="left")

milla_df[milla_df["ID"].duplicated() == True]

milla_df.loc[milla_df["ID"] == "660274893"]

milla_df = milla_df.drop_duplicates(subset="ID")
milla_df.shape

## Değişken isimlerinin düzenlenmesi

milla_df.columns = milla_df.columns.str.upper().str.replace(" ", "_")
milla_df.columns = milla_df.columns.str.replace("/","_")
milla_df.columns = milla_df.columns.str.replace("___","_")
milla_df.columns = milla_df.columns.str.replace("-","_")

milla_df.info()

milla_df.loc[(milla_df["KATEGORI"] == "Elbise") & (milla_df["RENK"] == "Kırmızı")]


## index değişkenlerinin drop edilmesi

drop_columns = ["UNNAMED:_0_X","UNNAMED:_0_Y"]

milla_df = milla_df.drop(drop_columns, axis=1)

milla_df.info()
milla_df.shape
milla_df.head()

## YANLIŞ DEĞİŞKEN TİPLERİNİN DÜZENLENMESİ:

milla_df["ID"] = milla_df["ID"].astype(str)
milla_df["FAVORITE_COUNT"] = milla_df["FAVORITE_COUNT"].fillna(0).astype("int64")


########################################################################################################################
## KEŞİFÇİ VERİ ANALİZİ
########################################################################################################################
def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### COUNT #####################")
    print(dataframe.count())

check_df(milla_df)

# NUMERİK VE KATEGORİK DEĞİŞKENLERİN ANALİZİ

def grap_col_names(dataframe, cat_th= 10, car_th=20):
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

cat_cols , num_cols, cat_but_car = grap_col_names(milla_df)

milla_df.head()

not_model_cols = ["ID","URUN_LINK","PROFF_LINK","URUN_NAME"]

cat_but_car = [col for col in cat_but_car if col not in not_model_cols]

cat_cols = cat_cols + cat_but_car
len(cat_cols)

## KATEGORİK DEĞİŞKENLERİN ANALİZİ:

def cat_summary_l(dataframe, cat_cols, plot=False):
    for col_name in cat_cols:
        print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                            "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
        print("##########################################")
        if plot:
            sns.countplot(x=dataframe[col_name], data=dataframe)
            plt.show()

cat_summary_l(milla_df, cat_cols)

## KATEGORİ SAYISI FAZLA OLAN DEĞİŞKENLER İÇİN GRAFİKLENDİRME

material_counts = milla_df['ORTAM'].value_counts()
top_materials = material_counts.nlargest(5)
other_materials_count = material_counts.iloc[5:].sum()
top_materials['Diğer'] = other_materials_count
plt.figure(figsize=(8, 5))
top_materials.plot(kind='bar')
plt.title('En Çok Kullanılan Materyaller')
plt.ylabel('Sayım')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

## NUMERİK DEĞİŞKENLERİN ANALİZİ
def num_summary(dataframe, numerical_col):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

for col in num_cols:
    num_summary(milla_df, col)

milla_df.info()

## numerik değerlerin grafiklendirilmesi:

plt.figure(figsize=(8, 5))
plt.hist(milla_df["FIYAT"], bins=50, edgecolor='black')
plt.title('Histogram Example')
plt.xlabel('FIYAT')
plt.ylabel('Count')
plt.show()

## HEDEF DEĞİŞKENİN OLUŞTURULMASI:

## Ürün Skoru = (Ortalama Puan / 5)^w1 * (1 + log10(Puan Sayısı + 1))^w2 * log10(Yorum Sayısı + 1)^w3
def validation_metric(weights):
    w1, w2, w3 = weights
    scores = (milla_df["RATING_DETAIL"] / 5)**w1 * (1 + np.log10(milla_df["RATE_COUNT"] + 1))**w2 * np.log10(milla_df["FAVORITE_COUNT"] + 1)**w3
    return -np.mean(scores)

# Parametre aralıkları
space = [
    Real(0.1, 2.0, name='w1'),
    Real(0.1, 2.0, name='w2'),
    Real(0.1, 2.0, name='w3')]

# Bayes Optimizasyonu
res = gp_minimize(validation_metric, space, n_calls=50, random_state=0)

# En iyi parametreler ve doğrulama metrik skoru
best_weights = res.x
best_score = -res.fun  # Negatif aldığımız için tekrar pozitife çeviriyoruz

print("En iyi ağırlıklar:", best_weights)
print("En iyi doğrulama skoru:", best_score)

## En iyi ağırlıklar: [0.1, 2.0, 2.0]
## En iyi doğrulama skoru: 166.31400040350468

milla_df["TOP_SCORE"] = (milla_df["RATING_DETAIL"] / 5)**0.01 * (1 + np.log10(milla_df["RATE_COUNT"] + 1))**2 * np.log10(milla_df["FAVORITE_COUNT"] + 1)**2

milla_df.shape
milla_df.head()
milla_df["TOP_SCORE"].describe().T
milla_df[milla_df["TOP_SCORE"]> 1000]


## NUMERİK DEĞİŞKENLERİN TARGETE GÖRE ANALİZİ:

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

for col in num_cols:
    target_summary_with_num(milla_df, "TOP_SCORE", col)

## KATEGORİK DEĞİŞKENLERİN TARGETE GÖRE ANALİZİ:
def target_summary_with_cat(dataframe, target, categorical_col):
    print(categorical_col)
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean(),
                        "Count": dataframe[categorical_col].value_counts(),
                        "Ratio": 100 * dataframe[categorical_col].value_counts() / len(dataframe)}), end="\n\n\n")

for col in cat_cols:
    target_summary_with_cat(milla_df, "TOP_SCORE", col)


## KORELASYON ANALİZİ:

milla_df[num_cols].corr()

# Korelasyon Matrisi
f, ax = plt.subplots(figsize=[18, 13])
sns.heatmap(milla_df[num_cols].corr(), annot=True, fmt=".2f", ax=ax, cmap="magma")
ax.set_title("Correlation Matrix", fontsize=20)
plt.show()

milla_df.corrwith(milla_df["TOP_SCORE"]).sort_values(ascending=False)


########################################################################################################################
## FEATURE ENGINEERING
########################################################################################################################

## EKSİK DEĞER ANALİZİ:

milla_df.isnull().sum()

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    if na_name:
        return na_columns,missing_df

na_columns, missing_df = missing_values_table(milla_df, na_name=True)

## %80 ni null olan değişkenlerin df den kaldırılması
drop_na_columns = missing_df[missing_df["ratio"] > 80].index
milla_df.drop(columns=drop_na_columns, axis=1, inplace=True)

## tekrar null değerlerin check edilmesi
na_columns, missing_df = missing_values_table(milla_df, na_name=True)

## EKSİK DEĞERLERİN GRAFİKLENDİRİLMESİ
msno.bar(milla_df)
plt.show()

msno.matrix(milla_df)
plt.show()

milla_df.info()

## na_colums her değişkeni kendi kategorisindeki mean ve mode ile doldurma.
na_columns_cat = [col for col in na_columns if col not in ["PROFF_LINK","ÜRÜN_DETAYI","TOP_SCORE","RATING_DETAIL"]]

## KATEGORİK DEĞİŞKENLER İÇİN:

for col in na_columns_cat:
    milla_df[col] = milla_df.groupby('KATEGORI')[col].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else np.nan))


milla_df.info()

drop_columns = ["KALINLIK","ASTAR_DURUMU","CEP","KOL_TIPI","YAKA_TIPI","KOL_BOYU"]

milla_df = milla_df.drop(drop_columns, axis=1)
milla_df = milla_df.drop("ÜRÜN_DETAYI", axis=1)

## TEKRAR EKSİK DEĞERLERİN CHECK EDİLMESİ:
na_columns, missing_df = missing_values_table(milla_df, na_name=True)

na_columns_cat = [col for col in na_columns if col not in ["RATING_DETAIL","TOP_SCORE"]]

for col in na_columns_cat:
    milla_df[col] = milla_df.groupby('KATEGORI')[col].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else np.nan))

## TEKRAR EKSİK DEĞERLERİN CHECK EDİLMESİ:
## bazı değişkenlerde kategorik bazda da mode değeri nan olanlar vardı bunlar tek tek incelendi.
## bu değişkenlerde olmayan özellikler olduğu için bunlara "yok" değeri atıldı.

mode_values = milla_df.groupby('KATEGORI')['KOLEKSIYON'].apply(lambda x: x.mode().iloc[0] if not x.mode().empty else None)
nan_modes = mode_values[mode_values.isna()]

milla_df[milla_df['KATEGORI'] == 'Elbise']['KOLEKSIYON'].mode()[0]

milla_df.loc[(milla_df['KATEGORI'] == 'Mezuniyet Elbisesi') & (milla_df['KOLEKSIYON'].isna()), 'KOLEKSIYON'] = 'Tesettür Giyim'

## NUMERİK DEĞİŞKENLERDE MEAN ATAMASI:
na_columns, missing_df = missing_values_table(milla_df, na_name=True)
milla_df["RATING_DETAIL"] = milla_df.groupby('KATEGORI')["RATING_DETAIL"].transform(lambda x: x.fillna(x.mean()))

## ESKİK DEĞERLERDEN SONRA TOP_SCORE DEĞERİNİN YENİDEN HESAPLANMASI:

milla_df["TOP_SCORE"] = (milla_df["RATING_DETAIL"] / 5)**0.01 * (1 + np.log10(milla_df["RATE_COUNT"] + 1))**2 * np.log10(milla_df["FAVORITE_COUNT"] + 1)**2

file_path = 'C:\\Users\\fyilanci\\Desktop\\data_bootcamp\\\TrendyolData\\MILLA_NOT_MISSING.csv'
milla_df.to_csv(file_path)

msno.matrix(milla_df)
plt.show()

## AYKIRI DEĞER ANALİZİ:

milla_df = pd.read_csv("MILLA_NOT_MISSING.csv")
milla_df.isnull().sum()
milla_df["ID"] = milla_df["ID"].astype(str)
milla_df["RATE_COUNT"] = milla_df["RATE_COUNT"].astype("int64")
milla_df["FAVORITE_COUNT"] = milla_df["FAVORITE_COUNT"].astype("int64")
cat_cols, num_cols, cat_but_car = grap_col_names(milla_df)


sns.boxplot(x=milla_df["RATING_DETAIL"])
plt.show()

def outlier_threshold (dataframe, col_name, q1=0.01, q3=0.99):
  quartile1 = dataframe[col_name].quantile(q1)
  quartile3 = dataframe[col_name].quantile(q3)
  interquantile_range = quartile3 - quartile1
  up_limit = quartile3 + 1.5 * interquantile_range
  low_limit = quartile1 - 1.5 * interquantile_range
  return low_limit, up_limit


def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_threshold(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

num_cols = [col for col in num_cols if col not in ["TOP_SCORE","Unnamed: 0"]]
for col in num_cols:
    print(check_outlier(milla_df, col))

milla_df["FIYAT"].describe().T
milla_df["RATE_COUNT"].describe().T
milla_df["FAVORITE_COUNT"].describe().T
milla_df["RATING_DETAIL"].describe().T



## AYKIRI DEĞER BASKILAMA:
def replace_with_thresholds(dataframe, variable, q1=0.05, q3=0.95):
    low_limit, up_limit = outlier_threshold(dataframe, variable, q1=0.05, q3=0.95)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

for col in num_cols:
    print(col, check_outlier(milla_df, col))
    if check_outlier(milla_df, col):
        replace_with_thresholds(milla_df, col)

for col in num_cols:
    print(check_outlier(milla_df, col))

## AYKIRI DEĞER BASKILAMA İŞLEMİNDEN SONRA TOP_SCORE DEĞERİNİN YENİDEN OLUŞTURULMASI:
milla_df["TOP_SCORE"] = (milla_df["RATING_DETAIL"] / 5)**0.01 * (1 + np.log10(milla_df["RATE_COUNT"] + 1))**2 * np.log10(milla_df["FAVORITE_COUNT"] + 1)**2


## ÖZELLİK ÇIKARIMI:

milla_df.head()
milla_df.info()
milla_df["MATERYAL_BILEŞENI"].value_counts()

## MATERYAL BİLEŞENLERİNE GÖRE KUMAŞ ORANLARINI AYIRMA:

def extract_material_percentages(mix):
    material_dict = {}
    parts = mix.split(',')
    for part in parts:
        if '%' in part:
            percentage, material = part.split('%')
            percentage = percentage.strip()
            material = material.strip()
            material_dict[material] = int(percentage)
    return material_dict

material_percentages = milla_df['MATERYAL_BILEŞENI'].apply(extract_material_percentages)
materials_df = pd.DataFrame(material_percentages.tolist())
materials_df.fillna(0, inplace=True)

milla_df = pd.concat([milla_df, materials_df], axis=1)

milla_df[500:510]

## MATERYAL_BILEŞENI  gerek kalmadığı için df den drop edilir.

milla_df = milla_df.drop("MATERYAL_BILEŞENI", axis=1)

milla_df["TOP_SCORE"].describe().T
milla_df["TOP_LABEL"] = pd.qcut(milla_df["TOP_SCORE"],q=5,labels=[1,2,3,4,5])

milla_df["TOP_LABEL"].value_counts()
milla_df.info()

drop_labes = ["Unnamed: 0","MARKA","URUN_NAME"]
milla_df = milla_df.drop(drop_labes, axis=1)

## ENCODİNG İŞLEMİ:
cat_cols, num_cols, cat_but_car = grap_col_names(milla_df)

end_cols = ["KATEGORI",
            "KALIP",
            "DESEN",
            "RENK",
            "SILUET",
            "DOKUMA_TIPI",
            "MATERYAL",
            "EK_ÖZELLIK",
            "KUMAŞ_TIPI",
            "ÜRÜN_TIPI",
            "BOY",
            "ORTAM",
            "SÜRDÜRÜLEBILIRLIK_DETAYI",
            "PERSONA",
            "YAŞ",
            "PAKET_İÇERIĞI",
            "KOLEKSIYON"]

## One - Hot Encoding İşlemi:

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe


milla_df = one_hot_encoder(milla_df, end_cols, drop_first=True)
milla_df.head()
file_path = 'C:\\Users\\fyilanci\\Desktop\\data_bootcamp\\\TrendyolData\\MILLA_yes_feature_enginer.csv'
milla_df.to_csv(file_path)

########################################################################################################################
## MODELLEME:
########################################################################################################################

milla_df = pd.read_csv("MILLA_yes_feature_enginer.csv")
milla_df.isnull().sum()
milla_df["ID"] = milla_df["ID"].astype(str)
milla_df["RATE_COUNT"] = milla_df["RATE_COUNT"].astype("int64")
milla_df["FAVORITE_COUNT"] = milla_df["FAVORITE_COUNT"].astype("int64")
milla_df = milla_df.drop("Unnamed: 0", axis=1)
milla_df.head(2)

y = milla_df['TOP_SCORE']
X = milla_df.drop(["ID","TOP_SCORE"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=17)

models = [('KNN', KNeighborsRegressor()),
          ('CART', DecisionTreeRegressor()),
          ('RF', RandomForestRegressor()),
          ('GBM', GradientBoostingRegressor()),
           ("LightGBM", LGBMRegressor())]

for name, regressor in models:
    rmse = np.mean(np.sqrt(-cross_val_score(regressor, X_train, y_train, cv=5, scoring="neg_mean_squared_error")))
    print(f"RMSE: {round(rmse, 4)} ({name}) ")

##RMSE: 339234257.2085 (LR)
##RMSE: 18.3133 (KNN)
##RMSE: 3.0367 (CART)
##RMSE: 1.6459 (RF)
##RMSE: 3.8413 (GBM)
##RMSE: 2.2876 (LightGBM)

lgbm = LGBMRegressor().fit(X_train, y_train)
y_pred = lgbm.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


##  Log dönüşümü yaparak model kurunuz ve rmse sonuçların gözlemlenmesi.

## y = np.log1p(milla_df['TOP_SCORE'])
## X = milla_df.drop(["ID","TOP_SCORE"], axis=1)

## X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=17)
## lgbm = LGBMRegressor().fit(X_train, y_train)

## y_pred = lgbm.predict(X_test)
## new_y = np.expm1(y_pred)
## new_y_test = np.expm1(y_test)
## np.sqrt(mean_squared_error(new_y_test, new_y))
## 2.2928071997059827

# HİPER PARAMETRE OPTİMİZASYONU:

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", category=ConvergenceWarning)

lgbm_model = LGBMRegressor(random_state=46)

rmse = np.mean(np.sqrt(-cross_val_score(lgbm_model, X, y, cv=5, scoring="neg_mean_squared_error")))
## 2.2876014581095143

lgbm_params = {
    "learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3],  # Belirli öğrenme oranları
    "n_estimators": [100, 200, 500, 800, 1000],  # Belirli ağaç sayıları
    "num_leaves": [20, 30, 40, 50],  # Belirli yaprak sayıları
    "max_depth": [3, 4, 5, 6, 7],  # Belirli maksimum derinlik değerleri
    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0]  # Belirli örnekleme oranları
}


lgbm_gs_best = RandomizedSearchCV(
    lgbm_model,
    lgbm_params,
    n_iter=100,  # Denenecek kombinasyon sayısı
    cv=5,
    n_jobs=-1,
    random_state=42,
    verbose=True,
    scoring="neg_mean_squared_error"  # RMSE için negatif MSE kullanımı
).fit(X, y)

print(lgbm_gs_best.best_params_)
## {'subsample': 0.6, 'num_leaves': 40, 'n_estimators': 1000, 'max_depth': 3, 'learning_rate': 0.3}



final_model = lgbm_model.set_params(**lgbm_gs_best.best_params_).fit(X, y)


rmse = np.mean(np.sqrt(-cross_val_score(final_model, X, y, cv=5, scoring="neg_mean_squared_error")))

print(rmse)
## 0.015503390059646105


## feature_importance:

def plot_importance(model, feature, num=len(X), save=False):
  feature_imp = pd.DataFrame({"Value": model.feature_importances_, "Feature": feature.columns})
  plt.figure(figsize=(10, 10))
  sns.set(font_scale=1)
  sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
  plt.title("Features")
  plt.tight_layout()
  plt.show()
  if save:
    plt.savefig("importances.png")

plot_importance(final_model, X,10)



## TAHMİNLEME:

link_df = pd.read_csv("MILLA_GENEL.csv")

link_df["ID"] = link_df["ID"].astype(str)

link_df[link_df["ID"] == "742585517"]

all_ids = milla_df['ID'].values
random_id = np.random.choice(all_ids)

random_row = milla_df[milla_df['ID'] == random_id]

X_random = random_row.drop(columns=["ID", "TOP_SCORE"])

y_random = random_row['TOP_SCORE']

## TÜM VERİ SETİ İÇİN TAHMİNLEME VE EN YÜK TOP 10 ÜRÜN:

X_all = milla_df.drop(columns=["ID", "TOP_SCORE"])
y_preds = final_model.predict(X_all)


results_df = pd.DataFrame({
    "ID": milla_df["ID"],
    "Predicted_TOP_SCORE": y_preds
})


sorted_results_df = results_df.sort_values(by="Predicted_TOP_SCORE", ascending=False)
top_10_ids = sorted_results_df.head(10)
print(top_10_ids)

##             ID  Predicted_TOP_SCORE
##11191  356769149              651.450
##16520  123113990              651.392
##4626   136131774              651.313
##62       6257002              651.001
##3719   297465432              650.952
##9890   316161340              650.943
##3262   732699160              650.909
##687   744572707              650.738
##13202   31639223              650.727
##3858    77763676              650.660

link_df[link_df["ID"] == "77763676"]
