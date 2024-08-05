#!pip install pandas numpy beautifulsoup4 selenium webdriver-manager requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import time
import requests

pd.set_option("display.width", 1000)
pd.set_option("display.max_columns", 500)
pd.set_option('display.max_colwidth', 500)

# Chrome options
chrome_options = Options()
chrome_options.add_argument("--headless")  # Operate in headless mode
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")

# Initialize the webdriver
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=chrome_options)

columns = ["ID", "MARKA", "URUN_NAME", "FIYAT", "RATE_COUNT", "URUN_LINK", "KATEGORI"]

all_rows = []

for page in range(0, 50):
    url = f"https://www.trendyol-milla.com/kadin-kaban-x-g1-c1075?pi={page}"
    print(f"Fetching , page {page}")
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        urunler = soup.find_all("div", class_="p-card-wrppr with-campaign-view")
        for urun in urunler:
            row = {}

            row["KATEGORI"] = "Kaban"

            ıd = urun.get("data-id")
            if ıd:
                row["ID"] = ıd
            else:
                row["ID"] = None

            link = urun.find("a", href=True)
            if link:
                row["URUN_LINK"] = link['href']
            else:
                row["URUN_LINK"] = None

            marka = urun.find("span", class_="prdct-desc-cntnr-ttl")
            if marka:
                row["MARKA"] = marka.get("title", "")
            else:
                row["MARKA"] = ""
            fiyat = urun.find("div", class_="prc-box-dscntd")
            if fiyat:
                row["FIYAT"] = float(fiyat.text.strip().replace("TL", "").replace(".", "").replace(",", "."))
            else:
                row["FIYAT"] = 0
            urundsc = urun.find("span", class_="prdct-desc-cntnr-name")
            urundsc2 = urun.find("div", class_="product-desc-sub-container")
            if urundsc or urundsc2:
                row["URUN_NAME"] = urundsc.get_text(strip=True) + " " + urundsc2.get_text(strip=True)
            else:
                row["URUN_NAME"] = None

            uratıng = urun.find("span", class_="ratingCount")
            if uratıng:
                uratıng = uratıng.get_text(strip=True).strip("()")
                try:
                    uratıng = int(uratıng)
                except ValueError:
                    uratıng = 0
                row["RATE_COUNT"] = uratıng
            else:
                row["RATE_COUNT"] = 0
            all_rows.append(row)

    except Exception as e:
        print(f"Hata oluştu: {e}")
        break
    page = page + 1

df = pd.DataFrame(all_rows, columns=columns)

df.head()
df[250:260]
df.shape


df.duplicated().any()
df = df.drop_duplicates()

file_path = 'C:\\Users\\fyilanci\\Desktop\\data_bootcamp\\TrendyolData\\general_df_woman_milla_kaban.csv'
df.to_csv(file_path)





