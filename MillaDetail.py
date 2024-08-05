import traceback

import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import time
import requests
import re


pd.set_option("display.max_columns", None)
pd.set_option("display.max_row",None)
pd.set_option("display.float_format", lambda x: "%.3f" % x)
pd.set_option("display.width", 500)
pd.set_option('display.max_colwidth', 500)

urun_detaylar = []
df = pd.read_csv("Dataset2/general_df_woman_milla_sweatshirt.csv")
df.shape


# Chrome options
chrome_options = Options()
chrome_options.add_argument("--headless")  # Operate in headless mode
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")

# Initialize the webdriver
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=chrome_options)

a = 0

for link in df["URUN_LINK"]:
    urun_detail_dict = {}
    a = a + 1
    if link:
        url_base2 = f"https://www.trendyol-milla.com{link}"
        try:
            driver.get(url_base2)
            time.sleep(1)  # Wait for the page to load

            # Parse the page source with BeautifulSoup
            soup = BeautifulSoup(driver.page_source, 'html.parser')

            pattern = r"-p-(\d+)"
            match = re.search(pattern, link)
            if match:
                urun_detail_dict["ID"] = match.group(1)
            else:
                urun_detail_dict["ID"] = None

            print(f"{a} -- {urun_detail_dict}")

            # Get favorite-count value
            favorite_count = soup.find("span", class_="favorite-count")
            if favorite_count:
                urun_detail_dict["FAVORITE_COUNT"] = favorite_count.get_text(strip=True)
            else:
                urun_detail_dict["FAVORITE_COUNT"] = None

            rating_detail = soup.find("div", class_="rating-line-count")
            if rating_detail:
                urun_detail_dict["RATING_DETAIL"] = rating_detail.get_text(strip=True)
            else:
                urun_detail_dict["RATING_DETAIL"] = None

            cargo_delivery_title = soup.find("span", class_="pr-dd-nr-text")
            cargo_delivery_value = soup.find("span", class_="dd-txt-vl")
            if cargo_delivery_title and cargo_delivery_value:
                urun_detail_dict["DELIVERY_DATE"] = cargo_delivery_value.get_text(strip=True)
            else:
                urun_detail_dict["DELIVERY_DATE"] = None

            proff_link_tag = soup.find('a', class_='rvw-cnt-tx')
            if proff_link_tag and 'href' in proff_link_tag.attrs:
                proff_link = proff_link_tag['href']
                urun_detail_dict["PROFF_LINK"] = proff_link
            else:
                urun_detail_dict["PROFF_LINK"] = None


            urun_detail = soup.find_all("div", class_="product-detail-container")
            for ul in urun_detail:
                for li in ul.find_all("li"):
                    attr_name = li.find("span", class_="attr-name attr-key-name-w")
                    attr_value = li.find("div", class_="attr-name attr-value-name-w")
                    if attr_name and attr_value:
                        urun_detail_dict[attr_name.get_text(strip=True)] = attr_value.get_text(strip=True)
                    else:
                        material_info = li.get("class") and "material-info" in li.get("class")
                        if material_info:
                            material_title = li.find("span").get_text(strip=True)
                            material_value = li.find("b").get_text(strip=True)
                            urun_detail_dict[material_title] = material_value

            urun_detaylar.append(urun_detail_dict)
        except Exception as e:
            tb = traceback.format_exc()
            # Hatanın satır numarasını al
            tb_lines = tb.splitlines()
            line_info = tb_lines[-3]  # Hata mesajındaki satır bilgisi genellikle sondan üçüncü satırdadır.

            print(f"Hata: {e}")
            print(f"Satır Bilgisi: {line_info}")
            print(f"Ayrıntılı Hata Bilgisi:\n{tb}")
            urun_detaylar.append({})
    else:
        urun_detaylar.append({})

# Close the driver
driver.quit()

urun_detaylar_df = pd.DataFrame(urun_detaylar)



urun_detaylar_df.shape

urun_detaylar_df[250:260]
urun_detaylar_df.head()

file_path = 'C:\\Users\\fyilanci\\Desktop\\data_bootcamp\\\TrendyolData\\\DetailData\\urun_detaylar_sweatshirt.csv'
urun_detaylar_df.to_csv(file_path)
