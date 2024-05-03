import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import PIL.Image
import urllib.request
import requests
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import fetch_california_housing
from sklearn.neighbors import KDTree
import warnings

warnings.filterwarnings("ignore")

california_housing = fetch_california_housing()
X = pd.DataFrame(california_housing.data, columns=california_housing.feature_names)
y = pd.DataFrame(california_housing.target, columns=["target"])
data = pd.concat([X, y], axis=1)

cities_df = pd.read_csv('https://raw.githubusercontent.com/lahongloc/HousePricePrediction/kiet/cal_cities_lat_long.csv')

cities_df_2 = pd.read_csv(
    'https://raw.githubusercontent.com/lahongloc/HousePricePrediction/kiet/cal_populations_city.csv')

with urllib.request.urlopen(
        'https://raw.githubusercontent.com/lahongloc/HousePricePrediction/kiet/california.png') as url:
    img = PIL.Image.open(url)
    california_img = np.array(img)

with urllib.request.urlopen(
        'https://raw.githubusercontent.com/lahongloc/HousePricePrediction/kiet/california-area-code-map.webp') as url:
    img = PIL.Image.open(url)
    california_img2 = np.array(img)

url = 'https://vi.wikipedia.org/wiki/Danh_s%C3%A1ch_%C4%91%C3%B4_th%E1%BB%8B_t%E1%BA%A1i_California'
html_data = requests.get(url).text
tables = pd.read_html(html_data, header=1)
cities_table = pd.DataFrame(tables[1])

cities_df['Name'] = cities_df['Name'].str.strip()
cities_df_2['City'] = cities_df_2['City'].str.strip()
cities_df = cities_df[cities_df['Name'].isin(cities_df_2['City'])]

cities_df['Latitude'] = cities_df['Latitude'].round(2)
cities_df['Longitude'] = cities_df['Longitude'].round(2)

data['Latitude'] = data['Latitude'].round(2)
data['Longitude'] = data['Longitude'].round(2)

coordinates = cities_df[['Latitude', 'Longitude']].values
tree = KDTree(coordinates)


def find_nearest_city(lat, lon):
    dist, ind = tree.query([[lat, lon]], k=1)  # k=1 tìm điểm gần nhất
    return cities_df.iloc[ind[0][0]]['Name']


data['City'] = data.apply(lambda row: find_nearest_city(row['Latitude'], row['Longitude']), axis=1)

city_to_country_map = cities_df_2.set_index('City')['Country'].to_dict()
data['Country'] = data['City'].map(city_to_country_map)
missing_cities = set(data['City']) - set(cities_df_2['City'])

cities_table = cities_table[['Loại', 'Quận']]
cities_table_unique = cities_table.drop_duplicates(subset=['Quận'], keep='first')
data = data.merge(cities_table_unique, left_on='Country', right_on='Quận', how='left')

data.drop(columns=['Quận'], inplace=True)

data.rename(columns={'Loại': 'Type'}, inplace=True)
data.rename(columns={'City': 'Area'}, inplace=True)
data.rename(columns={'Country': 'District'}, inplace=True)

type_encoder = LabelEncoder()
area_encoder = LabelEncoder()
district_encoder = LabelEncoder()

data['Type'] = type_encoder.fit_transform(data['Type'])
data['Area'] = area_encoder.fit_transform(data['Area'])
data['District'] = district_encoder.fit_transform(data['District'])


# print(data)

