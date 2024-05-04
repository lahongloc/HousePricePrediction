import numpy as np
import pandas as pd
import PIL.Image
import urllib.request
import requests
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.datasets import fetch_california_housing
from sklearn.neighbors import KDTree
import warnings
from shapely.geometry import shape, Point

warnings.filterwarnings("ignore")
california_housing = fetch_california_housing()
X = pd.DataFrame(california_housing.data, columns=california_housing.feature_names)
y = pd.DataFrame(california_housing.target, columns=["target"])
data = pd.concat([X, y], axis=1)

with urllib.request.urlopen(
        'https://raw.githubusercontent.com/lahongloc/HousePricePrediction/kiet/california.png') as url:
    img = PIL.Image.open(url)
    california_img = np.array(img)

url = 'https://vi.wikipedia.org/wiki/Danh_s%C3%A1ch_%C4%91%C3%B4_th%E1%BB%8B_t%E1%BA%A1i_California'
html_data = requests.get(url).text

tables = pd.read_html(html_data, header=1)
cities_table = pd.DataFrame(tables[1])

url = 'https://raw.githubusercontent.com/codeforgermany/click_that_hood/main/public/data/california-counties.geojson'
response = requests.get(url)
geo_data = response.json()

regions = []
for feature in geo_data['features']:
    geom = shape(feature['geometry'])
    centroid = geom.centroid
    regions.append({
        'name': feature['properties']['name'],
        'latitude': centroid.y,
        'longitude': centroid.x
    })
regions_df = pd.DataFrame(regions)
tree = KDTree(regions_df[['longitude', 'latitude']])
distances, indices = tree.query(data[['Longitude', 'Latitude']], k=1)
data['Region'] = regions_df.iloc[indices.flatten()]['name'].values

avg_price_per_region = data.groupby('Region')['target'].mean().reset_index()
region_encoder = LabelEncoder()

# Fit và transform cho mỗi cột
data['Region'] = region_encoder.fit_transform(data['Region'])
target = data['target']
data.drop(columns='target', inplace=True)
data['target'] = target

X = data.drop(['target', 'AveBedrms', 'Population'], axis=1)
y = data['target']

data = pd.concat([X, y], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Huấn luyện scaler và biến đổi dữ liệu huấn luyện
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)  # Convert back to DataFrame and add column names
X_test_scaled = scaler.transform(X_test)

lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)

rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train_scaled, y_train.values.ravel())

param_grid = {
    'n_estimators': [200],  # Số lượng cây
    'max_features': ['sqrt'],  # Số lượng đặc trưng để xem xét tại mỗi phân chia
    'max_depth': [None],  # Độ sâu tối đa của cây
    'min_samples_split': [2],  # Số lượng mẫu tối thiểu cần thiết để phân chia một nút
    'min_samples_leaf': [1]  # Số lượng mẫu tối thiểu tại mỗi lá
}
rf = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, verbose=2, scoring='neg_mean_squared_error',
                           n_jobs=-1)
grid_search.fit(X_train_scaled, y_train.values.ravel())
best_rf_model = grid_search.best_estimator_

gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gb_model.fit(X_train_scaled, y_train.values.ravel())


def predict_new_data(model, new_data):
    new_data_scaled = scaler.transform(np.array([new_data]))
    return model.predict(new_data_scaled)[0]


# new_data = [8.3252, 41.0, 6.984127, 322.0, 2.555556, 37.88, 37]
# predicted_price = predict_new_data(best_rf_model, new_data)
# print(f"Dự đoán giá nhà mới: ${predicted_price * 100000}")

