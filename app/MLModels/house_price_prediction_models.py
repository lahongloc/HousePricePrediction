import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import warnings

from sqlalchemy import create_engine
from urllib.parse import quote

password = quote('Omc6789#')  # Mã hóa ký tự đặc biệt trong mật khẩu
connection_string = f'mysql+pymysql://root:{password}@localhost/calihouse?charset=utf8mb4'
engine = create_engine(connection_string)

data = pd.DataFrame(pd.read_sql_query("SELECT * FROM housing", con=engine))
y = data['target']
X = data.drop(columns=['target', 'id'])
warnings.filterwarnings("ignore")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
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


new_data = [8.3252, 41.0, 6.984127, 1.023810, 322.0, 2.555556, 37.88, -122.23, 293, 49, 0]
predicted_price = predict_new_data(best_rf_model, new_data)
print(f"Dự đoán giá nhà mới: ${predicted_price * 100000}")
