import pandas as pd
from app import dao
from app.MLModels import house_price_prediction_models

# print(house_price_prediction_models.region_encoder.classes_)
# print(house_price_prediction_models.data['Region'].unique())

# area = pd.concat([pd.DataFrame(house_price_prediction_models.region_encoder.classes_),
#                   house_price_prediction_models.data['Region'].unique()], axis=1)

# region = pd.DataFrame()
#
# region['name'] = house_price_prediction_models.region_encoder.classes_
# region['encoded_name'] = house_price_prediction_models.data['Region'].unique()
# print(region.head())

# print(house_price_prediction_models.data['Region'].isna())
# print(house_price_prediction_models.data)

a = [{
    'longitude': -122.23,
    'latitude': 37.88
}]
a_df = pd.DataFrame(a)
distances, indices = house_price_prediction_models.tree.query(a_df[['longitude', 'latitude']], k=1)
a_df['Region'] = house_price_prediction_models.regions_df.iloc[indices.flatten()]['name'].values
region = dao.get_region_details(a_df['Region'][0])
print(region[0])
