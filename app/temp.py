from app.Data import houseData
import pandas as pd

print(houseData.find_nearest_city(37.88, -122.23))
# print(len(houseData.district_encoder.classes_))
# print(len(houseData.data['District'].unique()))

district = pd.DataFrame(houseData.area_encoder.classes_)
district_encoded = pd.DataFrame(houseData.data['Area'].unique())

# temp = pd.concat([district, district_encoded], axis=1)
area = pd.DataFrame()
area['district'] = district
area['district_encoded'] = district_encoded
print(area)




