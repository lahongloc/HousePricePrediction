from app import app, db
from sqlalchemy import Column, Integer, String, Boolean, ForeignKey, Enum, DateTime, Float
import pandas as pd
from sklearn.datasets import fetch_california_housing
from app.Data import houseData


class Housing(db.Model):
    id = Column(Integer, primary_key=True, autoincrement=True)
    med_inc = Column(Float)
    house_age = Column(Integer)
    ave_rooms = Column(Float)
    ave_bedrms = Column(Float)
    population = Column(Integer)
    ave_occup = Column(Float)
    latitude = Column(Float)
    longitude = Column(Float)
    target = Column(Float)
    area = Column(Float)
    district = Column(Integer)
    type = Column(Integer)


class Area(db.Model):
    id = Column(Integer, primary_key=True, autoincrement=True)
    area_name = Column(String(50))
    area = Column(Integer)
    district = Column(Integer)
    type = Column(Integer)


if __name__ == "__main__":
    with app.app_context():
        db.create_all()

        kv = pd.DataFrame(houseData.area_encoder.classes_)
        kv_encoded = pd.DataFrame(houseData.data['Area'].unique())
        temp = pd.DataFrame()
        temp['area_name'] = kv
        temp['Area'] = kv_encoded

        #
        area = houseData.data[['Area', 'District', 'Type']]
        area = area.drop_duplicates()

        rs = pd.merge(temp, area, on="Area", how="inner")
        # print(rs)

        for i in range(0, len(rs)):
            a = Area(area_name=rs["area_name"][i],
                     area=rs["Area"][i],
                     district=rs["District"][i],
                     type=rs["Type"][i])
            db.session.add(a)
            db.session.commit()
        #
        a = houseData.data
        for i in range(0, len(a)):
            h = Housing(med_inc=a["MedInc"][i],
                        house_age=a["HouseAge"][i],
                        ave_rooms=a["AveRooms"][i],
                        ave_bedrms=a["AveBedrms"][i],
                        population=a["Population"][i],
                        ave_occup=a["AveOccup"][i],
                        latitude=a["Latitude"][i],
                        longitude=a["Longitude"][i],
                        target=a["target"][i],
                        area=a['Area'][i],
                        district=a['District'][i],
                        type=a['Type'][i])
            db.session.add(h)
            db.session.commit()
