from app import app, db
from sqlalchemy import Column, Integer, String, Boolean, ForeignKey, Enum, DateTime, Float
import pandas as pd
from sklearn.datasets import fetch_california_housing
from app.Data import houseData
from app.MLModels import house_price_prediction_models


class Housing(db.Model):
    id = Column(Integer, primary_key=True, autoincrement=True)
    med_inc = Column(Float)
    house_age = Column(Integer)
    ave_rooms = Column(Float)
    ave_occup = Column(Float)
    latitude = Column(Float)
    longitude = Column(Float)
    region = Column(Integer)
    target = Column(Float)


class Region(db.Model):
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(50))
    encoded_name = Column(Integer)


if __name__ == "__main__":
    with app.app_context():
        db.create_all()

        a = house_price_prediction_models.data

        for i in range(0, len(a)):
            h = Housing(med_inc=a["MedInc"][i],
                        house_age=a["HouseAge"][i],
                        ave_rooms=a["AveRooms"][i],
                        ave_occup=a["AveOccup"][i],
                        latitude=a["Latitude"][i],
                        longitude=a["Longitude"][i],
                        region=a["Region"][i],
                        target=a["target"][i])
            db.session.add(h)
        db.session.commit()

        region = pd.DataFrame()

        region['name'] = house_price_prediction_models.region_encoder.classes_
        region['encoded_name'] = house_price_prediction_models.data['Region'].unique()
        for i in range(0, len(region)):
            r = Region(name=region["name"][i],
                       encoded_name=region["encoded_name"][i])
            db.session.add(r)
        db.session.commit()

        # ================
