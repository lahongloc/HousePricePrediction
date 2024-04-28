from app import app, db
from sqlalchemy import Column, Integer, String, Boolean, ForeignKey, Enum, DateTime, Float
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing


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


if __name__ == "__main__":
    with app.app_context():
        db.create_all()
        california_housing = fetch_california_housing()
        X = pd.DataFrame(california_housing.data, columns=california_housing.feature_names)
        y = pd.DataFrame(california_housing.target, columns=["target"])
        a = pd.concat([X, y], axis=1)

        for i in range(0, len(a)):
            h = Housing(med_inc=a["MedInc"][i],
                        house_age=a["HouseAge"][i],
                        ave_rooms=a["AveRooms"][i],
                        ave_bedrms=a["AveBedrms"][i],
                        population=a["Population"][i],
                        ave_occup=a["AveOccup"][i],
                        latitude=a["Latitude"][i],
                        longitude=a["Longitude"][i],
                        target=a["target"][i])
            db.session.add(h)
        db.session.commit()
