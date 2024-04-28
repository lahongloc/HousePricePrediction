from app import app
from flask import render_template, request
from app import dao
from app.MLModels import house_price_prediction_models


@app.route("/")
def home():
    houses = dao.get_house()
    return render_template("index.html", houses=houses)


@app.route("/", methods=['get', 'post'])
def implement_prediction():
    if request.method.__eq__("POST"):
        med_inc = float(request.form.get("MedInc"))
        house_age = float(request.form.get("HouseAge"))
        ave_rooms = float(request.form.get("AveRooms"))
        ave_bedrms = float(request.form.get("AveBedrms"))
        population = int(request.form.get("Population"))
        ave_occup = float(request.form.get("AveOccup"))
        latitude = float(request.form.get("Latitude"))
        longitude = float(request.form.get("Longitude"))

        al = request.form.get('al')
        model = ''
        if al == 'lr':
            model = house_price_prediction_models.lr_model
        if al == 'rf':
            model = house_price_prediction_models.rf_model
        if al == 'gb':
            model = house_price_prediction_models.gb_model

        new_data = [med_inc, house_age, ave_rooms, ave_bedrms, population, ave_occup, latitude, longitude]
        predicted = house_price_prediction_models.predict_new_data(model, new_data)

    return render_template("index.html", predicted_price=(predicted * 100000))


if __name__ == "__main__":
    from app.admin import *

    app.run(debug=True)
