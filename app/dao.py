from app import app, db
from app.models import Housing, Area


def get_house():
    with app.app_context():
        houses = Housing.query.all()
        return houses


def get_area_details(name):
    with app.app_context():
        house_details = db.session.query(Area.area, Area.district, Area.type).filter(Area.area_name.contains(name))
        return house_details.first()
