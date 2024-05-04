from app import app, db
from app.models import Housing, Region


def get_house():
    with app.app_context():
        houses = Housing.query.all()
        return houses


def get_region_details(name):
    with app.app_context():
        region = db.session.query(Region.encoded_name).filter(Region.name.contains(name))
        return region.first()

        # house_details = db.session.query(Area.area, Area.district, Area.type).filter(Area.area_name.contains(name))
        # return house_details.first()
