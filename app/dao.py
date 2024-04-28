from app import app, db
from app.models import Housing


def get_house():
    with app.app_context():
        houses = Housing.query.all()
        return houses
