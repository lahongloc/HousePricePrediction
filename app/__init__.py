from urllib.parse import quote
from flask_sqlalchemy import SQLAlchemy
from flask import Flask

app = Flask(__name__)

app.secret_key = '56%^&*987^&*(098&*((*&^&*&'
app.config['SQLALCHEMY_DATABASE_URI'] = str.format(
    'mysql+pymysql://root:%s@localhost/housess?charset=utf8mb4'
    % quote('Omc6789#'))
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True

db = SQLAlchemy(app=app)
