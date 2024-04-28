from flask_admin.contrib.sqla import ModelView
from flask_admin import Admin, BaseView, expose
from app import app, db
from flask import redirect
from app.models import Housing

admin = Admin(app=app, name="Houses Management Page", template_mode='bootstrap4')


class HomePage(BaseView):
    @expose('/')
    def index(self):
        return redirect('/')


admin.add_view(ModelView(Housing, db.session))
admin.add_view(HomePage(name='Home Page'))