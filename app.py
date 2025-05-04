# app.py
from flask import Flask
from flask_migrate import Migrate
from database import db, initialize_database

def create_app():
    app = Flask(__name__)
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///aegan.db'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    # Initialize database
    db.init_app(app)
    Migrate(app, db)
    
    # Create tables and admin user
    initialize_database(app)

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)