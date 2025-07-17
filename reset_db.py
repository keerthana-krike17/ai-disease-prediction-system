from app import db, app

with app.app_context():
    db.drop_all()
    db.create_all()
    print("✅ Fresh database created with all current models.")
