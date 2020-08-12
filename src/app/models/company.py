"""
Author: Mason Hu
------------------------------------------------------------------

This script contains the internal representation of companies for
the company resource.

------------------------------------------------------------------
"""

# Standard imports

# 3rd party imports

# Project level imports
from main.db import db


########################################################################################################################
class CompanyModel(db.Model):
    # Specify the db table name for sqlalchemy to link with this model
    __tablename__ = 'companies'

    # Specify the table columns for sqlalchemy to link with this model
    # The column variables must match with the instance attributes
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String)
    logo_url = db.Column(db.String)
    website_url = db.Column(db.String)
    # Evaluate lazily so it doesn't take up too much resource everytime a company is created
    reviews = db.relationship('ReviewModel', lazy='dynamic')
    review_count = db.Column(db.Integer)

    def __init__(self, name: str=None, logo_url: str=None, website_url: str=None):
        self.name = name
        self.logo_url = logo_url
        self.website_url = website_url
        self.review_count = len(self.reviews.all())

    def json(self):
        return {'id': self.id,
                'name': self.name,
                'logo_url': self.logo_url,
                'website_url': self.website_url,
                'review_count': self.review_count}

    # Find a company in the database by name
    @classmethod
    def find_by_id(cls, _id):
        return cls.query.filter_by(id=_id).first()

    def save_to_db(self):
        db.session.add(self)
        db.session.commit()

    def delete_from_db(self):
        db.session.delete(self)
        db.session.commit()

    def update_review_count(self):
        self.review_count = len(self.reviews.all())
        self.save_to_db()
