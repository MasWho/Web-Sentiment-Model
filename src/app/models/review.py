"""
Author: Mason Hu
------------------------------------------------------------------

This script contains the internal representation for the
review resource.

------------------------------------------------------------------
"""

# Standard imports

# 3rd party imports

# Project level imports
from main.db import db


########################################################################################################################
# Review model for the rating resource

class ReviewModel(db.Model):
    # Specify the db table name for sqlalchemy to link with this model
    __tablename__ = 'reviews'

    # Specify the table columns for sqlalchemy to link with this model
    # The column variables must match with the instance attributes except for primary key
    id = db.Column(db.Integer, primary_key=True)
    comment = db.Column(db.String)
    rating = db.Column(db.SmallInteger)
    suggested_rating = db.Column(db.SmallInteger)
    sentiment_score = db.Column(db.Float)
    # Define foreign key in the reviews table that links each entry to a single entry in the companies table.
    company_id = db.Column(db.Integer, db.ForeignKey('companies.id'))
    company = db.relationship('CompanyModel')
    user_agent = db.Column(db.String)
    ip_address = db.Column(db.String)

    def __init__(self, comment: str, rating: int,
                 suggested_rating: int,
                 sentiment_score: float,
                 company_id: int,
                 user_agent: str,
                 ip_address: str):
        self.comment = comment
        self.rating = rating
        self.suggested_rating = suggested_rating
        self.sentiment_score = sentiment_score
        self.company_id = company_id
        self.user_agent = user_agent
        self.ip_address = ip_address

    def json(self):
        return {'comment': self.comment,
                'rating': self.rating,
                'suggested_rating': self.suggested_rating,
                'sentiment_score': self.sentiment_score,
                'company': self.company.name,
                'user_agent': self.user_agent,
                'ip_address': self.ip_address}

    #  Get all reviews for a particular company
    @classmethod
    def find_by_company(cls, company_id: int):
        return cls.query.filter_by(company_id=company_id).all()

    def save_to_db(self):
        db.session.add(self)
        db.session.commit()

    def delete_from_db(self):
        db.session.delete(self)
        db.session.commit()
