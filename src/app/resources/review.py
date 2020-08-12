"""
Author: Mason Hu
------------------------------------------------------------------

This script contains the external representation for the review
resource.

------------------------------------------------------------------
"""

# Standard imports

# 3rd party imports
from flask_restful import Resource, reqparse

# Project level imports
from models.review import ReviewModel


########################################################################################################################
class ReviewResource(Resource):

    # Define the required payload content for the ReviewResource PSOT request
    # comment - review text from user
    # rating - rating provided by user
    # suggested_rating - rating predicted by sentiment model
    # sentiment_score - sentiment score between 0 and 1 provided by the model
    parser = reqparse.RequestParser()
    parser.add_argument('comment', type=str, required=True)
    parser.add_argument('rating', type=int, required=True)
    parser.add_argument('suggested_rating', type=int, required=True)
    parser.add_argument('sentiment_score', type=float, required=True)
    parser.add_argument('user_agent', type=str, required=True)
    parser.add_argument('ip_address', type=str, required=True)

    # Get all reviews for a particular company
    def get(self, company_id: str):
        company_id = company_id.split("-")[-1]  # endpoint input structure is companyid-<id>
        try:
            reviews = ReviewModel.find_by_company(company_id)
        except Exception as e:
            return {"msg": f"Failed with error: {e}"}, 500
        if reviews:
            return {"reviews": [review.json() for review in reviews]}, 200
        return {"msg": "No reviews found"}, 404

    # Send new review to DB for a company
    def post(self, company_id: str):
        company_id = company_id.split("-")[-1]
        payload = ReviewResource.parser.parse_args()
        review = ReviewModel(company_id=company_id, **payload)
        try:
            review.save_to_db()
            review.company.update_review_count()  # Update number of reviews
        except Exception as e:
            return {"msg": f"Failed with error: {e}"}, 500
        return review.json(), 201
