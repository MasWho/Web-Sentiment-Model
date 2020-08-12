"""
Author: Mason Hu
------------------------------------------------------------------

This script contains the external representation for the company
resource.

------------------------------------------------------------------
"""

# Standard imports

# 3rd party imports
from flask_restful import Resource

# Project level imports
from models.company import CompanyModel


########################################################################################################################
class CompanyResource(Resource):

    # Get a company by id from database
    def get(self, company_id):
        company_id = int(company_id.split("-")[-1])
        try:
            company = CompanyModel.find_by_id(company_id)
        except Exception as e:
            return {"msg": f"Failed with error: {e}"}, 500
        if company:
            company.update_review_count()  # Just incase if the review count didn't update for some reason
            return company.json(), 200
        return {"msg": f"Couldn't find company {company_id}"}, 404

    # Create a company by name
    # This should be reserved for admin roles
    def post(self):
        pass
