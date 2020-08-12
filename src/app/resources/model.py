"""
Author: Mason Hu
------------------------------------------------------------------

This script contains the external representation for the model
resource.

------------------------------------------------------------------
"""

# Standard imports

# 3rd party imports
from flask_restful import Resource, reqparse

# Project level imports
from models.model import ModelModel


########################################################################################################################
# Define the model resources
class ModelResource(Resource):
    # Create a class level parser for enforcing payload contents
    # The parser will be used to look through the payload
    # The arguments specified by the parser will be collected
    # This is useful that specific components in the payload is passed in as required
    parser = reqparse.RequestParser()
    parser.add_argument('review',
                        type=str,
                        required=True,
                        help="This field is required."
                        )

    # Retrieve text review input from front end
    # Then make a sentiment classfication
    def post(self):
        payload = ModelResource.parser.parse_args()
        # if the review is too short, it's a bad request
        if len(payload['review']) < 10:
            return {"msg": "Review is too short! It's less than 10 characters!"}, 400
        model = ModelModel()
        try:
            model.predict(payload['review'])
        except Exception as e:
            return {"msg": f"Failed with error:{e}"}, 500  # model failed, internal server error
        return model.json()

    # Basic CRUD request types. Some may not be used and serves as a placeholder.
    def get(sefl):
        pass

    def delete(self):
        pass

    def put(self):
        pass
