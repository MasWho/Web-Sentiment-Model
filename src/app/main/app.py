"""
Author: Mason Hu
------------------------------------------------------------------

This script contains all top level Flask REST API definitions and
resource setups.

------------------------------------------------------------------
"""

# Standard imports
import os

# 3rd party imports
from flask import Flask
from flask_restful import Api
import wget

# Project level imports
from resources.model import ModelResource
from resources.company import CompanyResource
from resources.review import ReviewResource

########################################################################################################################
# Define app and setups
app = Flask(__name__)
DB_URL = 'postgresql+psycopg2://{user}:{pw}@{url}/{db}'.format(
    user=os.environ.get("POSTGRES_USER"),
    pw=os.environ.get("POSTGRES_PW"),
    url=os.environ.get("POSTGRES_URL"),
    db=os.environ.get("POSTGRES_DB")
)
app.config['SQLALCHEMY_DATABASE_URI'] = DB_URL  # The postgres url components are stored as local env_vars
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False  # Save resources
app.secret_key = 'no_need_since_no_encryption_but_good_to_set_anyway'
api = Api(app)

# Download the model if it's not already stored locally
github_model_path = 'https://github.com/MasWho/Web-Sentiment-Model/releases/download/v0.0.0/model_trustpilot.pth'
model_name = 'model_trustpilot.pth'
model_path = './models/ai'

if model_name not in os.listdir(model_path):
    print(f'Downloading the trained model {model_name}')
    wget.download(
        github_model_path,
        out=model_path
    )
else:
    print('Model already saved to app/models/ai')

# Add resources to the REST API
api.add_resource(ModelResource, '/api/model')
api.add_resource(CompanyResource, '/api/<string:company_id>')
api.add_resource(ReviewResource, '/api/<string:company_id>/review')

if __name__ == "__main__":
    pass
