"""
This script is running on top of app.py mostly for avoiding 
circular import of db.py
"""

from main.app import app
from main.db import db
from models.company import CompanyModel
from pandas import read_csv

db.init_app(app)


# This decorator will cause the method to be called before any requests made to the app
# sqlalchemy knows which tables, and corresponding columns it must create
# since in the model classes, we've defined __tablename__ and all of the columns
# Note, all of the model classes extends the SQLAlchemy.Model class.
# The database / tables will be created only if it doesn't exist already.
# important to import the models for sqlalchemy where tables must be created
@app.before_first_request
def create_tables():
    db.create_all()
    # Check if the existing company table contain data, if not then initialise with csv
    s = db.session()
    if len(s.query(CompanyModel).all()) == 0:
        print('No data in the companies table detected.')
        print('Initialising the companies table in database.')
        engine = s.get_bind()
        df = read_csv('./companies.csv')
        # reset column names in dataframe to correspond with table column names
        # this must be done to insert the data properly into the table
        df.rename(columns={'Unnamed: 0': 'id',
                           'company_name': 'name',
                           'company_logo': 'logo_url',
                           'company_website': 'website_url'}, inplace=True)
        df['review_count'] = 0
        df.to_sql('companies',
                  con=engine,
                  if_exists='append',
                  chunksize=1000,
                  index=False)


# Don't need this when runnign with UWSGI on a remote server!!!
app.run(port=5000, debug=True)
