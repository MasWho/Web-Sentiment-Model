"""
Author: Mason Hu
------------------------------------------------------------------

This script contains all top level Flask REST API definitions and
resource setups.

------------------------------------------------------------------
"""

# Standard import
import os

# 3rd party import
import requests
from flask import request, Flask
import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from numpy.random import randint

# Project level import


########################################################################################################################
# External style sheets for the dash front end
# fontawesome icons
# Bootstrap css
# Google api icons
external_stylesheets = [
    "./assets/local_style.css",
    "https://fonts.googleapis.com/css2?family=Raleway:wght@600&display=swap",
    "https://use.fontawesome.com/releases/v5.0.7/css/all.css",
    'https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css'
]
api_url = os.environ.get('API_URL', 'http://0.0.0.0:5000')

# External JS scripts


# Define app
server = Flask(__name__)  # define flask app.server

app = dash.Dash(
    __name__,
    server=server,
    external_stylesheets=external_stylesheets,
    meta_tags=[
        {'name': 'viewport', 'content': 'width=device-width, initial-scale=1'}
    ],
    suppress_callback_exceptions=True
)

# # Define app layout
app.title = 'Reviews'
# app.layout = html.Div([
#     dcc.Location(id='url', refresh=False),
#     html.Div(id='page-content')
# ])

# Define front-end layout
app.layout = html.Div(
    [
        # ------- Company logo section -------
        html.Div(
            [
                html.A(
                    html.Img(
                        id='company_logo',
                        style={
                            'height': '100px',
                            'width': '370px',
                            'padding': '5px',
                            'object-fit': 'contain'
                        }
                    ),
                    id='company_link',
                    target='_blank'
                )
            ],
            style={
                'height': '100px',
                'backgroundColor': 'white',
                'borderRadius': '5px',
                'opacity': '0.7'
            }
        ),
        # ------- Company name section -------
        html.Div(
            html.H3(
                [
                    "What do you think of ",
                    html.Span(
                        id='company_name'
                    ),
                    " ?",
                    html.Span(
                        id='company_id',
                        className="d-none"
                    )
                ],
                className="h3 mb-3 mt-3 font-weight-normal text-light",
                # style={
                #     'color': 'white'
                # }
            ),
            id='company-name-div',
            className='d-flex adjust-content-center align-items-center',
            style={
                'height': '140px'
            }
        ),
        # ------- User text input section -------
        html.Div(
            [
                dcc.Textarea(
                    id="review",
                    className="form-control z-depth-1",
                    rows="8",
                    placeholder="Write something here..."
                )
            ],
            className="form-group shadow-textarea mb-4",
            style={
                'opacity': '0.7'
            }
        ),
        # ------- Sentiment analysis label -------
        html.H5(
            'SENTIMENT',
            className="text-light mt-4"
        ),
        # ------- Progress bar section -------
        dbc.Progress(
            html.Span(
                id='proba',
                className="text-dark"
            ),
            id="progress",
            striped=False,
            animated=False,
            className="mb-3"
        ),
        # ------- Propose rating label -------
        html.H5(
            'PROPOSE A RATING',
            className='text-light'
        ),
        # ------- Rating slider section -------
        html.Div(
            [
                dcc.Slider(
                    id='rating',
                    min=1,
                    max=5,
                    step=1,
                    marks={str(i): {'label': str(i),
                                    'style': {'color': 'white', 'font-size': 'large'}} for i in range(1, 6)}
                ),
            ],
            style={'marginBottom': '30px', 'font-size': 'large'}
        ),
        # ------- Submit button section -------
        html.Button(
            [
                html.Span(
                    'Submit',
                    className='mr-3'
                ),
                html.I(
                    className='fa fa-paper-plane m-l-7'
                )
            ],
            id='submit_button',
            className='btn btn-lg btn-success btn-block',
            role='submit',
            n_clicks_timestamp=0
        ),
        # ------- Switch button section -------
        html.Button(
            [
                html.Span(
                    'Review another brand',
                    className='mr-3'
                ),
                html.I(
                    className='fa fa-sync-alt'
                )
            ],
            id='switch_button',
            className='btn btn-lg btn-secondary btn-block',
            n_clicks_timestamp=0
        ),
        # ------- Admin switch section -------
        html.P(
            dcc.Link("Go to Admin", id="admin-link", href="/admin"),
            className="mt-2 mb-5 d-none"
        ),
        # ------- Author section -------
        html.H4(
            [
                html.A(
                    [
                        "MasWho@",
                        html.I(
                            className="fab fa-github",
                            style={
                                'margin-left': '10px',
                                'width': '20px',
                                'height': '20px'
                            }
                        )
                    ],
                    href='https://github.com/MasWho',
                    target="_blank")
            ],
            className='h4 mt-5 text-muted'
        )
    ],
    id='top-div',
    className='container text-center mt-5 form-review',
    style={
        'width': '400px'
    }
)


########################################################################################################################
# Callback functions

# Update the company logo and name when the submit button is clicked
# Insert the review data
@app.callback(
    [
        Output('company_logo', 'src'),
        Output('company_name', 'children'),
        Output('company_link', 'href'),
        Output('company_id', 'children'),
        Output('review', 'value')
    ],
    [
        Input('submit_button', 'n_clicks_timestamp'),
        Input('switch_button', 'n_clicks_timestamp')
    ],
    [
        State('company_id', 'children'),
        State('review', 'value'),
        State('rating', 'value'),
        State('progress', 'value')
    ]
)
def change_brand(submit_click_ts, switch_click_ts, company_id, review, rating, score):
    host = api_url
    # If the submit button is clicked, then send current data to database
    # Note the submit button is disabld if review is less than characters
    if submit_click_ts > switch_click_ts:
        endpoint_post = "/".join([host, "api", f"companyid-{company_id}", "review"])
        user_agent = request.headers.get('User-Agent')
        ip_address = request.remote_addr
        payload = {
            'comment': review,
            'rating': rating,
            'suggested_rating': rating,
            'sentiment_score': score,
            'user_agent': user_agent,
            'ip_address': ip_address
        }
        response = requests.post(endpoint_post, payload)
        if response.ok:
            print("Review Saved")
        else:
            print("Error Saving Review")
    # Randomly generate a company id and query it from database
    company_id_new = randint(0, 10623)
    host = api_url
    endpoint_get = "/".join([host, "api", f"companyid-{company_id_new}"])
    response = requests.get(endpoint_get).json()
    if not response['logo_url'].startswith('http'):
        response['logo_url'] = 'https://' + response['logo_url']
    return response['logo_url'], response['name'], response['website_url'], company_id_new, ''


# Update the progress bar with the sentiment score based on user text input
@app.callback(
    [
        Output('proba', 'children'),
        Output('progress', 'value'),
        Output('progress', 'color'),
        Output('rating', 'value'),
        Output('submit_button', 'disabled')
    ],
    [
        Input('review', 'value')
    ]
)
def update_proba(review):
    host = api_url
    endpoint = "/".join([host, "api", "model"])
    if review and len(review) >= 10:
        payload = {
            "review": review
        }
        response = requests.post(endpoint, payload).json()
        proba = round(response['sentiment_score'] * 100, 2)
        suggested_rating = min(int((proba / 100) * 5 + 1), 5)
        if proba >= (2 / 3) * 100:
            color = 'success'
        elif proba >= (1 / 3) * 100 and proba < (2 / 3) * 100:
            color = 'warning'
        elif proba < (1 / 3) * 100:
            color = 'danger'
        return f"{proba}%", proba, color, suggested_rating, False
    else:
        return "0.00%", 0, None, 0, True


########################################################################################################################
if __name__ == '__main__':
    app.run_server(debug=True)
