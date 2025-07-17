# analyzer

from flask import Flask, jsonify, request # Import request
from flask_cors import CORS

import requests
import json
import os
import logging # Import logging

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# It's better to store your API key as an environment variable
api_key = os.environ.get("GEMINI_PERSONAL_API_KEY") # Using GEMINI_PERSONAL_API_KEY as per your code

if not api_key:
    logging.error("Error: GEMINI_PERSONAL_API_KEY environment variable not set at app startup.")

@app.route('/')
def hello_world():
    return 'Conrad deployed another AI Flask App!'

@app.route('/explain_ai')
def explain_ai():
    if not api_key:
        logging.error("API key missing in /explain_ai request.")
        return jsonify({"error": "GEMINI_PERSONAL_API_KEY environment variable not set."}), 500 # Return 500 if key is missing

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
    headers = {'Content-Type': 'application/json'}
    data = {
        "contents": [
            {
                "parts": [{"text": "Explain how AI works"}]
            }
        ]
    }

    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        response.raise_for_status()  # Raise an exception for bad status codes (such as 4xx or 5xx)
        response_json = response.json()
        return jsonify(response_json)
    except requests.exceptions.RequestException as e:
        logging.error(f"Error during API call to Gemini in /explain_ai: {e}")
        return jsonify({"error": f"Error during API call: {e}"})
    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON response from Gemini in /explain_ai. Response content: {response.text}")
        return jsonify({"error": "Error decoding JSON response from Gemini."})

@app.route('/countycityfromcoordinates')
def get_county_city_from_coordinates():
    if not api_key:
        logging.error("API key missing in /gemini-chat request.")
        return jsonify({"error": "GEMINI_PERSONAL_API_KEY environment variable not set."}), 500

    decimal_latitude = request.args.get('latitude')
    decimal_longitude = request.args.get('longitude')
    coordinate_uncertainty = request.args.get('coordinate_uncertainty')


    content_received = f""""Received user_input: decimal_latitude'{decimal_latitude}' 
    and decimal_longitude '{decimal_longitude}'
    and coordinate_uncertainty '{coordinate_uncertainty}
    """

    logging.info(content_received) # Log the received input

    if not decimal_latitude or not decimal_longitude:
        return jsonify({"error": "Missing 'latitude' or longitude parameter in the query string."})

    # Construct the prompt for the Gemini API
    prompt_text = f"""what county and city is the following in? Make the response like so:

    {{"county": [County name],
    "city/town": [city/town name]}}

    decimal latitude: {decimal_latitude}
    decimal longitude: {decimal_longitude}
    coordinate uncertainty: {coordinate_uncertainty if coordinate_uncertainty else 'not specified'}""" # Added a fallback for uncertainty


    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
    headers = {'Content-Type': 'application/json'}
    data = {
        "contents": [
            {
                "parts": [
                    {"text": prompt_text}
                ]
            }
        ]
    }
    logging.info(f"Sending data to Gemini: {json.dumps(data)}") # Log the payload


    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        response.raise_for_status()  # Raise an exception for bad status codes
        response_json = response.json()
        logging.info("Gemini API call successful for /gemini-chat.")
        return jsonify(response_json)
    except requests.exceptions.RequestException as e:
        the_error_msg = f"""Error during API call to Gemini in /gemini-chat: {e}. 
        Response status: {response.status_code}, content: {response.text}"""
        logging.error(the_error_msg)
        return jsonify({"error": f"Error during API call: {e}"}), 500
    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON response from Gemini in /gemini-chat. Response content: {response.text}")
        return jsonify({"error": "Error decoding JSON response."}), 500
     

@app.route('/aichat')
def aichat_endpoint(): # Renamed the function to resolve the conflict
    if not api_key:
        return jsonify({"error": "GEMINI_PERSONAL_API_KEY environment variable not set."})

    user_input = request.args.get('text')

    if not user_input:
        return jsonify({"error": "Missing 'text' parameter in the query string."})

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
    headers = {'Content-Type': 'application/json'}
    data = {
        "contents": [
            {
                "parts": [{"text": user_input}]
            }
        ]
    }

    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        response.raise_for_status()  # Raise an exception for bad status codes
        response_json = response.json()
        return jsonify(response_json)
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Error during API call: {e}"})
    except json.JSONDecodeError:
        return jsonify({"error": "Error decoding JSON response."})
    

if __name__ == '__main__':
    app.run(debug=True)
