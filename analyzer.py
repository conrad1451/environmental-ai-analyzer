# analyzer.py

from flask import Flask, jsonify, request
from flask_cors import CORS
import requests
import json
import os
import logging
import concurrent.futures

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# It's better to store your API key as an environment variable
api_key = os.environ.get("GEMINI_SCHOOL_API_KEY")

if not api_key:
    logging.error("Error: GEMINI_SCHOOL_API_KEY environment variable not set at app startup.")


# --- Configuration for Native Batching ---
# These should be set as environment variables or secrets for your Flask app
GCS_INPUT_BUCKET = os.environ.get("GCS_INPUT_BUCKET")
GCS_OUTPUT_BUCKET = os.environ.get("GCS_OUTPUT_BUCKET")
GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID") # Your Google Cloud Project ID

# Initialize GCS client (will use GOOGLE_APPLICATION_CREDENTIALS by default)
try:
    storage_client = storage.Client(project=GCP_PROJECT_ID)
    logger.info("Google Cloud Storage client initialized successfully.")
except Exception as e:
    logger.error(f"Error initializing Google Cloud Storage client: {e}. Ensure GOOGLE_APPLICATION_CREDENTIALS is set.")
    storage_client = None # Set to None if initialization fails

# Initialize Generative Language client for batch operations
# This also uses GOOGLE_APPLICATION_CREDENTIALS by default
try:
    # Use ClientOptions to specify the endpoint, if necessary (e.g., for specific regions)
    # For global endpoint, this might not be strictly needed but good practice.
    genai_client_options = ClientOptions(api_endpoint="generativelanguage.googleapis.com")
    genai_client = glm.GenerativeServiceClient(client_options=genai_client_options)
    logger.info("Generative Language client initialized successfully.")
except Exception as e:
    logger.error(f"Error initializing Generative Language client: {e}. Ensure GOOGLE_APPLICATION_CREDENTIALS is set.")
    genai_client = None # Set to None if initialization fails



@app.route('/')
def hello_world():
    return 'Conrad deployed another AI Flask App!'

@app.route('/explain_ai')
def explain_ai():
    if not api_key:
        logging.error("API key missing in /explain_ai request.")
        return jsonify({"error": "GEMINI_SCHOOL_API_KEY environment variable not set."}), 500

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
        response.raise_for_status()
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
        logging.error("API key missing in /countycityfromcoordinates request.")
        return jsonify({"error": "GEMINI_SCHOOL_API_KEY environment variable not set."}), 500

    decimal_latitude = request.args.get('latitude')
    decimal_longitude = request.args.get('longitude')
    coordinate_uncertainty = request.args.get('coordinate_uncertainty')


    content_received = f""""Received user_input: decimal_latitude'{decimal_latitude}'
    and decimal_longitude '{decimal_longitude}'
    and coordinate_uncertainty '{coordinate_uncertainty}'
    """

    logging.info(content_received)

    if not decimal_latitude or not decimal_longitude:
        return jsonify({"error": "Missing 'latitude' or longitude parameter in the query string."})

    # Construct the prompt for the Gemini API
    prompt_text = f"""what county and city is the following in?
    decimal latitude: {decimal_latitude}
    decimal longitude: {decimal_longitude}
    coordinate uncertainty: {coordinate_uncertainty if coordinate_uncertainty else 'not specified'}"""

    # CHQ: Gemini AI added generationConfig to specify structured output
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
    headers = {'Content-Type': 'application/json'}
    data = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": prompt_text}
                ]
            }
        ],
        "generationConfig": { # NEW: Specify structured output
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "OBJECT",
                "properties": {
                    "county": {"type": "STRING"},
                    "city/town": {"type": "STRING"}
                },
                "propertyOrdering": ["county", "city/town"]
            }
        }
    }
    logging.info(f"Sending data to Gemini: {json.dumps(data)}")


    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        response.raise_for_status()
        response_json = response.json()
        logging.info("Gemini API call successful for /countycityfromcoordinates.")
        
        # Parse the structured response directly
        if response_json.get('candidates') and response_json['candidates'][0].get('content') and response_json['candidates'][0]['content'].get('parts'):
            # The content will already be parsed JSON if responseMimeType is application/json
            parsed_gemini_response = json.loads(response_json['candidates'][0]['content']['parts'][0]['text'])
            return jsonify(parsed_gemini_response) # Return only the relevant part
        else:
            logging.warning(f"Gemini response did not contain expected structured content: {response_json}")
            return jsonify({"error": "Gemini response did not contain expected structured content."}), 500
    except requests.exceptions.RequestException as e:
        the_error_msg = f"""Error during API call to Gemini in /countycityfromcoordinates: {e}.
        Response status: {response.status_code if response else 'N/A'}, content: {response.text if response else 'N/A'}"""
        logging.error(the_error_msg)
        return jsonify({"error": f"Error during API call: {e}"}), 500
    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON response from Gemini in /countycityfromcoordinates. Response content: {response.text if response else 'N/A'}")
        return jsonify({"error": "Error decoding JSON response."}), 500


@app.route('/countycityfromcoordinates_batch', methods=['POST'])
def get_county_city_from_coordinates_batch():
    counter = 0

    partititon_amount = 10

    if not api_key:
        logging.error("API key missing in /countycityfromcoordinates_batch request.")
        return jsonify({"error": "GEMINI_SCHOOL_API_KEY environment variable not set."}), 500

    try:
        batch_data = request.get_json()
        if not isinstance(batch_data, list):
            return jsonify({"error": "Request body must be a JSON array of coordinate objects."}), 400
    except Exception as e:
        logging.error(f"Error parsing batch request JSON: {e}")
        return jsonify({"error": f"Invalid JSON in request body: {e}"}), 400

    results = []
    # Using ThreadPoolExecutor to make concurrent API calls to Gemini
    # Adjust max_workers based on your server's capacity and Gemini's rate limits
    # A value of 5-10 is often a good starting point for external APIs
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:

    # with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_coords = {
            executor.submit(_process_single_coordinate, coords, api_key): coords
            for coords in batch_data
        }
        for future in concurrent.futures.as_completed(future_to_coords):
            coords = future_to_coords[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as exc:
                logging.error(f"Coordinate {coords} generated an exception: {exc}")
                results.append({
                    "latitude": coords.get('latitude'),
                    "longitude": coords.get('longitude'),
                    "error": str(exc),
                    "gbifID_original_index": coords.get('gbifID_original_index') # Ensure this is passed back even on error
                })
            if counter % 2 == partititon_amount:
                time.delay()
            counter += 1

    return jsonify(results)

def _process_single_coordinate(coords, api_key):
    """Helper function to process a single coordinate set for the batch endpoint.
    It calls the Gemini API for one coordinate and returns its processed result."""
    decimal_latitude = coords.get('latitude')
    decimal_longitude = coords.get('longitude')
    coordinate_uncertainty = coords.get('coordinate_uncertainty')

    if decimal_latitude is None or decimal_longitude is None:
        return {
            "latitude": decimal_latitude,
            "longitude": decimal_longitude,
            "error": "Missing 'latitude' or longitude parameter in coordinate object.",
            "gbifID_original_index": coords.get('gbifID_original_index')
        }

    prompt_text = f"""what county and city is the following in?
    decimal latitude: {decimal_latitude}
    decimal longitude: {decimal_longitude}
    coordinate uncertainty: {coordinate_uncertainty if coordinate_uncertainty else 'not specified'}"""

    # CHQ: Gemini AI added generationConfig to specify structured output
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
    headers = {'Content-Type': 'application/json'}
    data = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": prompt_text}
                ]
            }
        ],
        "generationConfig": { # NEW: Specify structured output
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "OBJECT",
                "properties": {
                    "county": {"type": "STRING"},
                    "city/town": {"type": "STRING"}
                },
                "propertyOrdering": ["county", "city/town"]
            }
        }
    }

    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        response.raise_for_status()
        response_json = response.json()

        # Extract the structured response directly
        if response_json.get('candidates') and response_json['candidates'][0].get('content') and response_json['candidates'][0]['content'].get('parts'):
            # The content will already be parsed JSON if responseMimeType is application/json
            parsed_gemini_response = json.loads(response_json['candidates'][0]['content']['parts'][0]['text'])
            return {
                "latitude": decimal_latitude,
                "longitude": decimal_longitude,
                "county": parsed_gemini_response.get('county'),
                "city/town": parsed_gemini_response.get('city/town'),
                "gbifID_original_index": coords.get('gbifID_original_index')
            }
        else:
            logging.warning(f"Gemini response did not contain expected structured content for ({decimal_latitude}, {decimal_longitude}). Full response: {response_json}")
            return {
                "latitude": decimal_latitude,
                "longitude": decimal_longitude,
                "error": "Gemini response did not contain expected structured content.",
                "gbifID_original_index": coords.get('gbifID_original_index')
            }
    except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
        error_detail = str(e)
        if hasattr(e, 'response') and e.response is not None:
            error_detail = f"HTTP Status {e.response.status_code}: {e.response.text}"
        logging.error(f"Error processing coordinate ({decimal_latitude}, {decimal_longitude}): {error_detail}")
        return {
            "latitude": decimal_latitude,
            "longitude": decimal_longitude,
            "error": error_detail,
            "gbifID_original_index": coords.get('gbifID_original_index')
        }

# @app.route('/countycityfromcoordinates_nativebatch', methods=['POST'])
# def get_county_city_from_coordinates_native_batching():
#     return 1

@app.route('/aichat')
def aichat_endpoint():
    if not api_key:
        return jsonify({"error": "GEMINI_SCHOOL_API_KEY environment variable not set."})

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
        response.raise_for_status()
        response_json = response.json()
        return jsonify(response_json)
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Error during API call: {e}"})
    except json.JSONDecodeError:
        return jsonify({"error": "Error decoding JSON response."})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=os.environ.get('PORT', 5000), debug=True)
