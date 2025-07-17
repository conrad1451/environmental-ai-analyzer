# analyzer.py

from flask import Flask, jsonify, request
from flask_cors import CORS
import requests
import json
import os
import logging
import concurrent.futures # <-- NEW IMPORT

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# It's better to store your API key as an environment variable
api_key = os.environ.get("GEMINI_PERSONAL_API_KEY")

if not api_key:
    logging.error("Error: GEMINI_PERSONAL_API_KEY environment variable not set at app startup.")

@app.route('/')
def hello_world():
    return 'Conrad deployed another AI Flask App!'

@app.route('/explain_ai')
def explain_ai():
    if not api_key:
        logging.error("API key missing in /explain_ai request.")
        return jsonify({"error": "GEMINI_PERSONAL_API_KEY environment variable not set."}), 500

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
        return jsonify({"error": "GEMINI_PERSONAL_API_KEY environment variable not set."}), 500

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
    prompt_text = f"""what county and city is the following in? Make the response like so:

    {{"county": "[County name]",
    "city/town": "[city/town name]"}}

    decimal latitude: {decimal_latitude}
    decimal longitude: {decimal_longitude}
    coordinate uncertainty: {coordinate_uncertainty if coordinate_uncertainty else 'not specified'}"""


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
    logging.info(f"Sending data to Gemini: {json.dumps(data)}")


    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        response.raise_for_status()
        response_json = response.json()
        logging.info("Gemini API call successful for /countycityfromcoordinates.")
        return jsonify(response_json)
    except requests.exceptions.RequestException as e:
        the_error_msg = f"""Error during API call to Gemini in /countycityfromcoordinates: {e}.
        Response status: {response.status_code if response else 'N/A'}, content: {response.text if response else 'N/A'}"""
        logging.error(the_error_msg)
        return jsonify({"error": f"Error during API call: {e}"}), 500
    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON response from Gemini in /countycityfromcoordinates. Response content: {response.text if response else 'N/A'}")
        return jsonify({"error": "Error decoding JSON response."}), 500


# --- NEW BATCH ENDPOINT ---
@app.route('/countycityfromcoordinates_batch', methods=['POST'])
def get_county_city_from_coordinates_batch():
    if not api_key:
        logging.error("API key missing in /countycityfromcoordinates_batch request.")
        return jsonify({"error": "GEMINI_PERSONAL_API_KEY environment variable not set."}), 500

    # Expect a JSON array of coordinate objects in the request body
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
    # If Gemini has a very strict per-second rate limit, you might need to lower this.
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_coords = {
            executor.submit(_process_single_coordinate, coords, api_key): coords
            for coords in batch_data
        }
        for future in concurrent.futures.as_completed(future_to_coords):
            coords = future_to_coords[future] # Retrieve the original coordinates to associate with the result
            try:
                result = future.result()
                results.append(result)
            except Exception as exc:
                logging.error(f"Coordinate {coords} generated an exception: {exc}")
                results.append({
                    "latitude": coords.get('latitude'),
                    "longitude": coords.get('longitude'),
                    "error": str(exc)
                })

    return jsonify(results)

# CHQ: Gemini AI debugged to remove extra characters inserted by AI prompt 
def _process_single_coordinate(coords, api_key):
    """Helper function to process a single coordinate set for the batch endpoint.
    It calls the Gemini API for one coordinate and returns its processed result."""
    decimal_latitude = coords.get('latitude')
    decimal_longitude = coords.get('longitude')
    coordinate_uncertainty = coords.get('coordinate_uncertainty')

    # Basic validation for the individual coordinate data
    if decimal_latitude is None or decimal_longitude is None: # Use 'is None' for clearer check
        return {
            "latitude": decimal_latitude, # Return what was received for debugging
            "longitude": decimal_longitude,
            "error": "Missing 'latitude' or longitude parameter in coordinate object."
        }

    # Construct the prompt for the Gemini API
    # Ensure the prompt asks for the response in the exact JSON format you expect.
    prompt_text = f"""what county and city is the following in? Make the response like so, ensure it's valid JSON:

    {{"county": "[County name]",
    "city/town": "[city/town name]"}}

    decimal latitude: {decimal_latitude}
    decimal longitude: {decimal_longitude}
    coordinate uncertainty: {coordinate_uncertainty if coordinate_uncertainty else 'not specified'}"""

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

    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        response_json = response.json()

        # Extract the county and city/town from Gemini's response
        # Gemini's response structure: response_json['candidates'][0]['content']['parts'][0]['text']
        gemini_text = response_json.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text')

        # CHQ: Gemini AI modified parsing logic
        if gemini_text:
            # --- START OF MODIFIED PARSING LOGIC ---
            # 1. Strip common problematic prefixes/suffixes like newlines, backticks, "json\n"
            cleaned_gemini_text = gemini_text.strip()
            # If Gemini sometimes wraps in single quotes, remove them
            if cleaned_gemini_text.startswith("'") and cleaned_gemini_text.endswith("'"):
                cleaned_gemini_text = cleaned_gemini_text[1:-1]
            
            # If Gemini includes "json\n" or similar, try to find the actual JSON start
            # This is robust because it finds the first '{' and last '}'
            json_start_index = cleaned_gemini_text.find('{')
            json_end_index = cleaned_gemini_text.rfind('}')

            if json_start_index != -1 and json_end_index != -1 and json_end_index > json_start_index:
                actual_json_string = cleaned_gemini_text[json_start_index : json_end_index + 1]
            else:
                # Fallback if no valid JSON structure is found
                logging.warning(f"Could not find valid JSON delimiters in Gemini response for ({decimal_latitude}, {decimal_longitude}): {cleaned_gemini_text}")
                return {
                    "latitude": decimal_latitude,
                    "longitude": decimal_longitude,
                    "error": f"Gemini response did not contain parsable JSON structure: {cleaned_gemini_text[:150]}..."
                }

            # --- END OF MODIFIED PARSING LOGIC ---

            try:
                parsed_gemini_response = json.loads(actual_json_string) # Use the extracted string
                return {
                    "latitude": decimal_latitude,
                    "longitude": decimal_longitude,
                    "county": parsed_gemini_response.get('county'),
                    "city/town": parsed_gemini_response.get('city/town'),
                    "gbifID_original_index": coords.get('gbifID_original_index') # IMPORTANT: Pass this back!
                }
            except json.JSONDecodeError as e:
                logging.error(f"Failed to parse CLEANED Gemini's JSON text for ({decimal_latitude}, {decimal_longitude}): {actual_json_string}. Error: {e}")
                return {
                    "latitude": decimal_latitude,
                    "longitude": decimal_longitude,
                    "error": f"Malformed JSON from Gemini (after cleaning): {actual_json_string[:100]}... Error: {e}",
                    "gbifID_original_index": coords.get('gbifID_original_index') # IMPORTANT: Pass this back!
                }
        else:
            logging.warning(f"Gemini response text was empty for ({decimal_latitude}, {decimal_longitude}). Full response: {response_json}")
            return {
                "latitude": decimal_latitude,
                "longitude": decimal_longitude,
                "error": "Gemini response text was empty or malformed.",
                "gbifID_original_index": coords.get('gbifID_original_index') # IMPORTANT: Pass this back!
            }
    except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
        # Catch network errors, timeouts, HTTP errors, and JSON decoding errors from requests.post
        error_detail = str(e)
        if hasattr(e, 'response') and e.response is not None:
            error_detail = f"HTTP Status {e.response.status_code}: {e.response.text}"
        logging.error(f"Error processing coordinate ({decimal_latitude}, {decimal_longitude}): {error_detail}")
        return {
            "latitude": decimal_latitude,
            "longitude": decimal_longitude,
            "error": error_detail,
            "gbifID_original_index": coords.get('gbifID_original_index') # IMPORTANT: Pass this back!
        }

@app.route('/aichat')
def aichat_endpoint():
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
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        response_json = response.json()
        return jsonify(response_json)
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Error during API call: {e}"})
    except json.JSONDecodeError:
        return jsonify({"error": "Error decoding JSON response."})

if __name__ == '__main__':
    # When deploying to Render/Heroku, they handle the host and port
    # For local testing, you can use app.run(debug=True)
    # For production, it's recommended to use a WSGI server like Gunicorn
    app.run(host='0.0.0.0', port=os.environ.get('PORT', 5000), debug=True)