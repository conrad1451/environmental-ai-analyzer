# analyzer.py

from flask import Flask, jsonify, request
from flask_cors import CORS
import requests
import json
import os
import logging
import concurrent.futures
import uuid # For unique filenames

# --- UPDATED IMPORTS ---
from google.cloud import storage
import google.generativeai as genai
from google.api_core.client_options import ClientOptions # Keep this if you need custom endpoint for genai.configure
from google.api_core.exceptions import GoogleAPIError

# --- NEW: Import tenacity for retry logic ---
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type, before_sleep_log

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

api_key = os.environ.get("GEMINI_SCHOOL_API_KEY")

if not api_key:
    logging.error("Error: GEMINI_SCHOOL_API_KEY environment variable not set at app startup.")

# Initialize GCS client
try:
    storage_client = storage.Client(project=os.environ.get("GCP_PROJECT_ID"))
    logger.info("Google Cloud Storage client initialized successfully.")
except Exception as e:
    logger.error(f"Error initializing Google Cloud Storage client: {e}. Ensure GOOGLE_APPLICATION_CREDENTIALS is set.")
    storage_client = None

# Configure genai SDK (This should be done once at startup)
try:
    if api_key:
        genai.configure(api_key=api_key)
        logger.info("Generative AI SDK configured with API key.")
    else:
        # The SDK will attempt to use GOOGLE_APPLICATION_CREDENTIALS if api_key is not configured
        logger.warning("GEMINI_SCHOOL_API_KEY not set. SDK will attempt to use GOOGLE_APPLICATION_CREDENTIALS for authentication.")
    
    # You generally don't need a `genai_client` variable for the new SDK,
    # as methods like `genai.GenerativeModel` and `genai.batch_generate_contents`
    # are top-level or accessed directly from the model.
    # The `ClientOptions` might be used for `genai.configure` if targeting a specific regional endpoint.
    
except Exception as e:
    logger.error(f"Error configuring Generative AI SDK: {e}. Ensure API key or GOOGLE_APPLICATION_CREDENTIALS is set.")


# --- Configuration for Native Batching ---
GCS_INPUT_BUCKET = os.environ.get("GCS_INPUT_BUCKET")
GCS_OUTPUT_BUCKET = os.environ.get("GCS_OUTPUT_BUCKET")
GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID")


@app.route('/')
def hello_world():
    return 'Conrad deployed another AI Flask App!'

@app.route('/explain_ai')
def explain_ai():
    if not api_key:
        logging.error("API key missing in /explain_ai request.")
        return jsonify({"error": "GEMINI_SCHOOL_API_KEY environment variable not set."}), 500

    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content("Explain how AI works")
        
        if response.candidates:
            text_response = response.candidates[0].content.parts[0].text
            return jsonify({"text": text_response})
        else:
            logging.warning(f"Gemini response did not contain candidates: {response}")
            return jsonify({"error": "Gemini response did not contain expected content."}), 500
    except Exception as e:
        logging.error(f"Error during API call to Gemini in /explain_ai: {e}")
        return jsonify({"error": f"Error during API call: {e}"})

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

    prompt_text = f"""what county and city is the following in?
    decimal latitude: {decimal_latitude}
    decimal longitude: {decimal_longitude}
    coordinate uncertainty: {coordinate_uncertainty if coordinate_uncertainty else 'not specified'}"""

    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        response_schema = genai.types.Schema(
            type=genai.types.Type.OBJECT,
            properties={
                "county": genai.types.Schema(type=genai.types.Type.STRING),
                "city/town": genai.types.Schema(type=genai.types.Type.STRING)
            },
        )

        response = model.generate_content(
            prompt_text,
            generation_config=genai.types.GenerationConfig(
                response_mime_type="application/json",
                response_schema=response_schema
            )
        )
        
        if response.candidates:
            parsed_gemini_response = json.loads(response.candidates[0].content.parts[0].text)
            return jsonify(parsed_gemini_response)
        else:
            logging.warning(f"Gemini response did not contain expected structured content: {response}")
            return jsonify({"error": "Gemini response did not contain expected structured content."}), 500
    except Exception as e:
        the_error_msg = f"""Error during API call to Gemini in /countycityfromcoordinates: {e}"""
        logging.error(the_error_msg, exc_info=True)
        return jsonify({"error": f"Error during API call: {e}"}), 500


@app.route('/countycityfromcoordinates_batch', methods=['POST'])
def get_county_city_from_coordinates_batch():
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


    loop_count = 0
    MAX_LOOPS = 4

    # Use max_workers=5 as a reasonable default for concurrent CPU-bound tasks
    # The actual effective concurrency will be limited by Gemini's API.
    # The _process_single_coordinate now has retries.
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor: 
        future_to_coords = {
            executor.submit(_process_single_coordinate_with_sdk, coords): coords # Use the SDK version
            for coords in batch_data
        }
        for future in concurrent.futures.as_completed(future_to_coords):
            coords = future_to_coords[future]

            if loop_count < MAX_LOOPS:
                try:
                    result = future.result()
                    results.append(result)
                except Exception as exc:
                    logging.error(f"Coordinate {coords} generated an exception: {exc}")
                    results.append({
                        "latitude": coords.get('latitude'),
                        "longitude": coords.get('longitude'),
                        "error": str(exc),
                        "gbifID_original_index": coords.get('gbifID_original_index')
                    })
                loop_count = loop_count + 1
    return jsonify(results)
   
# CHQ: Gemini AI generated the retry decorator and function
# --- NEW: Helper function using the Gen AI SDK and Tenacity ---
@retry(
    wait=wait_exponential(multiplier=1, min=1, max=10), # Wait 1s, 2s, 4s, etc., up to 10s
    stop=stop_after_attempt(5), # Max 5 attempts
    # Retry on specific Google API errors (like RESOURCE_EXHAUSTED for rate limits)
    # and general request exceptions.
    retry=retry_if_exception_type((
        GoogleAPIError,
        requests.exceptions.RequestException 
    )),
    reraise=True,
    before_sleep=before_sleep_log(logger, logging.WARNING) # Log before retrying
)
def _process_single_coordinate_with_sdk(coords):
    """Helper function to process a single coordinate set using the new Gen AI SDK.
    Includes retry logic for API calls."""
    
    decimal_latitude = coords.get('latitude')
    decimal_longitude = coords.get('longitude')
    coordinate_uncertainty = coords.get('coordinate_uncertainty')

    if decimal_latitude is None or decimal_longitude is None:
        raise ValueError("Missing 'latitude' or longitude parameter in coordinate object.")

    prompt_text = f"""what county and city is the following in?
    decimal latitude: {decimal_latitude}
    decimal longitude: {decimal_longitude}
    coordinate uncertainty: {coordinate_uncertainty if coordinate_uncertainty else 'not specified'}"""

    model = genai.GenerativeModel('gemini-2.0-flash')
    response_schema = genai.types.Schema(
        type=genai.types.Type.OBJECT,
        properties={
            "county": genai.types.Schema(type=genai.types.Type.STRING),
            "city/town": genai.types.Schema(type=genai.types.Type.STRING)
        }
    )
    
    try:
        response = model.generate_content(
            prompt_text,
            generation_config=genai.types.GenerationConfig(
                response_mime_type="application/json",
                response_schema=response_schema
            )
        )
        
        if response.candidates:
            # The SDK's response will already be parsed if response_mime_type is application/json
            parsed_gemini_response = json.loads(response.candidates[0].content.parts[0].text)
            return {
                "latitude": decimal_latitude,
                "longitude": decimal_longitude,
                "county": parsed_gemini_response.get('county'),
                "city/town": parsed_gemini_response.get('city/town'),
                "gbifID_original_index": coords.get('gbifID_original_index')
            }
        else:
            logger.warning(f"Gemini response did not contain expected structured content for ({decimal_latitude}, {decimal_longitude}). Full response: {response}")
            raise GoogleAPIError("Gemini response missing expected content.")
    except GoogleAPIError as e:
        # Re-raise GoogleAPIError so tenacity can catch it
        raise e
    except Exception as e:
        logger.error(f"Unexpected error in _process_single_coordinate_with_sdk for ({decimal_latitude}, {decimal_longitude}): {e}", exc_info=True)
        # Wrap other exceptions as a generic GoogleAPIError for tenacity to catch (or not if it's not retryable)
        raise GoogleAPIError(f"Unexpected error: {e}")


# --- OLD helper function, no longer used by /countycityfromcoordinates_batch ---
# Keeping it for reference, but it should be removed or updated if not needed.
def _process_single_coordinate(coords, api_key):
    """OLD helper function using requests.post directly.
    NOT USED by the updated /countycityfromcoordinates_batch endpoint."""
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
        "generationConfig": {
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
        
        if response_json.get('candidates') and response_json['candidates'][0].get('content') and response_json['candidates'][0]['content'].get('parts'):
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


@app.route('/countycityfromcoordinates_nativebatch', methods=['POST'])
def get_county_city_from_coordinates_native_batching():
    if not storage_client:
        logger.error("Google Cloud Storage client not initialized. Cannot perform native batching.")
        return jsonify({"error": "Server not configured for native batching (GCS client uninitialized)."}), 500
    
    if not all([GCS_INPUT_BUCKET, GCS_OUTPUT_BUCKET, GCP_PROJECT_ID]):
        logger.error("Missing GCS_INPUT_BUCKET, GCS_OUTPUT_BUCKET, or GCP_PROJECT_ID environment variables.")
        return jsonify({"error": "Server not fully configured for native batching (missing GCS/Project env vars)."}), 500

    try:
        batch_data = request.get_json()
        if not isinstance(batch_data, list) or not batch_data:
            return jsonify({"error": "Request body must be a non-empty JSON array of coordinate objects."}), 400
    except Exception as e:
        logger.error(f"Error parsing native batch request JSON: {e}")
        return jsonify({"error": f"Invalid JSON in request body: {e}"}), 400

    input_entries = []
    for item in batch_data:
        decimal_latitude = item.get('latitude')
        decimal_longitude = item.get('longitude')
        coordinate_uncertainty = item.get('coordinate_uncertainty')
        gbif_id_original_index = item.get('gbifID_original_index')

        if decimal_latitude is None or decimal_longitude is None:
            logger.warning(f"Skipping malformed item in batch: {item}")
            continue

        prompt_text = f"""what county and city is the following in?
        decimal latitude: {decimal_latitude}
        decimal longitude: {decimal_longitude}
        coordinate uncertainty: {coordinate_uncertainty if coordinate_uncertainty else 'not specified'}"""
        
        input_entries.append(genai.types.BatchGenerateContentsRequest.Input(
            contents=[genai.types.Content(role="user", parts=[genai.types.Part(text=prompt_text)])],
            metadata={
                "original_latitude": str(decimal_latitude),
                "original_longitude": str(decimal_longitude),
                "gbifID_original_index": str(gbif_id_original_index)
            }
        ))

    if not input_entries:
        return jsonify({"error": "No valid coordinate objects found in the batch data."}), 400

    unique_id = str(uuid.uuid4())
    input_blob_name = f"batch_input/{unique_id}/input.jsonl"
    output_prefix = f"batch_output/{unique_id}/"

    try:
        input_bucket = storage_client.bucket(GCS_INPUT_BUCKET)
        input_blob = input_bucket.blob(input_blob_name)
        
        jsonl_content = "\n".join([json.dumps(entry.to_dict()) for entry in input_entries])
        input_blob.upload_from_string(jsonl_content, content_type="application/jsonl")
        
        input_uri = f"gs://{GCS_INPUT_BUCKET}/{input_blob_name}"
        output_uri = f"gs://{GCS_OUTPUT_BUCKET}/{output_prefix}"
        logger.info(f"Uploaded input to GCS: {input_uri}")
    except Exception as e:
        logger.error(f"Error uploading input to GCS: {e}", exc_info=True)
        return jsonify({"error": f"Failed to upload input to GCS: {e}"}), 500

    try:
        generation_config = genai.types.GenerationConfig(
            response_mime_type="application/json",
            response_schema=genai.types.Schema(
                type=genai.types.Type.OBJECT,
                properties={
                    "county": genai.types.Schema(type=genai.types.Type.STRING),
                    "city/town": genai.types.Schema(type=genai.types.Type.STRING)
                }
            )
        )
        
        # Call the batch_generate_contents function from the SDK
        operation = genai.batch_generate_contents(
            model="gemini-2.0-flash",
            input_content_uri=input_uri,
            output_content_uri=output_uri,
            generation_config=generation_config
        )
        
        logger.info(f"Initiated native batch job: {operation.name}")
        
        return jsonify({
            "status": "Batch job initiated",
            "operation_name": operation.name,
            "input_uri": input_uri,
            "output_uri_prefix": output_uri,
            "message": "The ETL script will need to poll this operation_name and retrieve results from GCS."
        }), 202

    except GoogleAPIError as e:
        logger.error(f"Google API error initiating batch job: {e}", exc_info=True)
        return jsonify({"error": f"Failed to initiate native batch job with Gemini API: {e.message}"}), 500
    except Exception as e:
        logger.error(f"Unexpected error initiating batch job: {e}", exc_info=True)
        return jsonify({"error": f"An unexpected error occurred: {e}"}), 500


@app.route('/aichat')
def aichat_endpoint():
    if not api_key:
        return jsonify({"error": "GEMINI_SCHOOL_API_KEY environment variable not set."})

    user_input = request.args.get('text')

    if not user_input:
        return jsonify({"error": "Missing 'text' parameter in the query string."})

    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(user_input)
        
        if response.candidates:
            text_response = response.candidates[0].content.parts[0].text
            return jsonify({"text": text_response})
        else:
            logging.warning(f"Gemini response did not contain candidates: {response}")
            return jsonify({"error": "Gemini response did not contain expected content."}), 500
    except Exception as e:
        logging.error(f"Error during API call to Gemini in /aichat: {e}")
        return jsonify({"error": f"Error during API call: {e}"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=os.environ.get('PORT', 5000), debug=True)