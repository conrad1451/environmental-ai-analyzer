# analyzer.py

from flask import Flask, jsonify, request
from flask_cors import CORS
import requests
import json
import os
import logging
import concurrent.futures
import uuid # For unique filenames

# --- UPDATED IMPORTS FOR NEW GOOGLE GEN AI SDK ---
from google.cloud import storage
import google.generativeai as genai # Renamed import for clarity
from google.api_core.client_options import ClientOptions
from google.api_core.exceptions import GoogleAPIError

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__) # Use the logger instance

# Authentication for direct API key access (for non-batch endpoints)
api_key = os.environ.get("GEMINI_SCHOOL_API_KEY") # Current API key from user

if not api_key:
    logging.error("Error: GEMINI_SCHOOL_API_KEY environment variable not set at app startup.")


# --- Configuration for Native Batching ---
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
# --- UPDATED INITIALIZATION FOR NEW GOOGLE GEN AI SDK ---
try:
    # The 'google-generativeai' SDK usually handles API key or GOOGLE_APPLICATION_CREDENTIALS automatically.
    # For direct API key usage, you'd configure it like: genai.configure(api_key=api_key)
    # Since you're using GOOGLE_APPLICATION_CREDENTIALS for GCS, the SDK should pick it up.
    # If using a specific endpoint, you might need: genai.configure(client_options={'api_endpoint': 'generativelanguage.googleapis.com'})
    # For batching, we will use the client object directly if needed, or rely on SDK's internal client.
    
    # For batching, the SDK provides `generative_models.batch_generate_contents` as a top-level function
    # or through the client. Let's ensure the client is ready for direct use if needed.
    # The `genai` client itself doesn't need explicit initialization like `glm.GenerativeServiceClient`
    # if you're using `genai.GenerativeModel`.
    
    # For batching, we might need a specific client, or it might be handled internally.
    # Let's keep the `genai_client` variable for now, but its usage will change.
    # The `google-generativeai` library typically makes direct calls on the model object.
    
    # For now, let's ensure the API key is configured for direct calls if needed,
    # or ensure GOOGLE_APPLICATION_CREDENTIALS is set for batching.
    
    # If you intend to use the direct API key for all calls (including batching if supported by SDK),
    # then configure it like this:
    # genai.configure(api_key=api_key)
    # logger.info("Generative AI SDK configured with API key.")

    # If you intend to use GOOGLE_APPLICATION_CREDENTIALS for batching:
    # The `google-generativeai` library will pick up GOOGLE_APPLICATION_CREDENTIALS automatically.
    # We will instantiate the model directly where needed.
    logger.info("Generative AI SDK initialized (will use GOOGLE_APPLICATION_CREDENTIALS or API_KEY as per configuration).")
    genai_client = None # We won't use this directly as before, but keep for placeholder if needed.

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

    # --- UPDATED: Use new SDK for direct calls ---
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content("Explain how AI works")
        
        # The response object from google-generativeai is different
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

    # --- UPDATED: Use new SDK for direct calls with structured output ---
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        # Define the response schema using the SDK's types
        response_schema = genai.types.Schema(
            type=genai.types.Type.OBJECT,
            properties={
                "county": genai.types.Schema(type=genai.types.Type.STRING),
                "city/town": genai.types.Schema(type=genai.types.Type.STRING)
            },
            # property_ordering is not directly supported in genai.types.Schema,
            # but the order of properties in the dict usually maintains it.
        )

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
            return jsonify(parsed_gemini_response)
        else:
            logging.warning(f"Gemini response did not contain expected structured content: {response}")
            return jsonify({"error": "Gemini response did not contain expected structured content."}), 500
    except Exception as e:
        the_error_msg = f"""Error during API call to Gemini in /countycityfromcoordinates: {e}"""
        logging.error(the_error_msg, exc_info=True) # Log full traceback
        return jsonify({"error": f"Error during API call: {e}"}), 500


@app.route('/countycityfromcoordinates_batch', methods=['POST'])
def get_county_city_from_coordinates_batch():
    # This endpoint uses client-side batching with concurrent.futures.ThreadPoolExecutor
    # It will continue to use the 'requests' library and the API key directly.
    # No changes needed here, as it's not using the native batching feature.
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
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
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
                    "gbifID_original_index": coords.get('gbifID_original_index')
                })
    return jsonify(results)

def _process_single_coordinate(coords, api_key):
    # This helper function for client-side batching also needs to be updated
    # to use the new SDK if you want it to be consistent.
    # For now, keeping it with 'requests' to minimize changes, but it's less efficient.
    # If you want to use the SDK here, it would mirror the logic in /countycityfromcoordinates.
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

    # --- UPDATED: Use new SDK for _process_single_coordinate ---
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        response_schema = genai.types.Schema(
            type=genai.types.Type.OBJECT,
            properties={
                "county": genai.types.Schema(type=genai.types.Type.STRING),
                "city/town": genai.types.Schema(type=genai.types.Type.STRING)
            }
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
            return {
                "latitude": decimal_latitude,
                "longitude": decimal_longitude,
                "county": parsed_gemini_response.get('county'),
                "city/town": parsed_gemini_response.get('city/town'),
                "gbifID_original_index": coords.get('gbifID_original_index')
            }
        else:
            logging.warning(f"Gemini response did not contain expected structured content for ({decimal_latitude}, {decimal_longitude}). Full response: {response}")
            return {
                "latitude": decimal_latitude,
                "longitude": decimal_longitude,
                "error": "Gemini response did not contain expected structured content.",
                "gbifID_original_index": coords.get('gbifID_original_index')
            }
    except Exception as e:
        error_detail = str(e)
        logging.error(f"Error processing coordinate ({decimal_latitude}, {decimal_longitude}): {error_detail}", exc_info=True)
        return {
            "latitude": decimal_latitude,
            "longitude": decimal_longitude,
            "error": error_detail,
            "gbifID_original_index": coords.get('gbifID_original_index')
        }


@app.route('/countycityfromcoordinates_nativebatch', methods=['POST'])
def get_county_city_from_coordinates_native_batching():
    # This endpoint uses the native batching feature of the Generative Language API.
    # The `google-generativeai` SDK provides a higher-level function for this.
    if not storage_client: # genai_client is no longer used directly as before
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

    # 1. Prepare input data as JSONL
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

    # 2. Upload input file to GCS (still needed for native batching)
    unique_id = str(uuid.uuid4())
    input_blob_name = f"batch_input/{unique_id}/input.jsonl"
    output_prefix = f"batch_output/{unique_id}/"

    try:
        input_bucket = storage_client.bucket(GCS_INPUT_BUCKET)
        input_blob = input_bucket.blob(input_blob_name)
        
        # Convert input_entries to JSONL string for upload
        jsonl_content = "\n".join([json.dumps(entry.to_dict()) for entry in input_entries])
        input_blob.upload_from_string(jsonl_content, content_type="application/jsonl")
        
        input_uri = f"gs://{GCS_INPUT_BUCKET}/{input_blob_name}"
        output_uri = f"gs://{GCS_OUTPUT_BUCKET}/{output_prefix}"
        logger.info(f"Uploaded input to GCS: {input_uri}")
    except Exception as e:
        logger.error(f"Error uploading input to GCS: {e}", exc_info=True)
        return jsonify({"error": f"Failed to upload input to GCS: {e}"}), 500

    # 3. Initiate the Batch Job with Generative Language API (using the new SDK)
    try:
        # Define generation config for the batch job
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
        # This is a high-level function that handles the underlying API client.
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

    # --- UPDATED: Use new SDK for direct calls ---
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