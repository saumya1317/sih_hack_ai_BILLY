import io
import os
import json
import google.generativeai as genai
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
from dotenv import load_dotenv

# Load environment variables from a .env file at the top
load_dotenv()

# --- 1. Gemini and App Configuration ---
# ----------------------------------------
try:
    GOOGLE_API_KEY = os.environ['GOOGLE_API_KEY']
    genai.configure(api_key=GOOGLE_API_KEY)
    gemini_model = genai.GenerativeModel(model_name="gemini-1.5-flash")
    print("Gemini model configured and ready.")
    GEMINI_ENABLED = True
except KeyError:
    print("FATAL ERROR: GOOGLE_API_KEY environment variable not set.")
    GEMINI_ENABLED = False
    gemini_model = None

# Initialize the FastAPI App
app = FastAPI(
    title="Nagar Seva AI Assistant",
    description="Uses the Gemini API to classify civic issues in Bhagalpur and route them to the correct department.",
    version="2.0.0 (Hackathon Edition)"
)

# Add CORS Middleware to allow your frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 2. Application Logic and Mappings ---
# -----------------------------------------
# Expanded list of all possible issue categories
CLASS_NAMES = [
    'garbage', 'pothole', 'uneven_roads', 
    'water_logging', 'leaking_pipe', 'clogged_drain', 'stray_animals',
    'streetlight_working', 'streetlight_not_working', 'other'
]

# Mapping of issue categories to municipal departments
DEPARTMENT_MAPPING = {
    'garbage': 'Waste Management Department',
    'pothole': 'Public Works Department (Roads)',
    'uneven_roads': 'Public Works Department (Roads)',
    'water_logging': 'Water Supply & Drainage Department',
    'leaking_pipe': 'Water Supply & Drainage Department',
    'clogged_drain': 'Sanitation Department',
    'stray_animals': 'Animal Control Department',
    'streetlight_not_working': 'Electrical Department',
    'streetlight_working': 'Electrical Department',
    'other': 'General Administration'
}

# --- 3. Gemini Analysis Function ---
# -----------------------------------
def analyze_with_gemini(image: Image.Image, text_context: str) -> dict:
    if not GEMINI_ENABLED:
        raise HTTPException(status_code=503, detail="Gemini API is not configured on the server.")
    try:
        # A more detailed prompt to get a richer JSON response
        prompt = (
            f"You are an AI assistant for the Bhagalpur municipal corporation in Bihar, India. "
            f"Analyze the following image and user's description: '{text_context}'. "
            f"1. Classify the main issue into one of these categories: {', '.join(CLASS_NAMES)}. "
            f"2. Determine a severity level for the issue: 'Low', 'Medium', or 'High'. "
            f"3. Suggest a brief, one-sentence action for the responsible department. "
            "Respond with only a single, raw JSON object containing 'prediction', 'severity', and 'action' keys. "
            "For example: {\"prediction\": \"pothole\", \"severity\": \"High\", \"action\": \"Requires immediate road patching and inspection.\"}"
        )
        response = gemini_model.generate_content([prompt, image])
        
        # Clean up the response to ensure it's valid JSON
        clean_response = response.text.strip().replace("```json", "").replace("```", "")
        result = json.loads(clean_response)
        return result

    except Exception as e:
        print(f"An error occurred calling Gemini API: {e}")
        raise HTTPException(status_code=500, detail="Error communicating with the AI model.")

# --- 4. API Endpoint ---
# -----------------------
class ClassificationResponse(BaseModel):
    prediction: str
    department: str
    severity: str
    action: str

@app.post("/classify", response_model=ClassificationResponse)
async def classify_endpoint(
    file: UploadFile = File(...),
    text: str = Form(...)
):
    image = Image.open(io.BytesIO(await file.read())).convert('RGB')
    
    # Call the Gemini function to get the detailed analysis
    result = analyze_with_gemini(image, text)
    
    # Extract the prediction and look up the corresponding department
    prediction = result.get("prediction", "other")
    department = DEPARTMENT_MAPPING.get(prediction, 'General Administration')
    
    # Get the other details from the Gemini response
    severity = result.get("severity", "Medium")
    action = result.get("action", "Further inspection required.")
    
    return ClassificationResponse(
        prediction=prediction,
        department=department,
        severity=severity,
        action=action
    )

@app.get("/")
def root():
    return {"message": "Welcome to the Nagar Seva Hackathon API!"}