Product Assistant API
Overview
This is a FastAPI-based API for an AI-powered product assistant that identifies and recommends products based on image uploads and user preferences. Built with Python, LangGraph, and OpenAI GPT-4o, it supports image analysis, multi-turn conversations, and robust error handling.
Features

Image Upload: Upload product images via /upload-image.
Conversational Clarification: Asks follow-up questions (e.g., brand, color, size, price range) to refine preferences.
Product Identification: Recommends products based on image analysis and user responses.
Error Handling: Handles invalid images, malformed API responses, or unclear user inputs gracefully.
Extensibility: Modular design supports adding new product categories and attributes.
Session Persistence: Maintains conversation state using session IDs and JSON files.

Requirements

Python 3.8+
OpenAI API key (set as OPENAI_API_KEY environment variable)
Dependencies listed in requirements.txt

Installation

unzip the provided Recommendation Agent.zip file

Create a virtual environment:python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install dependencies:pip install -r requirements.txt


Set OpenAI API key:export OPENAI_API_KEY='your-api-key'  # On Windows: set OPENAI_API_KEY=your-api-key



Usage

Start the FastAPI server:
uvicorn main:app --reload


The API will be available at http://localhost:8000. Access the interactive API docs at http://localhost:8000/docs.

Endpoints:

POST /upload-image: Upload a product image to start identification.
Parameters: file (image file, JPEG/PNG), session_id (optional, to track conversation).
Example (using curl):curl -X POST "http://localhost:8000/upload-image" -F "file=@sample_product.jpg" -F "session_id=12345"


Response:{
  "message": "What color would you prefer for this product?",
  "session_id": "12345",
  "recommendations": []
}




POST /send-message: Send user messages for multi-turn conversations.
Body: JSON with message (user input) and session_id (required).
Example:curl -X POST "http://localhost:8000/send-message" -H "Content-Type: application/json" -d '{"message": "Blue", "session_id": "12345"}'


Response:{
  "message": "Do you have a preferred brand?",
  "session_id": "12345",
  "recommendations": []
}




GET /health: Check API health.
Response: {"status": "healthy"}




Testing Example:

Upload an image (e.g., sample_product.jpg) via /upload-image with a session_id.
Respond to questions (e.g., "Blue", "Nike", "Medium") via /send-message using the same session_id.
Receive a recommendation after sufficient preferences (e.g., color, brand, size).



Project Structure

main.py: FastAPI application with LangGraph workflow and API endpoints.
requirements.txt: Python dependencies.
README.md: This file.
uploads/: Directory for uploaded images (created automatically).
session_*.json: Session state files for conversation persistence.

Notes

OpenAI Integration: Uses GPT-4o for image analysis and text responses. Ensure a valid OPENAI_API_KEY is set.
JSON Extraction: Uses regex to robustly extract and validate JSON from GPT-4o responses for image analysis, and ensures plain text for recommendation messages.
Image Formats: Supports JPEG and PNG.
Session Persistence: Stores state in session_<session_id>.json files. Consider a database (e.g., Redis, SQLite) for production.
Extensibility: Add new attributes to the questions dictionary in ProductAssistant to support more product categories.
Error Handling: Validates image types, session IDs, and API responses, with graceful handling of errors.

Testing

Start the server:uvicorn main:app --reload


Use the API docs (/docs) or curl commands to test:
Upload a valid JPEG/PNG image and verify the assistant asks a relevant question.
Send responses (e.g., "Blue", "Nike", "Medium") and confirm recommendations.
Test error handling with an invalid image (e.g., a text file) or missing session_id.


Example Flow:# Upload image
curl -X POST "http://localhost:8000/upload-image" -F "file=@shoe.jpg" -F "session_id=12345"
# Respond with color
curl -X POST "http://localhost:8000/send-message" -H "Content-Type: application/json" -d '{"message": "Blue", "session_id": "12345"}'
# Respond with brand
curl -X POST "http://localhost:8000/send-message" -H "Content-Type: application/json" -d '{"message": "Nike", "session_id": "12345"}'
# Respond with size
curl -X POST "http://localhost:8000/send-message" -H "Content-Type: application/json" -d '{"message": "Medium", "session_id": "12345"}'



Future Improvements

Replace JSON-based session storage with a database (e.g., Redis, SQLite).
Integrate a real product database or e-commerce API for accurate recommendations.
Add authentication for session management.
Implement rate limiting and file size validation for image uploads.
Enhance error handling for edge cases in GPT-4o responses.
