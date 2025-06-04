from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from typing import Dict, List, Optional
from pydantic import BaseModel
import os
import json
import re
from langgraph.graph import StateGraph, END
from langchain.prompts import PromptTemplate
from PIL import Image
import base64
from io import BytesIO
import shutil
from typing import TypedDict
from openai import OpenAI, OpenAIError
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize OpenAI client
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# OpenAI integration for image and text processing
class GPT4oProcessor:
    def analyze_image(self, image_path: str) -> Dict:
        """Analyze image using OpenAI GPT-4o vision API with robust JSON extraction."""
        try:
            # Validate and convert image
            with Image.open(image_path) as img:
                if img.format not in ["JPEG", "PNG"]:
                    logger.error(f"Unsupported image format: {img.format}")
                    return {"error": "Unsupported image format. Use JPEG or PNG."}
                # Convert to JPEG
                buffered = BytesIO()
                img.convert("RGB").save(buffered, format="JPEG", quality=85)
                img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

            logger.info("Sending image analysis request to GPT-4o")
            response = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Analyze this image and identify the product type and possible attributes (e.g., color, brand, size, category). Return the result in JSON format with keys 'product_type' and 'possible_attributes', wrapped in ```json ``` code fences."
                            },
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}}
                        ]
                    }
                ],
                max_tokens=300
            )
            raw_content = response.choices[0].message.content
            logger.info(f"Raw GPT-4o response (analyze_image): {raw_content}")

            # Extract JSON using regex
            json_pattern = r'```json\s*([\s\S]*?)\s*```|\{[\s\S]*?\}'
            json_match = re.search(json_pattern, raw_content, re.MULTILINE)
            if not json_match:
                logger.error("No valid JSON found in response")
                return {"error": "No valid JSON found in GPT-4o response"}

            json_str = json_match.group(1) if json_match.group(1) else json_match.group(0)
            json_str = re.sub(r'\s*\n\s*', '', json_str.strip())
            logger.info(f"Extracted JSON string: {json_str}")

            # Parse JSON
            try:
                result = json.loads(json_str)
                if not isinstance(result, dict):
                    logger.error("Parsed result is not a dictionary")
                    return {"error": "Invalid JSON structure: result is not a dictionary"}
                if "product_type" not in result or "possible_attributes" not in result:
                    logger.error("Invalid JSON structure: missing required fields")
                    return {"error": "Invalid JSON structure: missing 'product_type' or 'possible_attributes'"}
                if not isinstance(result["possible_attributes"], dict):
                    logger.error("Invalid JSON structure: 'possible_attributes' is not a dictionary")
                    return {"error": "Invalid JSON structure: 'possible_attributes' must be a dictionary"}
                logger.info(f"Extracted and validated JSON: {result}")
                return result
            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing error: {str(e)}")
                return {"error": f"Failed to parse JSON from response: {str(e)}"}

        except OpenAIError as e:
            logger.error(f"OpenAI API error: {str(e)}")
            return {"error": f"Failed to analyze image: {str(e)}"}
        except Exception as e:
            logger.error(f"Image analysis error: {str(e)}")
            return {"error": f"Failed to analyze image: {str(e)}"}

    def generate_response(self, prompt: str, context: Dict) -> str:
        """Generate text response using OpenAI GPT-4o, ensuring plain text output."""
        try:
            logger.info(f"Sending generate_response request to GPT-4o with prompt: {prompt}")
            response = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful product assistant. Always respond with plain text. Do not include JSON, code fences (```), or any structured data format."
                    },
                    {"role": "user", "content": f"Context: {json.dumps(context)}\nPrompt: {prompt}"}
                ],
                max_tokens=150
            )
            raw_response = response.choices[0].message.content.strip()
            logger.info(f"Raw GPT-4o response (generate_response): {raw_response}")

            cleaned_response = re.sub(r'```[\s\S]*?```|\{[\s\S]*?\}|\[[\s\S]*?\]', '', raw_response).strip()
            cleaned_response = re.sub(r'\s+', ' ', cleaned_response)
            if not cleaned_response:
                logger.error("Cleaned response is empty")
                return "Unable to generate a recommendation due to an empty response. Please try again."
            logger.info(f"Cleaned response (generate_response): {cleaned_response}")
            return cleaned_response
        except Exception as e:
            logger.error(f"Text generation error: {str(e)}")
            return f"Error generating response: {str(e)}"

# State definition for LangGraph
class AssistantState(TypedDict):
    image_path: Optional[str]
    product_info: Dict
    user_preferences: Dict
    conversation_history: List[Dict]
    current_question: Optional[str]
    recommendations: List[Dict]
    error_message: Optional[str]

# LangGraph-based Product Assistant
class ProductAssistant:
    def __init__(self):
        self.llm = GPT4oProcessor()
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(AssistantState)
        workflow.add_node("analyze_image", self._analyze_image)
        workflow.add_node("ask_clarification", self._ask_clarification)
        workflow.add_node("process_user_response", self._process_user_response)
        workflow.add_node("generate_recommendations", self._generate_recommendations)
        workflow.add_node("handle_error", self._handle_error)

        workflow.set_entry_point("analyze_image")
        workflow.add_conditional_edges(
            "analyze_image",
            self._route_after_image_analysis,
            {"ask_clarification": "ask_clarification", "error": "handle_error"}
        )
        workflow.add_edge("ask_clarification", "process_user_response")
        workflow.add_conditional_edges(
            "process_user_response",
            self._route_after_user_response,
            {
                "ask_clarification": "ask_clarification",
                "generate_recommendations": "generate_recommendations",
                "error": "handle_error"
            }
        )
        workflow.add_edge("generate_recommendations", END)
        workflow.add_edge("handle_error", END)
        return workflow.compile()

    def _analyze_image(self, state: AssistantState) -> AssistantState:
        logger.info("Executing analyze_image node")
        if not state.get("image_path"):
            return {"error_message": "No image provided."}
        product_info = self.llm.analyze_image(state["image_path"])
        if "error" in product_info:
            return {"error_message": product_info["error"]}
        logger.info(f"analyze_image completed with product_info: {product_info}")
        return {
            "product_info": product_info,
            "user_preferences": {},
            "conversation_history": [{"role": "system", "content": "Image analyzed."}]  # Reset history
        }

    def _ask_clarification(self, state: AssistantState) -> AssistantState:
        logger.info("Executing ask_clarification node")
        product_info = state.get("product_info", {})
        preferences = state.get("user_preferences", {})
        questions = {
            "color": "What color would you prefer for this product?",
            "brand": "Do you have a preferred brand?",
            "size": "What size are you looking for?",
            "price_range": "What is your preferred price range?"
        }
        for attr, question in questions.items():
            if attr not in preferences and attr in product_info.get("possible_attributes", {}):
                logger.info(f"Asking question: {question}")
                return {
                    "current_question": question,
                    "conversation_history": state.get("conversation_history", []) + [{"role": "assistant", "content": question}]
                }
        logger.info("No clarification questions needed")
        return {"current_question": None}

    def _process_user_response(self, state: AssistantState) -> AssistantState:
        logger.info("Executing process_user_response node")
        user_response = state.get("conversation_history", [])[-1].get("content", "")
        current_question = state.get("current_question", "")
        preference_key = None
        if "color" in current_question.lower():
            preference_key = "color"
        elif "brand" in current_question.lower():
            preference_key = "brand"
        elif "size" in current_question.lower():
            preference_key = "size"
        elif "price" in current_question.lower():
            preference_key = "price_range"
        if preference_key:
            preferences = state.get("user_preferences", {})
            preferences[preference_key] = user_response
            logger.info(f"Updated preferences: {preferences}")
            return {
                "user_preferences": preferences,
                "conversation_history": state.get("conversation_history", []) + [{"role": "user", "content": user_response}]
            }
        logger.error("Could not understand user response")
        return {"error_message": "Could not understand your response. Please clarify."}

    def _generate_recommendations(self, state: AssistantState) -> AssistantState:
        logger.info("Executing generate_recommendations node")
        product_info = state.get("product_info", {})
        preferences = state.get("user_preferences", {})
        recommendations = [
            {
                "product_name": f"{preferences.get('brand', 'Generic')} {product_info.get('product_type', 'Product')}",
                "color": preferences.get("color", "N/A"),
                "size": preferences.get("size", "N/A"),
                "price": preferences.get("price_range", "N/A")
            }
        ]
        logger.info(f"Generating recommendation for: {recommendations[0]}")
        recommendation_text = self.llm.generate_response(
            prompt=f"Generate a plain text recommendation message for the product: {json.dumps(recommendations[0])}. Do not include JSON, code fences (```), or any structured data format. Example: 'I recommend a Nike T-shirt in Blue, size Medium.'",
            context=state
        )
        logger.info(f"Recommendation text: {recommendation_text}")
        return {
            "recommendations": recommendations,
            "conversation_history": state.get("conversation_history", []) + [{"role": "assistant", "content": recommendation_text}]
        }

    def _handle_error(self, state: AssistantState) -> AssistantState:
        logger.info("Executing handle_error node")
        error_message = state.get("error_message", "An unexpected error occurred.")
        return {
            "conversation_history": state.get("conversation_history", []) + [{"role": "assistant", "content": error_message}]
        }

    def _route_after_image_analysis(self, state: AssistantState) -> str:
        logger.info("Routing after image analysis")
        if state.get("error_message"):
            logger.info("Routing to handle_error")
            return "error"
        logger.info("Routing to ask_clarification")
        return "ask_clarification"

    def _route_after_user_response(self, state: AssistantState) -> str:
        logger.info("Routing after user response")
        if state.get("error_message"):
            logger.info("Routing to handle_error")
            return "error"
        if len(state.get("user_preferences", {})) >= 3:
            logger.info("Routing to generate_recommendations")
            return "generate_recommendations"
        logger.info("Routing to ask_clarification")
        return "ask_clarification"

    def process_input(self, image_path: Optional[str] = None, user_message: Optional[str] = None, session_id: Optional[str] = None) -> Dict:
        logger.info(f"Processing input with session_id: {session_id}")
        state_file = f"session_{session_id}.json" if session_id else None
        state = {"conversation_history": []}  # Always start with fresh state

        if image_path:
            state["image_path"] = image_path
        if user_message:
            state["conversation_history"].append({"role": "user", "content": user_message})

        logger.info("Invoking LangGraph workflow")
        result = self.graph.invoke(state)
        logger.info(f"Workflow result: {result}")

        if state_file:
            try:
                with open(state_file, "w") as f:
                    json.dump(result, f)
                logger.info(f"Saved state to {state_file}")
            except Exception as e:
                logger.error(f"Failed to save state file: {str(e)}")

        return result

# FastAPI setup
app = FastAPI(title="Product Assistant API")
assistant = ProductAssistant()

# Pydantic model for user message
class UserMessage(BaseModel):
    message: Optional[str] = None
    session_id: Optional[str] = None

# API Endpoints
@app.post("/upload-image")
async def upload_image(file: UploadFile = File(...), session_id: Optional[str] = None):
    """Endpoint to upload an image and start product identification."""
    logger.info(f"Uploading image: {file.filename} with session_id: {session_id}")
    if not file.content_type.startswith("image/"):
        logger.error("Invalid file type uploaded")
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)
    image_path = os.path.join(upload_dir, file.filename)
    with open(image_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = assistant.process_input(image_path=image_path, session_id=session_id)
    logger.info(f"Upload image result: {result}")

    try:
        response = {
            "message": result["conversation_history"][-1]["content"],
            "session_id": session_id,
            "recommendations": result.get("recommendations", [])
        }
        logger.info(f"Response to client: {response}")
        return JSONResponse(content=response)
    except KeyError as e:
        logger.error(f"Error constructing response: {str(e)}")
        raise HTTPException(status_code=500, detail="Error processing response")

@app.post("/send-message")
async def send_message(user_input: UserMessage):
    """Endpoint to send user messages for multi-turn conversation."""
    logger.info(f"Sending message with session_id: {user_input.session_id}")
    if not user_input.session_id:
        logger.error("Missing session_id")
        raise HTTPException(status_code=400, detail="Session ID is required for conversation.")

    result = assistant.process_input(user_message=user_input.message, session_id=user_input.session_id)
    logger.info(f"Send message result: {result}")

    try:
        response = {
            "message": result["conversation_history"][-1]["content"],
            "session_id": user_input.session_id,
            "recommendations": result.get("recommendations", [])
        }
        logger.info(f"Response to client: {response}")
        return JSONResponse(content=response)
    except KeyError as e:
        logger.error(f"Error constructing response: {str(e)}")
        raise HTTPException(status_code=500, detail="Error processing response")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    logger.info("Health check requested")
    return {"status": "healthy"}