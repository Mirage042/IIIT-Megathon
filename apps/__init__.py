from flask import Flask, request, jsonify
from flask_login import LoginManager
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
import os
from importlib import import_module
import google.generativeai as genai  # Import the Google Generative AI library
import random

# Initialize extensions
db = SQLAlchemy()
login_manager = LoginManager()

# Set up API key for Gemini
API_KEY = os.getenv("API_KEY", "AIzaSyABSVGA83cLKJ-RhSFFbzMJkzG3aYE_YWE")
genai.configure(api_key=API_KEY)  # Configure the Gemini API key

# Register extensions
def register_extensions(app):
    db.init_app(app)
    login_manager.init_app(app)

# Register blueprints
def register_blueprints(app):
    for module_name in ("authentication", "home"):
        module = import_module(f"apps.{module_name}.routes")
        app.register_blueprint(module.blueprint)

# Configure database
def configure_database(app):
    @app.before_first_request
    def initialize_database():
        db.create_all()

    @app.teardown_request
    def shutdown_session(exception=None):
        db.session.remove()

# Create the Flask app
def create_app(config):
    app = Flask(__name__)
    app.config.from_object(config)
    register_extensions(app)
    register_blueprints(app)
    configure_database(app)

    # Ensure the upload folder exists
    UPLOAD_FOLDER = os.path.join(os.getcwd(), "uploads")
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

    # Route for Gemini chatbot
    @app.route('/chat', methods=['POST'])
    def chat():
        try:
            data = request.json
            message = data.get('message')
            if not message:
                return jsonify({"error": "Message is required"}), 400
            
            total_shipments = random.randint(50, 150)  # Example: random total shipments between 50 and 150
            average_response_time = round(random.uniform(1.0, 10.0), 2)  # Random average response time in seconds
            customer_satisfaction_score = round(random.uniform(70.0, 100.0), 2)

            # Define context for the model
            context = (
                "You are a friendly and knowledgeable assistant that helps users understand their dashboard metrics in an engaging manner. "
                "You should provide clear and detailed explanations for the following metrics: "
                
                "- **Total Shipments:** Indicates the total number of shipments processed over a specific period. "
                "For instance, if the current value is 110, it means that 110 shipments have been successfully processed. "
                
                "- **Average Response Time:** Measures the average time taken to respond to customer inquiries, given in seconds. "
                "For example, if the current value is 3.2 seconds, it reflects prompt customer service response times. "
                
                "- **Customer Satisfaction Score:** A score reflecting customer satisfaction based on feedback, typically on a scale of 1 to 100. "
                "A current score of 88.5 indicates that customers are generally happy with the service provided. " 

                "If a user asks about a specific metric, provide an engaging explanation along with the current value. "
                "If they inquire about other metrics, respond with a list of available metrics and ask if they want to know more about a specific one. "
                "Always use natural and friendly language, and encourage further questions to enhance the interaction. "
                
                "User asks: "
            )

            full_input = context + " " + message

            # Use the Gemini API to generate a response
            model = genai.GenerativeModel("gemini-1.5-flash")  # Specify the model you want to use
            response = model.generate_content(full_input)

            return jsonify({"response": response.text})

        except Exception as e:
            app.logger.error(f"Error in /chat endpoint: {str(e)}")
            return jsonify({"error": "Internal server error"}), 500

    # Route for image upload
    @app.route("/upload_image", methods=["POST"])
    def upload_image():
        if "image" not in request.files:
            app.logger.error("No file part in request")
            return jsonify({"error": "No file uploaded"}), 400

        image = request.files["image"]
        if image.filename == "":
            app.logger.error("No selected file")
            return jsonify({"error": "No selected file"}), 400

        filename = secure_filename(image.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)

        try:
            image.save(filepath)
            app.logger.info(f"Image saved to {filepath}")
            # Further processing logic goes here, e.g., running an AI model on the image
            return jsonify({"message": f"Image {filename} processed successfully!"})
        except Exception as e:
            app.logger.error(f"Error saving image: {e}")
            return jsonify({"error": "Failed to save file"}), 500

    # Route for answering questions about the AI model
    @app.route("/ask", methods=["POST"])
    def ask_question():
        user_input = request.json.get("message")
        response = process_user_input(user_input)
        return jsonify({"response": response})

    def process_user_input(user_input):
        if "model" in user_input.lower():
            return "The AI model is based on CNN architecture."
        return "I don't understand the question."

    return app
