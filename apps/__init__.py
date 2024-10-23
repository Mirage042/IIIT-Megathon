from flask import Flask, request, jsonify, redirect, url_for, render_template
from flask_login import LoginManager
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
import os
from importlib import import_module
import google.generativeai as genai  # Import the Google Generative AI library
import subprocess
from . import ML  # Import your ML.py functions

ml_script_path = os.path.join(os.path.dirname(__file__), "ML.py")

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
    @app.route("/chat", methods=["POST"])
    def chat():
        try:
            data = request.json
            message = data.get("message")
            if not message:
                return jsonify({"error": "Message is required"}), 400

            # Get latest image and calculate metrics using ML.py functions
            try:
                latest_image_path = ML.get_latest_image(app.config["UPLOAD_FOLDER"])
                image = ML.io.imread(latest_image_path)
                brightness = ML.calculate_brightness(image)
                contrast = ML.calculate_contrast(image)
                sharpness = ML.calculate_sharpness(image)
                edge_intensity = ML.calculate_edge_intensity(image)
            except FileNotFoundError:
                return jsonify({"error": "No image found in the uploads folder."}), 404

            # Define context for the model
            context = (
                "You are a professional assistant that helps users understand image metrics. "
                f"Brightness: {brightness:.2f}, Contrast: {contrast:.2f}, "
                f"Sharpness: {sharpness:.2f}, Edge Intensity: {edge_intensity:.2f}."
            )

            full_input = context + " " + message
            model = genai.GenerativeModel("gemini-1.5-flash")  # Using Gemini API
            response = model.generate_content(full_input)

            # Send both the chatbot response and metrics like brightness to the front-end
            return jsonify(
                {
                    "response": response.text,
                    "brightness": brightness,
                    "contrast": contrast,
                    "sharpness": sharpness,
                    "edge_intensity": edge_intensity,
                }
            )

        except Exception as e:
            app.logger.error(f"Error in /chat endpoint: {str(e)}")
            return jsonify({"error": "Internal server error"}), 500

    # Process image and generate metrics plot
    def process_image(image_path):
        # Define output path for the plot
        output_plot_path = os.path.join("static", "images", "metrics_plot.png")

        # Calculate metrics
        metrics = ML.calculate_metrics(image_path)

        # Plot metrics and save the image
        ML.plot_metrics(metrics, output_plot_path)

    # Route for image upload
    @app.route("/upload", methods=["POST"])
    def upload_image():
        if "image" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["image"]

        if file.filename == "":
            return jsonify({"error": "No selected file"}), 400

        if file:
            # Save the uploaded file
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(filepath)

            # Run ML.py
            subprocess.run(
                ["python", ml_script_path, filepath]
            )  # Pass the path of the uploaded image as an argument

            return jsonify(
                {"message": f"Image {file.filename} processed successfully!"}
            )

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
