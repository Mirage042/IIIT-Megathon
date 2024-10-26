from flask import Flask, request, jsonify
from flask_login import LoginManager
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
import os
from importlib import import_module
import google.generativeai as genai  # Import the Google Generative AI library
from datetime import datetime
from . import ML  # Import your ML.py functions

# Initialize extensions
db = SQLAlchemy()
login_manager = LoginManager()

# Set up API key for Gemini
API_KEY = os.getenv("API_KEY", "AIzaSyABSVGA83cLKJ-RhSFFbzMJkzG3aYE_YWE")
genai.configure(api_key=API_KEY)  # Configure the Gemini API key


# Define database model for user interactions
class UserInteraction(db.Model):
    __tablename__ = "user_interactions"
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    user_input = db.Column(db.String, nullable=False)
    polarity = db.Column(db.String, nullable=False)
    extracted_concern = db.Column(db.String, nullable=False)


# Define database model for all metrics
class AllMetrics(db.Model):
    __tablename__ = "all_metrics"
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    polarity = db.Column(db.String, nullable=False)
    extracted_concern = db.Column(db.String, nullable=False)
    classified_disorder = db.Column(db.String, nullable=False)
    sentiment_score = db.Column(db.Float, nullable=False)
    sentiment_shift = db.Column(db.String, nullable=True)


def generate_response(message):
    """
    Generate a response from the Gemini API based on user input.
    Args:
        message (str): The user input message.
    Returns:
        str: The response from the API.
    """

    # Define the personal assistant context
    context = (
        "You are a personal assistant designed to help users with various tasks. "
        "You can manage schedules, set reminders, provide information, and assist with everyday queries. "
        "Your main objectives are to: "
        "1. Understand user requests clearly and respond appropriately. "
        "2. Offer suggestions based on user preferences and previous interactions. "
        "3. Provide timely reminders and updates about scheduled events. "
        "4. Answer questions with concise and accurate information. "
        "5. Learn from user feedback to improve assistance quality over time. "
        "Format your response as follows, without any extra explanation: "
        "- Task: [Task or question] "
        "- Suggestion: [Suggested action or information] "
        "- Reminder: [If applicable, details about the reminder or task]"
    )

    # Combine the context with the user message
    full_input = context + " " + message

    # Create a model instance for generating content
    model = genai.GenerativeModel(
        "gemini-1.5-flash"
    )  # Adjust the model name if necessary

    # Generate a response
    response = model.generate_content(full_input)

    # Return the generated text response
    return response.text


# Example usage (uncomment for testing)
# if __name__ == "__main__":
#     user_message = "What is the weather like today?"
#     response = generate_response(user_message)
#     print(response)


# Function to get extracted concerns from the database
def get_extracted_concerns():
    extracted_concerns = UserInteraction.query.with_entities(
        UserInteraction.extracted_concern
    ).all()
    concerns_list = [concern[0] for concern in extracted_concerns]
    return concerns_list


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

    # SQLite configuration
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///usernew_interactions.db"
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

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
            except FileNotFoundError:
                return jsonify({"error": "No image found in the uploads folder."}), 404

            # Get the list of extracted concerns for sentiment shift
            concerns_list = get_extracted_concerns()

            # Define context for the model including sentiment shift
            context = (
                "You are a professional mental health assistant that analyzes user text inputs to detect sentiment, extract specific concerns, "
                "and generate personalized insights and recommendations. You have access to historical inputs and previously extracted concerns, allowing you to track sentiment shifts and detect emerging patterns: "
                + ", ".join(concerns_list)
                + ". "
                "For each user input, please: "
                "1. Identify the sentiment as Positive, Negative, or Neutral, and track any emotional shifts over time. "
                "2. Extract key words or phrases that reflect the user's primary concerns, emotions, or emotional states. "
                "3. Classify each concern into specific mental health categories: Anxiety Disorders, Mood Disorders, Stress-Related Disorders, Interpersonal Stress, or Obsessive-Compulsive Disorders. "
                "4. Use weighted keyword scoring to assess the intensity of each concern. Use these specific weights for core keywords: "
                "    - 'anxious': 1.2, "
                "    - 'nervous': 1.2, "
                "    - 'overwhelmed': 1.2, "
                "    - 'panic': 1.2, "
                "    - 'tense': 1.2, "
                "    - 'sad': 1.5, "
                "    - 'hopeless': 1.5, "
                "    - 'depressed': 1.5, "
                "    - 'empty': 1.5, "
                "    - 'fatigued': 1.5, "
                "    - 'unstable': 1.3, "
                "    - 'irritable': 1.3, "
                "    - 'joyful': 1.3, "
                "    - 'mood swings': 1.3, "
                "    - 'compulsive': 1.4, "
                "    - 'obsessive': 1.4, "
                "    - 'ritual': 1.4, "
                "    - 'traumatic': 1.5, "
                "    - 'flashback': 1.5, "
                "    - 'avoidant': 1.5, "
                "    - 'fear': 1.5, "
                "    - 'lonely': 1.1, "
                "    - 'isolated': 1.1, "
                "    - 'misunderstood': 1.1, "
                "    - 'conflict': 1.1, "
                "    - 'binge': 1.6, "
                "    - 'restrict': 1.6, "
                "    - 'body image': 1.6, "
                "    - 'unhealthy': 1.6, "
                "    - 'craving': 1.4, "
                "    - 'withdrawal': 1.4, "
                "    - 'addiction': 1.4, "
                "    - 'substance': 1.4, "
                "    - 'manic': 1.4, "
                "    - 'euphoric': 1.4, "
                "    - 'extreme mood': 1.4, "
                "    - 'intense': 1.3, "
                "    - 'difficult': 1.3, "
                "    - 'relationships': 1.3. "
                "If an unknown keyword is detected, determine the closest related weight based on similarity with the provided keywords. "
                "Assign a weight based on context, or default to a score of 1.0 if no related keyword exists. "
                "5. Generate a cumulative intensity score if multiple high-intensity keywords appear in a single input, reflecting compounded emotional impact. "
                "6. Monitor user-specific sentiment trajectories over time, recognizing patterns or shifts toward worsening or improving mental health conditions. "
                "Use these patterns to anticipate sentiment changes and identify emerging risk factors based on recent interactions. "
                "7. Predict sentiment shifts based on recent user input history. If a user shows signs of deteriorating sentiment, such as increased frequency of keywords like 'hopeless,' 'fatigued,' or 'lonely,' adjust the model's sensitivity to these concerns accordingly. "
                "8. Provide automated reasoning behind the sentiment shift analysis by identifying core keywords that contribute most significantly to the detected change. For instance, if a shift from 'anxious' to 'isolated' occurs, highlight this as a possible sign of worsening interpersonal stress. "
                "9. Generate customized recommendations based on detected sentiment patterns. Tailor actions to help users address their specific emotional concerns. For example: "
                "    - Anxiety Disorders: Suggest deep breathing exercises or mindfulness techniques. "
                "    - Mood Disorders: Suggest journaling or seeking support from a mental health professional. "
                "    - Stress-Related Disorders: Recommend relaxation practices or small lifestyle adjustments to reduce daily stress. "
                "    - Interpersonal Stress: Suggest communication techniques or virtual social interactions to alleviate feelings of isolation. "
                "    - Obsessive-Compulsive Disorders: Recommend structured routines or mental exercises to ease compulsive behavior. "
                "Format your response as follows, without any extra explanation: "
                "- Polarity: [Positive/Negative/Neutral] "
                "- Extracted Concern: [Phrase or keywords] "
                "- Classified Disorder: [Disorder based on keywords] "
                "- Sentiment Score: [Calculated score based on weights] "
                "- Sentiment Shift: [Based on historical context] "
                "- Recommendation: [Action or resource suggestion based on classified disorder]"
            )

            full_input = context + " " + message
            model = genai.GenerativeModel("gemini-1.5-flash")  # Using Gemini API
            response = model.generate_content(full_input)

            # Extract data from response
            response_lines = response.text.splitlines()
            polarity = next(
                (
                    line.split(": ")[1]
                    for line in response_lines
                    if line.startswith("- Polarity:")
                ),
                "Unknown",
            )
            extracted_concern = next(
                (
                    line.split(": ")[1]
                    for line in response_lines
                    if line.startswith("- Extracted Concern:")
                ),
                "None",
            )
            classified_disorder = next(
                (
                    line.split(": ")[1]
                    for line in response_lines
                    if line.startswith("- Classified Disorder:")
                ),
                "Unknown",
            )
            sentiment_score = float(
                next(
                    (
                        line.split(": ")[1]
                        for line in response_lines
                        if line.startswith("- Sentiment Score:")
                    ),
                    0,
                )
            )
            sentiment_shift = next(
                (
                    line.split(": ")[1]
                    for line in response_lines
                    if line.startswith("- Sentiment Shift:")
                ),
                "None",
            )

            # Log interaction in UserInteraction table
            interaction = UserInteraction(
                user_input=message,
                polarity=polarity,
                extracted_concern=extracted_concern,
            )
            db.session.add(interaction)

            # Log metrics in AllMetrics table
            metrics = AllMetrics(
                polarity=polarity,
                extracted_concern=extracted_concern,
                classified_disorder=classified_disorder,
                sentiment_score=sentiment_score,
                sentiment_shift=sentiment_shift,
            )
            db.session.add(metrics)
            db.session.commit()

            # Send chatbot response to front-end
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
            return jsonify({"success": True}), 200
        except Exception as e:
            app.logger.error(f"Failed to save image: {str(e)}")
            return jsonify({"error": "Failed to save image"}), 500

    return app
