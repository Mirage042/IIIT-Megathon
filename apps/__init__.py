from flask import Flask, request, jsonify
from flask_login import LoginManager
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
import os
from importlib import import_module

db = SQLAlchemy()
login_manager = LoginManager()


def register_extensions(app):
    db.init_app(app)
    login_manager.init_app(app)


def register_blueprints(app):
    for module_name in ("authentication", "home"):
        module = import_module("apps.{}.routes".format(module_name))
        app.register_blueprint(module.blueprint)


def configure_database(app):
    @app.before_first_request
    def initialize_database():
        db.create_all()

    @app.teardown_request
    def shutdown_session(exception=None):
        db.session.remove()


from apps.authentication.oauth import github_blueprint


def create_app(config):
    app = Flask(__name__)
    app.config.from_object(config)
    register_extensions(app)

    app.register_blueprint(github_blueprint, url_prefix="/login")
    register_blueprints(app)
    configure_database(app)

    # Ensure the upload folder exists
    UPLOAD_FOLDER = os.path.join(os.getcwd(), "uploads")
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

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
            return jsonify({"message": f"Image {filename} processed successfully!"})
        except Exception as e:
            app.logger.error(f"Error saving image: {e}")
            return jsonify({"error": "Failed to save file"}), 500

        # Process the image (AI model integration here)
        response_data = {"message": f"Image {filename} processed successfully!"}
        return jsonify(response_data)

    # Route for answering questions about the AI model
    @app.route("/ask", methods=["POST"])
    def ask_question():
        user_input = request.json.get("message")
        response = process_user_input(user_input)
        return jsonify({"response": response})

    def process_user_input(user_input):
        # Sample response based on the AI model query
        if "model" in user_input.lower():
            return "The AI model is based on CNN architecture."
        return "I don't understand the question."

    return app
