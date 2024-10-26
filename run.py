# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""
import os
from flask import request, jsonify
from flask_migrate import Migrate
from flask_minify import Minify
from sys import exit
from apps import (
    create_app,
    db,
    generate_response,
)  # Import generate_response from __init__.py
from apps.config import config_dict

# WARNING: Don't run with debug turned on in production!
DEBUG = os.getenv("DEBUG", "False") == "True"

# The configuration
get_config_mode = "Debug" if DEBUG else "Production"

try:
    # Load the configuration using the default values
    app_config = config_dict[get_config_mode.capitalize()]
except KeyError:
    exit("Error: Invalid <config_mode>. Expected values [Debug, Production] ")

app = create_app(app_config)
Migrate(app, db)

if not DEBUG:
    Minify(app=app, html=True, js=False, cssless=False)

if DEBUG:
    app.logger.info("DEBUG       = " + str(DEBUG))
    app.logger.info("DBMS        = " + app_config.SQLALCHEMY_DATABASE_URI)
    app.logger.info("ASSETS_ROOT = " + app_config.ASSETS_ROOT)


@app.route("/chatbot", methods=["POST"])
def chatbot():
    data = request.get_json()
    user_message = data.get("message")
    response = generate_response(user_message)  # Call the function from __init__.py
    return jsonify({"response": response})


if __name__ == "__main__":
    app.run()
