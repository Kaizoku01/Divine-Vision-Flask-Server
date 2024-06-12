# Divine Vision Flask Server

This is Backend Server made in flask for our Divine Vision Mobile Application, for usage of our machine learning model that is made to identify and generate a caption according to the image passed of the Indian God and Goddess.

## Overview

This project is aimed at generating captions for images of Indian gods and goddesses using a custom-built machine learning model. The model is integrated into a Flask server serving as the backend for a mobile application developed using Flutter. By sending images through the mobile app to the Flask server's /predict route, users can receive descriptive captions for the input images.

## Features

- Image caption generation for Indian god and goddess images.
- Custom-built machine learning model for accurate captions.
- Integration with a Flutter mobile application.
- Flask server backend for handling image caption requests.

## Dependencies
This project uses the following technologies and frameworks:

- Python: Python is a high-level, dynamically typed programming language known for its simplicity and readability.
- https: A package for creating api calls to establish a communication medium b/w backend and frontend.
- [Divine Vision Mobile Application](https://github.com/Kaizoku01/Divine-Vision-Mobile-Application)

## Usage
STEP 1 :
``bash
python -m venv venv
``
<br>
STEP 2 :
``bash
venv/Scripts/activate
``
<br>
STEP 3 :
``bash
pip install Flask
``
<br>
STEP 4 :
``bash
python app.py
``

## Contributing

Contributions to this project are welcome! If you'd like to contribute, please follow the usual GitHub flow: fork the repository, make your changes, and submit a pull request.
