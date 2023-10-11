# Image-to-Cartoon Flask API Documentation

## Introduction
This API provides image-to-cartoon conversion functionality. Users can register, log in, and upload an image to receive a cartoonized version of the image.

## Endpoints

### Register a User (POST)
- **URL**: `/register`
- **Description**: Register a new user.
- **Request Parameters**:
  - `username` (string, required): The username for registration.
  - `password` (string, required): The password for registration.
- **Response**:
  - Successful registration: Status code 201 (Created) and a JSON response with a success message.
  - Registration failure: Status code 400 (Bad Request) and a JSON response with an error message.

### Log In and Obtain Token (POST)
- **URL**: `/login`
- **Description**: Log in and obtain a token for accessing protected resources.
- **Request Parameters**:
  - `username` (string, required): The username for login.
  - `password` (string, required): The password for login.
- **Response**:
  - Successful login: Status code 302 (Found) and a redirect to the protected resource with a token.
  - Login failure: Status code 401 (Unauthorized) and a JSON response with an error message.

### Upload Image for Cartoonization (POST)
- **URL**: `/protected`
- **Description**: Upload an image for cartoonization (protected resource).
- **Request Parameters**:
  - `file` (file, required): The image file to be cartoonized.
- **Response**:
  - Successful upload: Status code 200 (OK) and a rendered HTML page displaying the cartoonized image.
  - Invalid token or token expiration: Status code 401 (Unauthorized) and a JSON response with an error message.

## Authentication
- Authentication for protected resources is token-based. Users obtain a token by logging in, and they must include the token in the request headers for protected resource access.


