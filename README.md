# Face and Speech Recognition System with MongoDB Integration

This project is a real-time face and speech recognition system that uses computer vision, voice recognition, and a MongoDB database to identify people, capture what they say, and store or update their information.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [How It Works](#how-it-works)
- [Future Improvements](#future-improvements)
- [License](#license)

## Overview
This application captures real-time facial landmarks using a camera feed and analyzes the speech of the detected person. It stores the speech transcript along with facial distance information in a MongoDB database. When a person speaks again, their entry is updated in the database based on the closest matching facial distance.

## Features
- **Face Mesh Detection**: Detects and tracks facial landmarks using MediaPipe.
- **Distance Calculation**: Calculates facial distances to identify the user.
- **Speech Recognition**: Captures speech through a microphone and converts it to text.
- **MongoDB Integration**: Stores user data in a MongoDB instance and updates entries when the same user is recognized based on speech.
- **Real-Time Processing**: Uses threading to perform speech and facial recognition concurrently.
- **Color-Coded Feedback**: Visual feedback indicating if a user's distance from the camera is within a specific range.

## Technologies Used
- **Python**: The primary programming language.
- **OpenCV**: For camera feed and facial landmark visualization.
- **MediaPipe**: For facial landmark detection.
- **SpeechRecognition**: For converting speech to text.
- **PyMongo**: For MongoDB interaction.
- **Threading**: To run concurrent processes for face and speech recognition.
- **NumPy**: For mathematical operations and distance calculations.

## Usage
1. Update the `database.py` file to include the correct MongoDB connection details:
    ```python
    from pymongo import MongoClient

    client = MongoClient("mongodb://your-mongodb-uri")
    collection = client['your-database']['your-collection']
    ```
2. Run the program:
    ```bash
    python facial_speech_recognition.py
    ```
3. Once the program starts, it will:
   - Access your webcam to capture facial data.
   - Listen for speech through your microphone.
   - Calculate facial distances and compare them to entries in the MongoDB collection.
   - Insert new records or update existing ones based on the user's speech and facial distance.

## Configuration
You can modify some parameters in `main.py`:
- **Reference Distance & Scaling Factor**: Customize the reference real-world distance and corresponding pixel distance to adjust scaling for facial landmark distances.
- **Speech Recognition Timeout**: Adjust the time limits for capturing and processing speech in the `listen_for_speech` function.
- **Face Mesh Detection Confidence**: Adjust the `min_detection_confidence` parameter for more or less sensitivity in facial recognition.

## How It Works
1. **Facial Detection & Distance Measurement**:
   - Uses the camera feed to detect facial landmarks (e.g., nose tip, lips, eyes).
   - Calculates distances between key facial features and scales them to a real-world distance.
   - Provides visual feedback based on the user's distance from the camera.

2. **Speech Recognition**:
   - Continuously listens for speech through the microphone.
   - Converts recognized speech into text and stores it as `recognized_text`.

3. **MongoDB Integration**:
   - Every 5 seconds, checks if there is recognized speech.
   - Calculates the average distance between facial landmarks.
   - Finds the closest match in the MongoDB collection based on the distance.
   - **If a match is found**: Updates the existing entry with the new speech and distance.
   - **If no match is found**: Inserts a new entry with the current speech and distance.

4. **Data Print Every 10 Seconds**:
   - Prints the text of the database entry closest to the current facial distance every 10 seconds.

## Future Improvements
- **Better Facial Recognition**: Implement a more accurate facial recognition algorithm to improve matching accuracy.
- **Error Handling**: Improve handling of speech recognition errors and database connection failures.
- **Optimized Performance**: Explore options to enhance real-time performance for both speech and facial recognition.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---

Feel free to contribute to this project by submitting issues or pull requests!

