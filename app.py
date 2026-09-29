from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import sqlite3
import time

app = Flask(__name__)

# YOLO model
model = YOLO("yolo11n.pt")

# Crowd limit
LIMIT = 3

# Latest detection
latest_count = 0
latest_status = "NORMAL"

last_saved_count = -1
last_saved_status = ""
last_saved_time = 0


# ==============================
# DATABASE
# ==============================

def save_crowd_data(count, status):

    try:

        connection = sqlite3.connect("crowd_data.db")

        cursor = connection.cursor()

        cursor.execute("""
            INSERT INTO crowd_logs
            (timestamp, people_count, status)
            VALUES (datetime('now', 'localtime'), ?, ?)
        """, (count, status))

        connection.commit()
        connection.close()

        print("Database saved:", count, status)

    except Exception as e:

        print("Database error:", e)


# ==============================
# HOME PAGE
# ==============================

@app.route("/")
def index():

    return render_template(
        "index.html",
        limit=LIMIT
    )


# ==============================
# AI DETECTION
# ==============================

@app.route("/detect", methods=["POST"])
def detect():

    global latest_count
    global latest_status
    global last_saved_count
    global last_saved_status
    global last_saved_time

    try:

        # Receive image from browser
        file = request.files["image"]

        image_bytes = file.read()

        # Convert image to OpenCV format
        np_array = np.frombuffer(
            image_bytes,
            np.uint8
        )

        frame = cv2.imdecode(
            np_array,
            cv2.IMREAD_COLOR
        )

        if frame is None:

            return jsonify({
                "success": False,
                "error": "Invalid image"
            })

        # YOLO person detection
        results = model(
            frame,
            verbose=False,
            classes=[0],
            conf=0.4
        )

        person_count = 0

        for result in results:

            for box in result.boxes:

                person_count += 1

        # Crowd status
        if person_count > LIMIT:

            status = "OVER CROWDED"

        else:

            status = "NORMAL"


        # Update latest values
        latest_count = person_count
        latest_status = status


        # Save database
        current_time = time.time()

        if (
            person_count != last_saved_count
            or status != last_saved_status
            or current_time - last_saved_time >= 5
        ):

            save_crowd_data(
                person_count,
                status
            )

            last_saved_count = person_count
            last_saved_status = status
            last_saved_time = current_time


        return jsonify({

            "success": True,

            "people": person_count,

            "status": status,

            "limit": LIMIT

        })


    except Exception as e:

        print("Detection error:", e)

        return jsonify({

            "success": False,

            "error": str(e)

        })


# ==============================
# STATUS
# ==============================

@app.route("/status")
def status():

    return jsonify({

        "people": latest_count,

        "status": latest_status,

        "limit": LIMIT,

        "camera": True

    })


# ==============================
# DATABASE HISTORY
# ==============================

@app.route("/history")
def history():

    connection = sqlite3.connect(
        "crowd_data.db"
    )

    cursor = connection.cursor()

    cursor.execute("""
        SELECT id, timestamp, people_count, status
        FROM crowd_logs
        ORDER BY id DESC
        LIMIT 10
    """)

    records = cursor.fetchall()

    connection.close()

    return jsonify(records)


# ==============================
# START SERVER
# ==============================

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )
