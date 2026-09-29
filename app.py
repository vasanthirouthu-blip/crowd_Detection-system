from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import onnxruntime as ort
import sqlite3
import time
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "yolo11n.onnx"
)

session = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name

LIMIT = 3

latest_count = 0
latest_status = "NORMAL"

last_saved_count = -1
last_saved_status = ""
last_saved_time = 0


def save_crowd_data(count, status):
    try:
        connection = sqlite3.connect(
            os.path.join(BASE_DIR, "crowd_data.db")
        )

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


def detect_people(frame):

    image = cv2.resize(frame, (640, 640))

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    image = image.astype(
        np.float32
    ) / 255.0

    image = np.transpose(
        image,
        (2, 0, 1)
    )

    image = np.expand_dims(
        image,
        axis=0
    )

    outputs = session.run(
        None,
        {input_name: image}
    )

    predictions = outputs[0][0]

    person_count = 0

    for detection in predictions.T:

        class_scores = detection[4:]

        class_id = int(
            np.argmax(class_scores)
        )

        confidence = float(
            class_scores[class_id]
        )

        if (
            class_id == 0
            and confidence >= 0.40
        ):
            person_count += 1

    return person_count


@app.route("/")
def index():

    return render_template(
        "index.html",
        limit=LIMIT
    )


@app.route(
    "/detect",
    methods=["POST"]
)
def detect():

    global latest_count
    global latest_status
    global last_saved_count
    global last_saved_status
    global last_saved_time

    try:

        file = request.files["image"]

        image_bytes = file.read()

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

        person_count = detect_people(
            frame
        )

        if person_count > LIMIT:

            status = "OVER CROWDED"

        else:

            status = "NORMAL"

        latest_count = person_count
        latest_status = status

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

        print(
            "Detection error:",
            e
        )

        return jsonify({

            "success": False,

            "error": str(e)

        })


@app.route("/status")
def status():

    return jsonify({

        "people": latest_count,

        "status": latest_status,

        "limit": LIMIT,

        "camera": True

    })


@app.route("/history")
def history():

    try:

        connection = sqlite3.connect(
            os.path.join(
                BASE_DIR,
                "crowd_data.db"
            )
        )

        cursor = connection.cursor()

        cursor.execute("""
            SELECT id, timestamp,
                   people_count, status
            FROM crowd_logs
            ORDER BY id DESC
            LIMIT 10
        """)

        records = cursor.fetchall()

        connection.close()

        return jsonify(records)

    except Exception as e:

        print(
            "History error:",
            e
        )

        return jsonify([])


if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )
