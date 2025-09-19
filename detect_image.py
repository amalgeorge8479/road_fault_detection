import cv2
import time
import csv
import sqlite3
from ultralytics import YOLO


MODEL_PATH = "best.pt"         
IMAGE_PATH = "road.jpg"    
OUTPUT_IMAGE = "output.jpg"     
CSV_FILE = "detections.csv"
DB_FILE = "detections.db"
CONF_THRESHOLD = 0.7           

model = YOLO(MODEL_PATH)

frame = cv2.imread(IMAGE_PATH)
if frame is None:
    raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")
frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))

conn = sqlite3.connect(DB_FILE)
cursor = conn.cursor()
cursor.execute('''CREATE TABLE IF NOT EXISTS faults (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    detected_class TEXT,
                    confidence REAL,
                    x1 INTEGER,
                    y1 INTEGER,
                    x2 INTEGER,
                    y2 INTEGER
                )''')

csv_file = open(CSV_FILE, mode="a", newline="")
csv_writer = csv.writer(csv_file)

if csv_file.tell() == 0:
    csv_writer.writerow(["timestamp", "class", "confidence", "x1", "y1", "x2", "y2"])

results = model.predict(frame, conf=CONF_THRESHOLD, verbose=False)

detections_found = False

for r in results:
    for box in r.boxes:
        detections_found = True
        cls = model.names[int(box.cls[0])]
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{cls} {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        csv_writer.writerow([timestamp, cls, conf, x1, y1, x2, y2])

        cursor.execute("""INSERT INTO faults 
                          (timestamp, detected_class, confidence, x1, y1, x2, y2) 
                          VALUES (?,?,?,?,?,?,?)""",
                       (timestamp, cls, conf, x1, y1, x2, y2))
        conn.commit()

cv2.imwrite(OUTPUT_IMAGE, frame)

if detections_found:
    print("✅ Detections completed. Results saved to CSV, SQLite, and output image.")
else:
    print("⚠️ No faults detected in this image.")

cv2.imshow("Road Fault Detection", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

csv_file.close()
conn.close()
