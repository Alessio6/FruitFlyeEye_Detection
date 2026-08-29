import os
import json
import uuid
import tempfile
from flask import Flask, render_template, request, redirect, url_for, session, send_file, jsonify
import cv2
import numpy as np
from ultralytics import YOLO
import pathlib
import platform
import sqlite3
import hashlib
import yaml
import pycountry
from datetime import datetime
import pandas as pd
import io
import re
from PIL import Image
import zipfile
import shutil
from io import BytesIO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
from scipy.spatial import distance

app = Flask(__name__)
app.secret_key = "dh7B6aglhan4GAg6AH"

if platform.system() == 'Windows': #Might need more for other OS types
    pathlib.PosixPath = pathlib.WindowsPath
else:
    pathlib.WindowsPath = pathlib.PosixPath
#Loading both object detection models
model = YOLO('best.pt')
oda_model = YOLO('best_oda.pt')

_SESSION_DIR = os.path.join(tempfile.gettempdir(), "flyapp_sessions")
os.makedirs(_SESSION_DIR, exist_ok=True)

#Builds a safe file path for a session queue file by stripping any characters that could cause issues in a filename, such as strokes or spaces.
def _session_path(key):
    safe = "".join(c for c in key if c.isalnum() or c == "-")
    return os.path.join(_SESSION_DIR, f"{safe}.json")

#Serialises the current queue state to a JSON file on disk. This is necessary because Flask does not keep data in memory between requests
def store_queue_data(pending, queue, queue_total):
    key  = session.get("queue_key") or str(uuid.uuid4())
    data = {"pending": pending, "queue": queue, "queue_total": queue_total}
    with open(_session_path(key), "w") as fh:
        json.dump(data, fh)
    session["queue_key"] = key
    session.modified = True

#Reads the queue state from disk. Returns safe defaults if no queue file exists, such as when no upload has been started yet
def load_queue_data():
    key = session.get("queue_key")
    if not key:
        return None, [], 0
    path = _session_path(key)
    if not os.path.exists(path):
        return None, [], 0
    with open(path) as fh:
        data = json.load(fh)
    return data.get("pending"), data.get("queue", []), data.get("queue_total", 0)

#Updates the pending item and remaining queue whilst preserving the original queue_total, so progress counters remain accurate throughout the batch
def update_queue_data(pending, queue):
    _, _, queue_total = load_queue_data()
    store_queue_data(pending, queue, queue_total)

#Removes both the session key and the associated queue file from disk. Called once all images in a batch have been processed
def clear_queue_data():
    key = session.pop("queue_key", None)
    session.modified = True
    if key:
        path = _session_path(key)
        if os.path.exists(path):
            os.remove(path)

#Login database creation
def init_db():
    conn = sqlite3.connect("flytest.db")
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            email TEXT NOT NULL UNIQUE,
            password TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

init_db()
#Image database creation
def init_image_results_db():
    conn = sqlite3.connect("flytest.db")
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS image_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_path TEXT NOT NULL,
            annotated_image TEXT,
            image_origin TEXT NOT NULL,
            image_date TEXT NOT NULL,
            result TEXT NOT NULL,
            created_by_id INTEGER,
            public INTEGER DEFAULT 0
        )
    """)
    try:
        cur.execute("ALTER TABLE image_results ADD COLUMN public INTEGER DEFAULT 0")
    except sqlite3.OperationalError:
        pass
    conn.commit()
    conn.close()

init_image_results_db()
#Object detection image database creation
def init_object_results_db(): #Needed for main table
    conn = sqlite3.connect("flytest.db")
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS object_results (
            main_id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_id INTEGER,
            image_path TEXT NOT NULL,
            date DATE,
            location TEXT,
            object_id INTEGER,
            object_type TEXT NOT NULL,
            polygon_net TEXT NOT NULL,
            centroid TEXT NOT NULL,
            imgwidth_ppx INTEGER,
            area_ppx2 INTEGER,
            perimeter_ppx INTEGER,
            imgwidth_um DOUBLE,
            area_um2 DOUBLE,
            perimeter_um DOUBLE,
            confidence DOUBLE,
            bbox_width_px DOUBLE,
            bbox_height_px DOUBLE,
            major_axis_um DOUBLE,
            minor_axis_um DOUBLE,
            bbox_width_um DOUBLE,
            bbox_height_um DOUBLE,
            scale_source TEXT
        )
    """)
    for col, typedef in [
        ("perimeter_ppx",  "INTEGER"),
        ("perimeter_um",   "DOUBLE"),
        ("bbox_width_px",  "DOUBLE"),
        ("bbox_height_px", "DOUBLE"),
        ("major_axis_um",  "DOUBLE"),
        ("minor_axis_um",  "DOUBLE"),
        ("bbox_width_um",  "DOUBLE"),
        ("bbox_height_um", "DOUBLE"),
        ("scale_source",   "TEXT"),
    ]:
        try:
            cur.execute(f"ALTER TABLE object_results ADD COLUMN {col} {typedef}")
        except sqlite3.OperationalError:
            pass
    try:
        cur.execute("ALTER TABLE object_results ADD COLUMN perimeter_ppx INTEGER")
    except sqlite3.OperationalError:
        pass
    try:
        cur.execute("ALTER TABLE object_results ADD COLUMN perimeter_um DOUBLE")
    except sqlite3.OperationalError:
        pass
    conn.commit()
    conn.close()
#ODA image database creation
def init_ODA_db():
    conn = sqlite3.connect("flytest.db")
    cur = conn.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS ODA_root(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_path TEXT NOT NULL,
            annotated_image TEXT,
            image_origin TEXT NOT NULL,
            image_date TEXT NOT NULL,
            result TEXT NOT NULL,
            created_by_id INTEGER,
            public INTEGER DEFAULT 0
        )
    """)
    cur.execute("""CREATE TABLE IF NOT EXISTS ODA_object(
            main_id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_id INTEGER,
            image_path TEXT NOT NULL,
            date DATE,
            location TEXT,
            mask_net TEXT NOT NULL,
            mask_centroid TEXT NOT NULL,
            mask_imgwidth_ppx INTEGER,
            mask_area_ppx2 INTEGER,
            mask_perimeter_ppx INTEGER,
            mask_imgwidth_um DOUBLE,
            mask_area_um2 DOUBLE,
            mask_perimeter_um DOUBLE,
            mask_confidence DOUBLE,
            ommatidia_count INTEGER,
            avg_diameter_px DOUBLE,
            avg_diameter_real DOUBLE,
            diameter_sd_real DOUBLE)""")
    conn.commit()
    conn.close()

init_image_results_db()
init_object_results_db()
init_ODA_db()

def analyse_eyes(filepath): #Original
    img = cv2.imread(filepath)

    if img is None:
        return None, None, "Failed to load image."

    results = model(img) # Run inference on the image and store detection results

    detection = [] # List to store detection data for each detected object
    for r in results:
        for box in r.boxes: # Iterates through detected bounding box
            label = model.names[int(box.cls)] # Get class name form class index
            confidence = float(box.conf)      #Get confidence score as a float
            x1, y1, x2, y2 = map(int, box.xyxy[0]) # Extract bounding box coordinates
            # append detection details to list
            detection.append({
                "label": label,
                "confidence": confidence,
                "bbox": [[x1, y1, x2, y2]]
            })
            #Draws bouning box for annotated image
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"{label} {confidence:.2f}", (x1, y1 - 10),  # draws conifidence score above bounding box
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    #Build filename and defines fike path for the annotated image
    annotated_filename = "annotated_" + os.path.splitext(os.path.basename(filepath))[0] + ".jpg"
    annotated_path = os.path.join("static/uploads", annotated_filename)
    cv2.imwrite(annotated_path, img)

    return detection, annotated_filename, None

def polygon_analyse_eyes(filepath): #Modified code
    img = cv2.imread(filepath)
    

    if img is None:
        return None, None, "Failed to load image."

    #Needs to be 3 channel RGB
    if len(img.shape) == 2 or img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    results = model(img)
    detection = []

    with open('OriginalDataset/dataset.yaml', 'r') as f:
        data_config = yaml.safe_load(f)
    colors = data_config.get('colors')

    for r in results:
        if r.masks is not None:
            for mask, box in zip(r.masks.xy, r.boxes):
                label = model.names[int(box.cls)]
                color = colors[int(box.cls)]
                confidence = float(box.conf)
                polygon_points = np.array(mask, dtype=np.int32)

                detection.append({
                    "label": label,
                    "confidence": confidence,
                    "polygon": mask.tolist()
                })

                cv2.polylines(img, [polygon_points], isClosed=True, color=color, thickness=2)
                overlay = img.copy()
                cv2.fillPoly(overlay, [polygon_points], color)
                cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)

                text_x, text_y = polygon_points[polygon_points[:, 1].argmin()]
                cv2.putText(img, f"{label} {confidence:.2f}", (text_x, text_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    annotated_filename = "annotated_" + os.path.splitext(os.path.basename(filepath))[0] + ".jpg"
    annotated_path = os.path.join("static/uploads", annotated_filename)
    cv2.imwrite(annotated_path, img)

    return detection, annotated_filename, None

def read_tif_scale_um_per_px(filepath):
    """
    Returns µm per pixel from TIFF XResolution tag.
    ResolutionUnit=2 → inches; ResolutionUnit=3 → cm.
    Returns None if not a TIFF or tag is missing.
    """
    try:
        import tifffile
        with tifffile.TiffFile(filepath) as tif:
            tags = {t.name: t.value for t in tif.pages[0].tags.values()}
            xres = tags.get('XResolution')   # tuple (numerator, denominator)
            unit = tags.get('ResolutionUnit', 2)  # 2=inch, 3=cm
            if xres and xres[1] != 0:
                px_per_unit = xres[0] / xres[1]
                unit_to_um = 25400 if unit == 2 else 10000  # inch→µm or cm→µm
                return unit_to_um / px_per_unit   # µm per pixel
    except Exception:
        pass
    return None

def measure_eye(polygon_points):
    """
    Given a polygon as a list of [x, y] pairs, returns:
      - bbox_width_px, bbox_height_px  (axis-aligned bounding box)
      - major_axis_px, minor_axis_px   (PCA-based ellipse fit)
      - area_px2, perimeter_px         (already in calculate_geometry, repeated here)
    """
    pts = np.array(polygon_points, dtype=np.float32)
    
    # Axis-aligned bounding box
    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)
    bbox_w = float(x_max - x_min)
    bbox_h = float(y_max - y_min)

    # PCA for major/minor axes
    centered = pts - pts.mean(axis=0)
    cov = np.cov(centered.T)
    eigenvalues, _ = np.linalg.eigh(cov)
    eigenvalues = np.sort(eigenvalues)[::-1]
    # 2*sqrt(eigenvalue) ≈ axis length (like an ellipse)
    major_ax = 2 * float(np.sqrt(max(eigenvalues[0], 0)))
    minor_ax = 2 * float(np.sqrt(max(eigenvalues[1], 0)))

    return {
        'bbox_width_px':  round(bbox_w, 2),
        'bbox_height_px': round(bbox_h, 2),
        'major_axis_px':  round(major_ax, 2),
        'minor_axis_px':  round(minor_ax, 2),
    }


def calculate_geometry(polygon):
    #Finds the centroid, area and perimter of a polygon provided
    x = [p[0] for p in polygon]
    y = [p[1] for p in polygon]

    centroid_x = sum(x) / len(polygon)
    centroid_y = sum(y) / len(polygon)

    area = 0.5 * abs(sum(x[i] * y[i+1] - x[i+1] * y[i] for i in range(-1, len(polygon)-1)))
    perimeter = sum(((polygon[i][0] - polygon[i-1][0])**2 + (polygon[i][1] - polygon[i-1][1])**2)**0.5 for i in range(len(polygon)))

    return f"{round(centroid_x, 2)}, {round(centroid_y, 2)}", int(area), int(perimeter)

def process_detections(image_origin, image_date, data, image_id=1, image_path="image.jpg", size=1, scale="ppx", imgwidth=1, img_set="full_image"):
    results = []
    for idx, item in enumerate(data):
        polygon = item['polygon']
        size = float(size)
        imgwidth = int(imgwidth)
    
        
        centroid, area, perimeter = calculate_geometry(polygon)
        measurements = measure_eye(polygon)
        um_per_px = read_tif_scale_um_per_px(image_path)

        # Determine scale factor and track the source
        if um_per_px:
            scale_factor = um_per_px
            scale_source = "TIF_Metadata"
            effective_size = um_per_px * imgwidth
            convert_scale = scale_factor
        else:
            if scale == "nm":
                size = size / 1000
                scale = "um"
                
            effective_size = size * imgwidth if img_set == "per_pixel" else size
            convert_scale = (effective_size / imgwidth) ** 2
            scale_factor = effective_size / imgwidth
            scale_source = "User_Supplied"

        # Calculate Micron-based dimensions
        bbox_w_um  = round(measurements['bbox_width_px']  * scale_factor, 3)
        bbox_h_um  = round(measurements['bbox_height_px'] * scale_factor, 3)
        major_um   = round(measurements['major_axis_px']  * scale_factor, 3)
        minor_um   = round(measurements['minor_axis_px']  * scale_factor, 3)
        #Add to record
        record = (
            image_id,                       # image_id
            image_path,                     # image_path
            image_date,                     # date
            image_origin,                   # location
            idx,                            # object_id
            item['label'],                  # object_type
            json.dumps(polygon),            # polygon_net
            centroid,                       # centroid
            imgwidth,                       # imgwidth_ppx
            area,                           # area_ppx2
            perimeter,                      # perimeter_ppx
            effective_size,                 # imgwidth_um
            float(area) * convert_scale,    # area_um2
            float(perimeter) * (convert_scale ** 0.5), # perimeter_um
            round(item['confidence'], 5),   # confidence
            measurements['bbox_width_px'],  # bbox_width_px
            measurements['bbox_height_px'], # bbox_height_px
            major_um,                       # major_axis_um
            minor_um,                       # minor_axis_um
            bbox_w_um,                      # bbox_width_um
            bbox_h_um,                      # bbox_height_um
            scale_source                    # scale_source
        )
        
        results.append(record)
    return results

def process_oda_geometry(pending_entry):
    stats = pending_entry.get('oda_stats', {})
    meta = pending_entry.get('metadata', {})
    polygon_net = pending_entry.get('mask_net', [])
    
    # Setup Scaling
    size = float(meta.get('size', 1))
    img = cv2.imread(pending_entry['image_path'])
    imgwidth = img.shape[1] if img is not None else 1
    conversion_factor = size / imgwidth

    # Calculate Geometry
    centroid_str, area_px, perimeter_px = calculate_geometry(polygon_net)
    
    # Real-world conversions
    area_um2 = area_px * (conversion_factor ** 2)
    perimeter_um = perimeter_px * conversion_factor

    record = {
        "image_path": pending_entry['image_path'],
        "date": meta.get('date'),
        "location": meta.get('origin'),
        "mask_net": json.dumps(polygon_net),
        "mask_centroid": centroid_str,
        "mask_imgwidth_ppx": imgwidth,
        "mask_area_ppx2": area_px,
        "mask_perimeter_ppx": perimeter_px,
        "mask_imgwidth_um": size,
        "mask_area_um2": area_um2,
        "mask_perimeter_um": perimeter_um,
        "mask_confidence": pending_entry.get('confidence', 0),
        "ommatidia_count": stats.get('ommatidia_count', 0),
        "avg_diameter_px": stats.get('ommatidial_diameter', 0) / conversion_factor if conversion_factor != 0 else 0,
        "avg_diameter_real": stats.get('ommatidial_diameter', 0),
        "diameter_sd_real": stats.get('ommatidial_diameter_SD', 0)
    }
    
    return record

def _build_pending_entry(filepath, image_origin, image_date, size, scale, img_set, user_account, detections):
    return {
        'polygons': [
            {'points': det['polygon'], 'cls': label_to_cls(det['label']), 'conf': det['confidence']}
            for det in (detections or [])
        ],
        'image_path':   filepath,
        'image_origin': image_origin,
        'image_date':   image_date,
        'size':         size,
        'scale':        scale,
        'img_set':      img_set,
        'user_account': user_account
    }


def draw_scale_bar(img, um_per_px, bar_um=100):
    """Draws a 100µm scale bar in the bottom-left corner."""
    h, w = img.shape[:2]
    bar_px = int(bar_um / um_per_px)
    x0, y0 = 20, h - 30
    cv2.line(img, (x0, y0), (x0 + bar_px, y0), (255, 255, 255), 3)
    cv2.putText(img, f"{bar_um} um", (x0, y0 - 8),
    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    

def _render_and_save(img, pending, detections_for_db):
    h, w = img.shape[:2]
    imgsize = max(h, w)
    if imgsize < 500:
        fontscale = 0.4
        thickness = 1
    elif imgsize < 1000:
        fontscale = 0.7
        thickness = 2
    elif imgsize < 2000:
        fontscale = 1.0
        thickness = 2
    else:
        fontscale = 1.5
        thickness = 3

    um_per_px = read_tif_scale_um_per_px(pending['image_path'])

    for poly in detections_for_db:
        pts  = np.array(poly['polygon'], dtype=np.int32)
        cls  = label_to_cls(poly['label'])
        conf = float(poly['confidence'])
        col  = CLS_COLOURS.get(cls, CLS_COLOURS[4])

        overlay = img.copy()
        cv2.fillPoly(overlay, [pts], col)
        cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)
        cv2.polylines(img, [pts], True, col, 2)

        #Label at the topmost point of the polygon
        top = pts[pts[:, 1].argmin()]
        tx, ty = int(top[0]), int(top[1])
        labtext = poly['label'].replace('_', ' ').title() + ' ' + str(round(conf, 2))
        (tw, th), _ = cv2.getTextSize(labtext, cv2.FONT_HERSHEY_SIMPLEX, fontscale, thickness)
        cv2.rectangle(img, (tx - 4, ty - th - 8), (tx + tw + 4, ty), col, -1)
        cv2.putText(img, labtext, (tx, ty - 4), cv2.FONT_HERSHEY_SIMPLEX, fontscale, (0, 0, 0), thickness)

        if um_per_px:
            pts_f = pts.astype(np.float32)
            x_min = int(pts_f[:, 0].min());  x_max = int(pts_f[:, 0].max())
            y_min = int(pts_f[:, 1].min());  y_max = int(pts_f[:, 1].max())
            bbox_w_um = (x_max - x_min) * um_per_px
            bbox_h_um = (y_max - y_min) * um_per_px

            # Horizontal dimension line (below bounding box)
            mid_y = y_max + 16
            cv2.arrowedLine(img, (x_min, mid_y), (x_max, mid_y), (255, 255, 0), 1, tipLength=0.1)
            cv2.putText(img, f"{bbox_w_um:.1f}um", (x_min, mid_y + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, fontscale * 0.7, (255, 255, 0), thickness)

            # Vertical dimension line (right of bounding box)
            mid_x = x_max + 16
            cv2.arrowedLine(img, (mid_x, y_min), (mid_x, y_max), (255, 255, 0), 1, tipLength=0.1)
            cv2.putText(img, f"{bbox_h_um:.1f}um", (mid_x + 4, (y_min + y_max) // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, fontscale * 0.7, (255, 255, 0), thickness)

    # Draw scale bar once (bottom-left corner)
    if um_per_px:
        draw_scale_bar(img, um_per_px)

    base     = os.path.splitext(os.path.basename(pending['image_path']))[0]
    ann_name = 'annotated_' + base + '.jpg'
    ann_path = os.path.join('static/uploads', ann_name)
    cv2.imwrite(ann_path, img)
    return ann_path, w

def write_to_db(pending, detections_for_db, ann_path, img_w):
    conn    = sqlite3.connect('flytest.db')
    cur     = conn.cursor()
    user    = pending['user_account']
    id_row  = cur.execute('SELECT id FROM users WHERE email = ?', (user,)).fetchone()
    user_id = str(id_row[0]) if id_row else None
    cur.execute(
        'INSERT INTO image_results (image_path, annotated_image, image_origin, image_date, result, created_by_id) VALUES (?, ?, ?, ?, ?, ?)',
        (pending['image_path'], ann_path, pending['image_origin'], pending['image_date'],
         json.dumps(detections_for_db), user_id)
    )
    image_id  = cur.execute('SELECT last_insert_rowid()').fetchone()[0]
    processed = process_detections(
        pending['image_origin'], pending['image_date'],
        detections_for_db, image_id, pending['image_path'],
        pending['size'], pending['scale'], img_w, pending['img_set']
    )
    cur.executemany("""
        INSERT INTO object_results (
    image_id, 
    image_path, 
    date, 
    location, 
    object_id, 
    object_type, 
    polygon_net, 
    centroid, 
    imgwidth_ppx, 
    area_ppx2, 
    perimeter_ppx, 
    imgwidth_um, 
    area_um2, 
    perimeter_um, 
    confidence,
    bbox_width_px,
    bbox_height_px,
    major_axis_um,
    minor_axis_um,
    bbox_width_um,
    bbox_height_um,
    scale_source
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
    """, processed)
    conn.commit()
    conn.close()

def next_or_results():
    pending, queue, queue_total = load_queue_data()
    if queue:
        return redirect(url_for('next_image'))
    clear_queue_data()
    return redirect(url_for('resultsf'))

def _write_to_oda_db(pending, detections_for_db, ann_path, img_w):
    conn    = sqlite3.connect('flytest.db')
    cur     = conn.cursor()
    user    = pending['user_account']
    id_row  = cur.execute('SELECT id FROM users WHERE email = ?', (user,)).fetchone()
    user_id = str(id_row[0]) if id_row else None
    cur.execute(
        'INSERT INTO ODA_root (image_path, annotated_image, image_origin, image_date, result, created_by_id) VALUES (?, ?, ?, ?, ?, ?)',
        (pending['image_path'], ann_path, pending['image_origin'], pending['image_date'],
         json.dumps(detections_for_db), user_id)
    )
    image_id  = cur.execute('SELECT last_insert_rowid()').fetchone()[0]
    processed = process_detections(
        pending['image_origin'], pending['image_date'],
        detections_for_db, image_id, pending['image_path'],
        pending['size'], pending['scale'], img_w, pending['img_set']
    )
    cur.executemany("""
        INSERT INTO ODA_object (
            image_id,
            image_path,
            date,
            location,
            mask_net,
            mask_centroid,
            mask_imgwidth_ppx,
            mask_area_ppx2,
            mask_perimeter_ppx,
            mask_imgwidth_um,
            mask_area_um2,
            mask_perimeter_um,
            mask_confidence,
            ommatidia_count,
            avg_diameter_px,
            avg_diameter_real,
            diameter_sd_real,
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, processed)
    conn.commit()
    conn.close()

def draw_scale_bar(img, um_per_px, bar_um=100):
    """Draws a 100µm scale bar in the bottom-left corner."""
    if um_per_px is None:
        return
    h, w = img.shape[:2]
    bar_px = int(bar_um / um_per_px)
    x0, y0 = 20, h - 30
    cv2.line(img, (x0, y0), (x0 + bar_px, y0), (255, 255, 255), 3)
    cv2.putText(img, f"{bar_um} µm", (x0, y0 - 6),
    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"] # Retrieve the uploade image file from the POST request
    img_bytes = file.read()  # Read the raw bytes from the uploaded file
    npimg = np.frombuffer(img_bytes, np.uint8) # Convert the raw bytes into a NumPy arrar for OpenCV Processing
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR) # Decode the NumPy array into an OPenCV RGB image
    results = model(img) # r
    detection = []
    for r in results:
        for box in r.boxes:
            detection.append({
                "label": model.names[int(box.cls)],
                "confidence": float(box.conf),
                "bbox": box.xyxy.tolist()
            })
    return render_template("resultsf.html", detections=detection)

#Route for home page
@app.route("/newflyweb")
def newflyweb():
    return render_template("newflyweb.html")

#Route for single image upload
@app.route("/newflyupload")
def newflyupload():
    countries = sorted([country.name for country in pycountry.countries])
    return render_template("newflyupload.html", countries = countries)

#Route for ODA upload
@app.route("/ODA")
def ODA():
    countries = sorted([country.name for country in pycountry.countries])
    return render_template("ODA.html", countries = countries)

#Plots detected ommatidia positions over the original image, coloured by nearest-neighbour distance. The right-hand sidebar combines a colour gradient with a rotated histogram to show the distribution of diameters
def generate_spatial_heatmap(img_path, coords, point_dists, title, output_fn):
    img_bgr = cv2.imread(img_path)
    if img_bgr is None or len(coords) == 0:
        return

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    x_coords = coords[:, 1]
    y_coords = coords[:, 0]

    cmap = plt.get_cmap('viridis')
    fig, ax = plt.subplots(figsize=(10, 8), facecolor='white')

    ax.imshow(img_rgb)

    ax.scatter(x_coords, y_coords, c=point_dists, cmap=cmap,
                         s=12, edgecolors='none', alpha=0.7)

    ax.set_title(title, fontsize=12, pad=15)
    ax.axis('off')

    cax_hist = fig.add_axes([0.78, 0.15, 0.05, 0.7])
    gradient = np.linspace(0, 1, 256).reshape(256, 1)
    min_d, max_d = np.nanmin(point_dists), np.nanmax(point_dists)

    cax_hist.imshow(gradient, aspect='auto', cmap=cmap, origin='lower',
                    extent=[0, 1, min_d, max_d])

    counts, bins = np.histogram(point_dists, bins=30)
    counts_norm = counts / (np.max(counts) * 1.1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    cax_hist.plot(counts_norm, bin_centers, color='white', lw=1.5)

    cax_hist.set_ylabel(f'Ommatidial Diameter (N={len(point_dists)})', rotation=270, labelpad=20)
    cax_hist.yaxis.set_label_position('right')
    cax_hist.set_xticks([])

    plt.savefig(output_fn, dpi=200, bbox_inches='tight')
    plt.close(fig)

def ODA_run(img_path, mask_path, bright_peak, high_pass, plot_fn):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if img is None or mask is None:
        return None

    # Pre-processing
    blurred = cv2.GaussianBlur(img, (51, 51), 0)
    if high_pass:
        processed = cv2.subtract(img, blurred)
        processed = cv2.normalize(processed, None, 0, 255, cv2.NORM_MINMAX)
    else:
        processed = img

    # Masking
    _, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    eye_area = np.sum(mask_bin > 0)
    masked_img = cv2.bitwise_and(processed, processed, mask=mask_bin)

    # Peak Detection
    working_img = cv2.bitwise_not(masked_img) if not bright_peak else masked_img
    coords = peak_local_max(working_img, min_distance=10, threshold_rel=0.2)
    count = len(coords)

    # Diameter estimation
    nearest_dists = np.array([])
    avg_diam, std_diam = 0, 0

    if count > 1:
        dist_matrix = distance.cdist(coords, coords)
        np.fill_diagonal(dist_matrix, np.inf)
        nearest_dists = np.min(dist_matrix, axis=1)
        avg_diam = np.mean(nearest_dists)
        std_diam = np.std(nearest_dists)
    if plot_fn and count > 1:
        title_str = f"Count: {count} | Avg Dist: {avg_diam:.2f}px"
        generate_spatial_heatmap(img_path, coords, nearest_dists, title_str, plot_fn)

    return {
        "ommatidia_count": count,
        "eye_area": eye_area,
        "ommatidial_diameter": avg_diam,
        "ommatidial_diameter_SD": std_diam
    }

@app.route("/ODA_run", methods=["GET","POST"])
def ODA_route():
    if request.method == "POST":
        app.logger.info("ODA Batch Processing Started")
        
        # Setup directories
        BASE_DIR = "static/ODA"
        UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
        MASK_DIR = os.path.join(BASE_DIR, "masks")
        OUTPUT_DIR = os.path.join(BASE_DIR, "results")
        for d in [UPLOAD_DIR, MASK_DIR, OUTPUT_DIR]:
            os.makedirs(d, exist_ok=True)

        # Get data from form
        files = request.files.getlist("image")
        user_account = str(session.get("user"))
        image_origin = request.form.get("image_origin")
        image_date = request.form.get("image_date")
        size = float(request.form.get("image_size", 1))
        scale = request.form.get("image_scale")
        img_set = request.form.get("type_of_image")

        valid_files = [f for f in files if f and f.filename != ""]
        if not valid_files:
            return render_template("ODA.html", message="No files selected.", countries=sorted([c.name for c in pycountry.countries]))

        queue_entries = []
        first_stats = None

        for f in valid_files:
            # 1. Save File
            img_fn = f.filename
            img_path = os.path.join(UPLOAD_DIR, img_fn)
            f.save(img_path)

            # 2. Generate Mask via Model
            img_cv = cv2.imread(img_path)
            results = oda_model(img_cv)
            
            mask_fn = os.path.splitext(img_fn)[0] + ".png"
            mask_path = os.path.join(MASK_DIR, mask_fn)
            
            mask_found = False
            for result in results:
                if result.masks is not None:
                    raw_polygon = result.masks.xy[0].tolist()
                    confidence_score = float(result.boxes.conf[0].item())
                    mask = result.masks.data[0].cpu().numpy()
                    mask = cv2.resize(mask, (result.orig_shape[1], result.orig_shape[0]))
                    binary_mask = (mask * 255).astype(np.uint8)
                    cv2.imwrite(mask_path, binary_mask)
                    mask_found = True
            
            if not mask_found:
                app.logger.warning(f"No eye detected in {img_fn}")
                continue

            # 3. Run ODA Analysis
            heatmap_fn = f"{os.path.splitext(img_fn)[0]}_heatmap.png"
            plot_output_path = os.path.join(OUTPUT_DIR, heatmap_fn)
            
            # ... inside the valid_files loop in ODA_route ...
            stats = ODA_run(img_path, mask_path, True, True, plot_output_path)

            if stats:
                # 4. Apply Scaling and FIX NumPy Types
                img_w = img_cv.shape[1]
                conversion_factor = size / img_w
                
                # Use .item() to convert numpy types to native Python types
                # Or use int()/float() to be safe
                processed_stats = {
                    "ommatidia_count": int(stats["ommatidia_count"]),
                    "eye_area": int(stats["eye_area"]),
                    "ommatidial_diameter": float(stats["ommatidial_diameter"] * conversion_factor),
                    "ommatidial_diameter_SD": float(stats["ommatidial_diameter_SD"]),
                    "filename": img_fn,
                    "heatmap_fn": heatmap_fn
                }
                
                # 5. Build queue entry using processed_stats
                entry = {
                    "image_path": img_path,
                    "mask_path": mask_path,
                    "confidence": confidence_score,
                    "mask_net": raw_polygon,
                    "oda_stats": processed_stats,
                    "metadata": {
                        "origin": image_origin,
                        "date": image_date,
                        "scale": scale,
                        "user": user_account,
                        "size": size,           
                        "type_of_image": img_set
                    }
                }
                queue_entries.append(entry)
                
                if first_stats is None:
                    first_stats = processed_stats

        if not queue_entries:
            return render_template("ODA.html", message="Analysis failed for all files.", countries=sorted([c.name for c in pycountry.countries]))
        
         # 6. Stash queue data
        store_queue_data(
            pending=queue_entries[0],
            queue=queue_entries[1:],
            queue_total=len(queue_entries)
        )

        return render_template("ODA.html", 
                            oda_stats=first_stats, 
                            queue_index=0, 
                            queue_total=len(queue_entries),
                            countries=sorted([c.name for c in pycountry.countries]))
    pending, queue, queue_total = load_queue_data()
    if pending:
        # Calculate current index based on how many are left in queue
        current_index = queue_total - len(queue) - 1
        
        return render_template("ODA.html", 
                               oda_stats=pending['oda_stats'], 
                               queue_index=current_index, 
                               queue_total=queue_total,
                               countries=sorted([c.name for c in pycountry.countries]))

    # Default if no files were uploaded and no queue exists
    return render_template("ODA.html", 
                           countries=sorted([c.name for c in pycountry.countries]))

@app.route("/save_oda_next")
def save_oda_next():
    pending, queue, queue_total = load_queue_data()
    
    if not pending:
        return redirect(url_for('ODA_route'))

    # 1. Process Geometry and Biometrics
    oda_data = process_oda_geometry(pending)
    
    # 2. Database Connection
    conn = sqlite3.connect("flytest.db")
    cur = conn.cursor()
    
    user = session.get("user")
    cur.execute("SELECT id FROM users WHERE email = ?", (user,))
    user_row = cur.fetchone()
    user_id = user_row[0] if user_row else None

    try:
        # A. Insert Root Entry (Metadata)
        cur.execute("""
            INSERT INTO ODA_root (image_path, annotated_image, image_origin, image_date, result, created_by_id)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            oda_data['image_path'],
            f"static/ODA/results/{pending['oda_stats']['heatmap_fn']}",
            oda_data['location'],
            oda_data['date'],
            json.dumps(pending['oda_stats']),
            user_id
        ))
        
        image_id = cur.lastrowid

        # B. Insert Object Entry (Biometrics & Scaling)
        record = oda_data
        # Ensure your database table has these new columns!
        cur.execute("""
        INSERT INTO ODA_object (
            image_id, 
            image_path, 
            date, 
            location, 
            mask_net, 
            mask_centroid, 
            mask_imgwidth_ppx, 
            mask_area_ppx2, 
            mask_perimeter_ppx,
            mask_imgwidth_um, 
            mask_area_um2, 
            mask_perimeter_um, 
            mask_confidence,
            ommatidia_count, 
            avg_diameter_px, 
            avg_diameter_real, 
            diameter_sd_real
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
         """, (
            image_id,
            record['image_path'],
            record['date'],
            record['location'],
            record['mask_net'],
            record['mask_centroid'],
            record['mask_imgwidth_ppx'],
            record['mask_area_ppx2'],
            record['mask_perimeter_ppx'],
            record['mask_imgwidth_um'],
            record['mask_area_um2'],
            record['mask_perimeter_um'],
            record['mask_confidence'],
            record['ommatidia_count'],
            record['avg_diameter_px'],
            record['avg_diameter_real'],
            record['diameter_sd_real']
        ))
        conn.commit()
    except Exception as e:
        app.logger.error(f"ODA Save Error: {e}")
        conn.rollback()
    finally:
        conn.close()

    # 3. Handle Queue Movement
    if queue:
        store_queue_data(queue[0], queue[1:], queue_total)
        return redirect(url_for('ODA_route'))
    else:
        # Clear Session
        key = session.get("queue_key")
        if key:
            path = _session_path(key)
            if os.path.exists(path): os.remove(path)
        session.pop("queue_key", None)
        return redirect(url_for('oda_results'))
    
@app.route("/skip_to_results_oda")
def skip_to_results():
    # 1. Load initial queue state
    pending, queue, queue_total = load_queue_data()
    
    if not pending:
        return redirect(url_for('oda_results'))

    conn = sqlite3.connect("flytest.db")
    cur = conn.cursor()
    
    user = session.get("user")
    cur.execute("SELECT id FROM users WHERE email = ?", (user,))
    user_row = cur.fetchone()
    user_id = user_row[0] if user_row else None

    # 2. Loop until no images are left in the queue
    while pending:
        try:
            oda_data = process_oda_geometry(pending)
            
            # Insert Root Entry
            cur.execute("""
                INSERT INTO ODA_root (image_path, annotated_image, image_origin, image_date, result, created_by_id)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                oda_data['image_path'],
                f"static/ODA/results/{pending['oda_stats']['heatmap_fn']}",
                oda_data['location'],
                oda_data['date'],
                json.dumps(pending['oda_stats']),
                user_id
            ))
            
            image_id = cur.lastrowid

            # Insert Object Entry
            cur.execute("""
            INSERT INTO ODA_object (
                image_id, image_path, date, location, mask_net, mask_centroid, 
                mask_imgwidth_ppx, mask_area_ppx2, mask_perimeter_ppx,
                mask_imgwidth_um, mask_area_um2, mask_perimeter_um, 
                mask_confidence, ommatidia_count, avg_diameter_px, 
                avg_diameter_real, diameter_sd_real
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                image_id, oda_data['image_path'], oda_data['date'], oda_data['location'],
                oda_data['mask_net'], oda_data['mask_centroid'], oda_data['mask_imgwidth_ppx'],
                oda_data['mask_area_ppx2'], oda_data['mask_perimeter_ppx'], oda_data['mask_imgwidth_um'],
                oda_data['mask_area_um2'], oda_data['mask_perimeter_um'], oda_data['mask_confidence'],
                oda_data['ommatidia_count'], oda_data['avg_diameter_px'], oda_data['avg_diameter_real'],
                oda_data['diameter_sd_real']
            ))
            
            conn.commit()
        except Exception as e:
            app.logger.error(f"ODA Skip Process Error: {e}")
            conn.rollback()

        # Move to the next item in the queue
        if queue:
            pending = queue[0]
            queue = queue[1:]
        else:
            pending = None
            queue = []

    # 3. Cleanup and Final Redirect
    conn.close()
    
    # Clear Session/Queue Files
    key = session.get("queue_key")
    if key:
        path = _session_path(key)
        if os.path.exists(path): os.remove(path)
    session.pop("queue_key", None)
    
    return redirect(url_for('oda_results'))

#Renders the processing page shown to the user whilst the model is running
@app.route("/fprocess")
def fprocess():
    return render_template("fprocess.html")
    
#Route for results page
@app.route("/resultsf")
def resultsf():
    conn = sqlite3.connect("flytest.db")
    cur = conn.cursor()

    user = session.get("user")
    if user:
        cur.execute("SELECT id FROM users WHERE email = ?", (user,))
        row = cur.fetchone()
        user_id = str(row[0]) if row else None
        cur.execute("""
            SELECT ir.id, ir.image_path, ir.annotated_image, ir.image_origin, 
                   ir.image_date, ir.result, ir.public, u.name
            FROM image_results ir
            LEFT JOIN users u ON u.id = CAST(ir.created_by_id AS INTEGER)
            WHERE ir.created_by_id = ? OR ir.public = 1
            ORDER BY ir.id DESC
        """, (user_id,))
    else:
        cur.execute("""
            SELECT ir.id, ir.image_path, ir.annotated_image, ir.image_origin, 
                   ir.image_date, ir.result, ir.public, u.name
            FROM image_results ir
            LEFT JOIN users u ON u.id = CAST(ir.created_by_id AS INTEGER)
            WHERE ir.public = 1
            ORDER BY ir.id DESC
        """)

    rows = cur.fetchall()
    conn.close()

    images = []
    for row in rows:
        id, image_path, annotated_image, image_origin, image_date, result, public, owner_name = row
        detections = json.loads(result)
        fly_count = len(detections)
        avg_conf = sum(d["confidence"] for d in detections) / fly_count if fly_count > 0 else 0
        summary = {
            "fly_count": fly_count,
            "confidence": round(avg_conf * 100, 2),
            "public": public,
            "owner": owner_name or "Unknown"
        }
        images.append((id, image_path, annotated_image, image_origin, image_date, summary))
    countries = sorted([country.name for country in pycountry.countries])
    return render_template("resultsf.html", images=images, countries=countries)

@app.route("/oda_results")
def oda_results():
    conn = sqlite3.connect("flytest.db")
    # This allows us to access columns by name: row['ommatidia_count']
    conn.row_factory = sqlite3.Row 
    cur = conn.cursor()

    user = session.get("user")
    user_id = None
    
    if user:
        cur.execute("SELECT id FROM users WHERE email = ?", (user,))
        user_row = cur.fetchone()
        user_id = str(user_row['id']) if user_row else None

    # Base Query: Joins ODA_root with the ODA_object data and the user name
    # We use a LEFT JOIN on ODA_object assuming 1 object result per root image
    query = """
        SELECT 
            r.id, r.image_path, r.annotated_image, r.image_origin, r.image_date, 
            r.public, u.name as owner_name,
            o.ommatidia_count, o.avg_diameter_real, o.diameter_sd_real, 
            o.mask_confidence, o.mask_area_um2
        FROM ODA_root r
        LEFT JOIN users u ON u.id = r.created_by_id
        LEFT JOIN ODA_object o ON r.id = o.image_id
    """

    if user_id:
        query += " WHERE r.created_by_id = ? OR r.public = 1"
        cur.execute(query + " ORDER BY r.id DESC", (user_id,))
    else:
        query += " WHERE r.public = 1"
        cur.execute(query + " ORDER BY r.id DESC")

    rows = cur.fetchall()
    conn.close()

    oda_list = []
    for row in rows:
        # Construct a summary dictionary for the template
        summary = {
            "count": row["ommatidia_count"] or 0,
            "diameter": round(row["avg_diameter_real"], 2) if row["avg_diameter_real"] else 0,
            "sd": round(row["diameter_sd_real"], 3) if row["diameter_sd_real"] else 0,
            "confidence": round((row["mask_confidence"] or 0) * 100, 1),
            "area": round(row["mask_area_um2"], 2) if row["mask_area_um2"] else 0,
            "owner": row["owner_name"] or "Unknown",
            "public": row["public"]
        }
        
        oda_list.append({
            "id": row["id"],
            "image_path": row["image_path"],
            "annotated_image": row["annotated_image"],
            "origin": row["image_origin"],
            "date": row["image_date"],
            "stats": summary,
            "public": row["public"]
        })

    countries = sorted([country.name for country in pycountry.countries])
    return render_template("oda_results.html", results=oda_list, countries=countries)

#Route for fly welfare page
@app.route("/flywelfare")
def flywelfare():
    return render_template("flywelfare.html")

#Route for login page
@app.route("/flogin")
def flogin():
    return render_template("flogin.html")

#Route for registration page
@app.route("/fregister")
def fregister():
    return render_template("fregister.html")

#Registration process
@app.route("/register", methods=["GET", "POST"])
def register():
    message = None
    if request.method == "POST":
        name = request.form["name"]
        email = request.form["email"]
        password = request.form["password"]

        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_pattern, email):
            return render_template("fregister.html", message="Invalid email address.")

        password_errors = []
        if len(password) < 8:
            password_errors.append("Password must be at least 8 characters.")
        if not re.search(r'[A-Z]', password):
            password_errors.append("Password must contain at least one uppercase letter.")
        if not re.search(r'[a-z]', password):
            password_errors.append("Password must contain at least one lowercase letter.")
        if not re.search(r'[0-9]', password):
            password_errors.append("Password must contain at least one number.")
        if not re.search(r'[!@#$%^&*(),.?\":{}|<>]', password):
            password_errors.append("Password must contain at least one special character.")

        if password_errors:
            return render_template("fregister.html", message=" ".join(password_errors))

        hashed_pw = hashlib.sha256(password.encode()).hexdigest()
        conn = sqlite3.connect("flytest.db")
        cur = conn.cursor()
        try:
            cur.execute(
                "INSERT INTO users (name, email, password) VALUES (?, ?, ?)",
                (name, email, hashed_pw)
            )
            conn.commit()
            session["user"] = email
            return redirect(url_for("newflyweb"))
        except sqlite3.IntegrityError:
            message = "Name or email already exists."
        finally:
            conn.close()
    return render_template("fregister.html", message=message)

#Login process
@app.route("/login", methods=["GET", "POST"])
def login():
    message = None
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]
        hashed_pw = hashlib.sha256(password.encode()).hexdigest()
        conn = sqlite3.connect("flytest.db")
        cur = conn.cursor()
        cur.execute(
            "SELECT * FROM users WHERE email = ? AND password = ?",
            (email, hashed_pw)
        )
        user = cur.fetchone()
        conn.close()
        if user:
            session["user"] = email
            return redirect(url_for("newflyweb"))
        else:
            message = "Invalid email or password!"
    return render_template("flogin.html", message=message)

#Logout process
@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("newflyweb"))


#Uploading an image for analyse process
@app.route("/upload", methods=["GET", "POST"])
def upload():
    message = None
    detections = None
    annotated_image = None

    if request.method == "POST":
        files = request.files.getlist("filename")
        user_account = session.get("user")
        user_account = str(user_account)
        app.logger.info(user_account)
        image_origin = request.form.get("image_origin")
        image_date = request.form.get("image_date")
        size = request.form.get("image_size")
        scale = request.form.get("image_scale")
        img_set = request.form.get("type_of_image")
        date_check = datetime.strptime(image_date, '%Y-%m-%d').date()
        today = datetime.today().date()
        if date_check > today:
            image_date = today.strftime('%Y-%m-%d')
            app.logger.info(f"Date was in future, reset to: {image_date}")

        valid_files = [f for f in files if f and f.filename != ""]
        if not valid_files:
            message = "No file selected."
            return render_template("newflyupload.html", message=message, detections=None, annotated_image=None,
                                   countries=sorted([c.name for c in pycountry.countries]))

        queue_entries = []
        first_detections = None
        first_annotated = None

        for f in valid_files:
            filepath = os.path.join("static/uploads", f.filename)
            f.save(filepath)
            if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
                return "File not found or empty", 400
            file_detections, file_annotated, error = polygon_analyse_eyes(filepath)
            if error:
                app.logger.warning(f"Analysis error for {filepath}: {error}")
                continue
            entry = _build_pending_entry(filepath, image_origin, image_date, size, scale, img_set, user_account, file_detections)
            queue_entries.append(entry)
            if first_detections is None:
                first_detections = file_detections
                first_annotated = file_annotated

        if not queue_entries:
            message = "All uploaded files failed analysis."
            return render_template("newflyupload.html", message=message, detections=None, annotated_image=None,
                                   countries=sorted([c.name for c in pycountry.countries]))

        #Stash everything on disk so the polygon editor can use it
        #Database write happens after editing in /save_detections
        store_queue_data(
            pending     = queue_entries[0],
            queue       = queue_entries[1:],
            queue_total = len(queue_entries)
        )

        detections = first_detections
        annotated_image = first_annotated
        message = "Analysis complete."

    _, _, queue_total = load_queue_data()
    pending, _, _ = load_queue_data()
    current_filename = (os.path.basename(pending['image_path']) if detections and pending else "")

    return render_template("newflyupload.html", message=message, detections=detections, annotated_image=annotated_image,
                           countries=sorted([country.name for country in pycountry.countries]),
                           queue_index=0, queue_total=queue_total or 1, current_filename=current_filename)

@app.route("/next_image")
def next_image():
    pending, queue, queue_total = load_queue_data()
    if not queue:
        clear_queue_data()
        return redirect(url_for('resultsf'))

    next_item = queue.pop(0)
    filepath  = next_item['image_path']

    detections_raw, annotated_image, error = polygon_analyse_eyes(filepath)
    if error or detections_raw is None:
        app.logger.warning(f"Skipping {filepath}: {error}")
        update_queue_data(pending, queue)
        return redirect(url_for('next_image'))

    next_item['polygons'] = [
        {'points': det['polygon'], 'cls': label_to_cls(det['label']), 'conf': det['confidence']}
        for det in detections_raw
    ]
    update_queue_data(next_item, queue)

    queue_index      = queue_total - 1 - len(queue)
    current_filename = os.path.basename(filepath)

    return render_template(
        "newflyupload.html",
        message="Analysis complete.",
        detections=detections_raw,
        annotated_image=annotated_image,
        countries=sorted([c.name for c in pycountry.countries]),
        queue_index=queue_index,
        queue_total=queue_total,
        current_filename=current_filename
    )

#Route for comparrion page
@app.route("/compare")
def compare():

    id1 = request.args.get("id1")
    id2 = request.args.get("id2")

    if not id1 or not id2:
        return redirect(url_for("resultsf"))

    conn = sqlite3.connect("flytest.db")
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute("SELECT annotated_image, image_origin, image_date, result FROM image_results WHERE id = ?", (id1,))
    image1 = cur.fetchone() #Fetching image 1 details

    cur.execute("SELECT annotated_image, image_origin, image_date, result FROM image_results WHERE id = ?", (id2,))
    image2 = cur.fetchone() #Fetching image 2 details

    conn.close()
    #Fetching individual details from fetched details
    detections1 = json.loads(image1['result'])
    fly_count1 = len(detections1)
    avg_conf_1 = sum(d["confidence"] for d in detections1) / fly_count1 * 100
    stats1 = {"Parts_Detected":fly_count1,"Average_Confidence":avg_conf_1}
    detections2 = json.loads(image2['result'])
    fly_count2 = len(detections2)
    avg_conf_2 = sum(d["confidence"] for d in detections2) / fly_count2 * 100
    stats2 = {"Parts_Detected":fly_count2,"Average_Confidence":avg_conf_2}

    if not image1 or not image2:
        return redirect(url_for("resultsf"))

    return render_template("comparef.html", image1=image1, image2=image2,stats1=stats1,stats2=stats2)

@app.route("/compare_oda")
def compare_oda():
    #Works similary to compare on fly eyes
    id1 = request.args.get("id1")
    id2 = request.args.get("id2")

    if not id1 or not id2:
        return redirect(url_for("oda_results"))

    conn = sqlite3.connect("flytest.db")
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # Query to get both the image metadata and the calculated object stats
    query = """
        SELECT 
            r.annotated_image, r.image_origin, r.image_date,
            o.ommatidia_count, o.avg_diameter_real, o.diameter_sd_real, o.mask_area_um2
        FROM ODA_root r
        LEFT JOIN ODA_object o ON r.id = o.image_id
        WHERE r.id = ?
    """

    cur.execute(query, (id1,))
    image1 = cur.fetchone()

    cur.execute(query, (id2,))
    image2 = cur.fetchone()
    conn.close()

    if not image1 or not image2:
        return redirect(url_for("oda_results"))

    # Map the columns directly to your 'data' structure
    data1 = {
        "stats": {
            "count": image1["ommatidia_count"] or 0,
            "diameter": image1["avg_diameter_real"] or 0,
            "sd": image1["diameter_sd_real"] or 0,
            "area": image1["mask_area_um2"] or 0
        }
    }

    data2 = {
        "stats": {
            "count": image2["ommatidia_count"] or 0,
            "diameter": image2["avg_diameter_real"] or 0,
            "sd": image2["diameter_sd_real"] or 0,
            "area": image2["mask_area_um2"] or 0
        }
    }

    return render_template("compare_oda.html", image1=image1, image2=image2, data1=data1, data2=data2)


@app.route("/timespan", methods=['GET', 'POST'])
def timespan():
    #Get the data for get/post
    all_countries = sorted([country.name for country in pycountry.countries])
    all_parts = ["left_eye","right_eye","eye_top"]
    ids_param = request.args.get('ids')
    rows = []

    conn = sqlite3.connect("flytest.db")
    cur = conn.cursor()
    #If coming from filter
    if ids_param:
        id_list = ids_param.split(',')
        placeholders = ', '.join(['?'] * len(id_list))
        query = f"""
            SELECT image_id, object_type, date, area_um2
            FROM object_results 
            WHERE object_type != 'antennae' 
            AND image_id IN ({placeholders})
            ORDER BY date ASC
        """
        cur.execute(query, id_list)
        rows = cur.fetchall()
    elif request.method == 'POST':
        #If comming from timespan itself, get dates, parts and countries for filtering
        start_str = request.form.get('date_earliest')
        end_str = request.form.get('date_latest')
        if start_str == '':
            start_str = "0001-01-01"
        if end_str == '':
            end_str = "9999-12-31"
        countries = request.form.get('selected_countries')
        if countries:
            try:
                countries = json.loads(countries)
            except json.JSONDecodeError:
                app.logger.error("JSON ERROR")
                countries = []
        else:
            countries = []
        parts = request.form.get('selected_parts')
        if parts:
            try:
                parts = json.loads(parts)
            except json.JSONDecodeError:
                app.logger.error("JSON ERROR")
                parts = all_parts
        else:
            parts = all_parts
        start_date = datetime.strptime(start_str, '%Y-%m-%d').date()
        end_date = datetime.strptime(end_str, '%Y-%m-%d').date()
        
        placeholders2 = f"({', '.join(['?'] * len(parts))})"
        #If no countries, add all countries
        if len(countries) == 0:
            app.logger.info(placeholders2)
            app.logger.info(parts[0])
            query = f"""
                SELECT image_id, object_type, date, area_um2 
                FROM object_results 
                WHERE object_type IN {placeholders2} 
                AND date BETWEEN ? AND ? 
                ORDER BY date ASC
                """
            params = (*parts,start_date, end_date)
            cur.execute(query, params)
        #If has one country or more on filter, only use them
        else:
            placeholders = ', '.join(['?'] * len(countries))
            app.logger.info(placeholders)
            app.logger.info(countries[0])
            query = f"""
                SELECT image_id, object_type, date, area_um2
                FROM object_results 
                WHERE object_type IN {placeholders2} 
                AND date BETWEEN ? AND ? 
                AND location IN ({placeholders})
                ORDER BY date ASC
                """
            params = (*parts,start_date, end_date, *countries)
            cur.execute(query, params)
    
        rows = cur.fetchall()
        app.logger.info(f"Query Results: {rows}")
        conn.close()

    datasets = {}
    all_unique_dates = sorted(list(set(row[2] for row in rows)))
    #Show the objects on the timespan page scatter graph
    for row in rows:
        obj_type = row[1] 
        if obj_type not in datasets:
            datasets[obj_type] = []
        
        datasets[obj_type].append({'x': row[2], 'y': row[3]})
    app.logger.info("results created succesfully")
    return render_template('timespan.html', datasets=datasets, all_dates=all_unique_dates,countries=all_countries,parts=all_parts)
    

@app.route("/export", methods=['GET', 'POST'])
def export():
    #Works similarly to timespan, with similar setup on data filtering
    all_countries = sorted([country.name for country in pycountry.countries])
    all_parts = ["left_eye","right_eye","eye_top"]
    ids_param = request.args.get('ids')
    df = pd.DataFrame()

    conn = sqlite3.connect("flytest.db")
    if ids_param:
        id_list = ids_param.split(',')
        placeholders = ', '.join(['?'] * len(id_list))
        query = f"SELECT * FROM object_results WHERE image_id IN ({placeholders})"
        df = pd.read_sql_query(query, conn, params=id_list)
        conn.close()
        
        buffer = io.BytesIO()
        df.to_csv(buffer, index=False, encoding='utf-8')
        buffer.seek(0)

        return send_file(
            buffer,
            as_attachment=True,
            download_name='EyeObjectResults.csv',
            mimetype='text/csv'
        )
    
    elif request.method == 'POST':
        start_str = request.form.get('date_earliest')
        end_str = request.form.get('date_latest')
        if start_str == '':
            start_str = "0001-01-01"
        if end_str == '':
            end_str = "9999-12-31"
        countries = request.form.get('selected_countries')
        if countries:
            try:
                countries = json.loads(countries)
            except json.JSONDecodeError:
                app.logger.error("JSON ERROR")
                countries = [] 
        else:
            countries = []
        parts = request.form.get('selected_parts')
        if parts:
            try:
                parts = json.loads(parts)
            except json.JSONDecodeError:
                app.logger.error("JSON ERROR")
                parts = all_parts
        else:
            parts = all_parts
        start_date = datetime.strptime(start_str, '%Y-%m-%d').date()
        end_date = datetime.strptime(end_str, '%Y-%m-%d').date()
        
        placeholders2 = f"({', '.join(['?'] * len(parts))})"

        if len(countries) == 0:
            app.logger.info(placeholders2)
            app.logger.info(parts[0])
            query = f"""
                SELECT * 
                FROM object_results 
                WHERE object_type IN {placeholders2} 
                AND date BETWEEN ? AND ? 
                ORDER BY date ASC
                """
            input_params = (*parts,start_date, end_date)
            df = pd.read_sql_query(query,conn,params=input_params)
        
        else:
            placeholders = ', '.join(['?'] * len(countries))
            app.logger.info(placeholders)
            app.logger.info(countries[0])
            query = f"""
                SELECT *
                FROM object_results 
                WHERE object_type IN {placeholders2} 
                AND date BETWEEN ? AND ? 
                AND location IN ({placeholders})
                ORDER BY date ASC
                """
            input_params = (*parts,start_date, end_date, *countries)
            df = pd.read_sql_query(query,conn,params=input_params)
    
        conn.close()
        #Convert the df into a csv and export it as a send file
        buffer = io.BytesIO()
        df.to_csv(buffer, index=False, encoding='utf-8')
        buffer.seek(0)
    
        return send_file(
            buffer,
            as_attachment=True,
            download_name='EyeObjectResults.csv',
            mimetype='text/csv'
        )
    return render_template("export.html",countries=all_countries,parts=all_parts)

@app.route("/export_oda", methods=['GET', 'POST'])
def export_oda():
    #Same as export heads, except using the oda database
    app.logger.info("ODA")
    all_countries = sorted([country.name for country in pycountry.countries])
    all_parts = ["left_eye","right_eye","eye_top"]
    ids_param = request.args.get('ids')
    df = pd.DataFrame()

    conn = sqlite3.connect("flytest.db")
    if ids_param:
        id_list = ids_param.split(',')
        placeholders = ', '.join(['?'] * len(id_list))
        query = f"SELECT * FROM ODA_object WHERE image_id IN ({placeholders})"
        df = pd.read_sql_query(query, conn, params=id_list)
        conn.close()
        
        buffer = io.BytesIO()
        df.to_csv(buffer, index=False, encoding='utf-8')
        buffer.seek(0)

        return send_file(
            buffer,
            as_attachment=True,
            download_name='ODAResults.csv',
            mimetype='text/csv'
        )
    
    elif request.method == 'POST':
        start_str = request.form.get('date_earliest')
        end_str = request.form.get('date_latest')
        if start_str == '':
            start_str = "0001-01-01"
        if end_str == '':
            end_str = "9999-12-31"
        countries = request.form.get('selected_countries')
        if countries:
            try:
                countries = json.loads(countries)
            except json.JSONDecodeError:
                app.logger.error("JSON ERROR")
                countries = [] 
        else:
            countries = []
        parts = request.form.get('selected_parts')
        if parts:
            try:
                parts = json.loads(parts)
            except json.JSONDecodeError:
                app.logger.error("JSON ERROR")
                parts = all_parts
        else:
            parts = all_parts
        start_date = datetime.strptime(start_str, '%Y-%m-%d').date()
        end_date = datetime.strptime(end_str, '%Y-%m-%d').date()
        
        placeholders2 = f"({', '.join(['?'] * len(parts))})"

        if len(countries) == 0:
            app.logger.info(placeholders2)
            app.logger.info(parts[0])
            query = f"""
                SELECT * 
                FROM ODA_object
                WHERE date BETWEEN ? AND ? 
                ORDER BY date ASC
                """
            input_params = (start_date, end_date)
            df = pd.read_sql_query(query,conn,params=input_params)
        
        else:
            placeholders = ', '.join(['?'] * len(countries))
            app.logger.info(placeholders)
            app.logger.info(countries[0])
            query = f"""
                SELECT *
                FROM ODA_object
                WHERE date BETWEEN ? AND ? 
                AND location IN ({placeholders})
                ORDER BY date ASC
                """
            input_params = (start_date, end_date, *countries)
            df = pd.read_sql_query(query,conn,params=input_params)
    
        conn.close()
        buffer = io.BytesIO()
        df.to_csv(buffer, index=False, encoding='utf-8')
        buffer.seek(0)

        return send_file(
            buffer,
            as_attachment=True,
            download_name='ODAResults.csv',
            mimetype='text/csv'
        )
    return render_template("export.html",countries=all_countries,parts=all_parts)

@app.route("/")
def index():
    return redirect(url_for("newflyweb"))

"""
@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404
"""

@app.route("/delete_results", methods=["POST"])
def delete_results():
    ids = request.form.get("ids", "")
    user = session.get("user")
    if ids and user:
        conn = sqlite3.connect("flytest.db")
        cur = conn.cursor()
        cur.execute("SELECT id FROM users WHERE email = ?", (user,))
        row = cur.fetchone()
        user_id = str(row[0]) if row else None
        id_list = ids.split(",")
        placeholders = ", ".join(["?"] * len(id_list))
        cur.execute(f"DELETE FROM image_results WHERE id IN ({placeholders}) AND created_by_id = ?", (*id_list, user_id))
        cur.execute(f"DELETE FROM object_results WHERE image_id IN ({placeholders}) AND image_id IN (SELECT id FROM image_results WHERE created_by_id = ?)", (*id_list, user_id))
        conn.commit()
        conn.close()
    return redirect(url_for("resultsf"))

@app.route("/delete_results_oda", methods=["POST"])
def delete_results_oda():
    ids = request.form.get("ids", "")
    user = session.get("user")
    if ids and user:
        conn = sqlite3.connect("flytest.db")
        cur = conn.cursor()
        cur.execute("SELECT id FROM users WHERE email = ?", (user,))
        row = cur.fetchone()
        user_id = str(row[0]) if row else None
        id_list = ids.split(",")
        placeholders = ", ".join(["?"] * len(id_list))
        cur.execute(f"DELETE FROM ODA_root WHERE id IN ({placeholders}) AND created_by_id = ?", (*id_list, user_id))
        cur.execute(f"DELETE FROM ODA_object WHERE image_id IN ({placeholders}) AND image_id IN (SELECT id FROM ODA_root WHERE created_by_id = ?)", (*id_list, user_id))
        conn.commit()
        conn.close()
    return redirect(url_for("oda_results"))


@app.route("/make_public", methods=["POST"])
def make_public():
    ids = request.form.get("ids", "")
    user = session.get("user")
    if ids and user:
        conn = sqlite3.connect("flytest.db")
        cur = conn.cursor()
        cur.execute("SELECT id FROM users WHERE email = ?", (user,))
        row = cur.fetchone()
        user_id = str(row[0]) if row else None
        id_list = ids.split(",")
        placeholders = ", ".join(["?"] * len(id_list))
        cur.execute(f"UPDATE image_results SET public = 1 WHERE id IN ({placeholders}) AND created_by_id = ?", (*id_list, user_id))
        conn.commit()
        conn.close()
    return redirect(url_for("resultsf"))

@app.route("/make_private", methods=["POST"])
def make_private():
    ids = request.form.get("ids", "")
    user = session.get("user")
    if ids and user:
        conn = sqlite3.connect("flytest.db")
        cur = conn.cursor()
        cur.execute("SELECT id FROM users WHERE email = ?", (user,))
        row = cur.fetchone()
        user_id = str(row[0]) if row else None
        id_list = ids.split(",")
        placeholders = ", ".join(["?"] * len(id_list))
        cur.execute(f"UPDATE image_results SET public = 0 WHERE id IN ({placeholders}) AND created_by_id = ?", (*id_list, user_id))
        conn.commit()
        conn.close()
    return redirect(url_for("resultsf"))

@app.route("/make_oda_public", methods=["POST"])
def make_oda_public():
    app.logger.info("Public")
    ids = request.form.get("ids", "")
    user = session.get("user")
    if ids and user:
        conn = sqlite3.connect("flytest.db")
        cur = conn.cursor()
        # Get the user's ID from the users table
        cur.execute("SELECT id FROM users WHERE email = ?", (user,))
        row = cur.fetchone()
        user_id = row[0] if row else None
        
        if user_id:
            id_list = ids.split(",")
            placeholders = ", ".join(["?"] * len(id_list))
            # Target the ODA_root table specifically
            cur.execute(f"UPDATE ODA_root SET public = 1 WHERE id IN ({placeholders}) AND created_by_id = ?", (*id_list, user_id))
            conn.commit()
        conn.close()
    return redirect(url_for("oda_results")) 

@app.route("/make_oda_private", methods=["POST"])
def make_oda_private():
    ids = request.form.get("ids", "")
    user = session.get("user")
    if ids and user:
        conn = sqlite3.connect("flytest.db")
        cur = conn.cursor()
        cur.execute("SELECT id FROM users WHERE email = ?", (user,))
        row = cur.fetchone()
        user_id = row[0] if row else None
        
        if user_id:
            id_list = ids.split(",")
            placeholders = ", ".join(["?"] * len(id_list))
            cur.execute(f"UPDATE ODA_root SET public = 0 WHERE id IN ({placeholders}) AND created_by_id = ?", (*id_list, user_id))
            conn.commit()
        conn.close()
    return redirect(url_for("oda_results")) 

"""
@app.errorhandler(Exception)
def handle_unexpected_error(e):
    app.logger.error(f"Server Error: {e}")
    return render_template("generic_error.html"), 500
"""

#Polygon editor routes
#Maps model label strings to class indices
def label_to_cls(label):
    mapping = {
        'left_eye':  0,
        'right_eye': 1,
        'eye_top':   2,
        'antennae':  3
    }
    return mapping.get(label.lower().replace(' ', '_'), 4)

CLS_COLOURS = {
    0: (50,  205, 50),
    1: (0,   255, 255),
    2: (80,  100,  255),
    3: (0,   165, 255),
    4: (255, 0,   255)
}

#Class 0 is left eye from the fly's perspective, appears on the right when viewed head-on. I switched the label names later on to fix this
CLS_NAMES = {
    0: 'right_eye',
    1: 'left_eye',
    2: 'eye_top',
    3: 'antennae',
    4: 'other'
}

@app.route("/polygon_editor")
def polygon_editor():
    pending, _, _ = load_queue_data()
    app.logger.info(pending)
    if not pending:
        return redirect(url_for('newflyupload'))
    
    # If the key itself is the ID:
    return render_template("polygon_editor.html", data=pending)

@app.route("/get_detections")
def get_detections():
    pending, _, _ = load_queue_data()
    if not pending:
        return jsonify({'error': 'No pending detections - please re-upload your image.'})
    filepath = pending['image_path']

    #Uses /get_image to convert tiff to jpeg for the browser
    return jsonify({
        'filename':  os.path.basename(filepath),
        'image_url': '/get_image',
        'image_id':  None,
        'polygons':  pending['polygons']
    })

@app.route("/get_image")
def get_image():
    #Reads the pending image as jpeg
    pending, _, _ = load_queue_data()
    if not pending:
        return 'No image', 404
    filepath = pending['image_path']
    img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
    if img is None:
        return 'Could not read image', 404
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 1:
        img = cv2.cvtColor(img.squeeze(), cv2.COLOR_GRAY2BGR)
    ok, buf = cv2.imencode('.jpg', img)
    if not ok:
        return 'Encode failed', 500
    return send_file(io.BytesIO(buf.tobytes()), mimetype='image/jpeg')


@app.route("/save_detections", methods=["POST"])
def save_detections():
    pending, queue, _ = load_queue_data()
    if not pending:
        return jsonify({'success': False, 'error': 'Session expired, please re-upload.'})
    data = request.get_json()
    if not data:
        return jsonify({'success': False, 'error': 'No data received.'})
    edited   = data.get('polygons', [])
    try:
        img = cv2.imread(pending['image_path'])
        if img is None:
            return jsonify({'success': False, 'error': 'Could not read image.'})
        if len(img.shape) == 2 or img.shape[2] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        detections_for_db = [
            {'label': CLS_NAMES.get(int(p['cls']), 'other'), 'confidence': float(p['conf']), 'polygon': p['points']}
            for p in edited
        ]
        ann_path, img_w = _render_and_save(img, pending, detections_for_db)
        write_to_db_and_track(pending, detections_for_db, ann_path, img_w)

        _, _, queue_total = load_queue_data()
        store_queue_data(pending=None, queue=queue, queue_total=queue_total)

        if queue:
            next_route = 'next_image_folder' if session.get('is_folder_batch') else 'next_image'
            return jsonify({'success': True, 'redirect': url_for(next_route)})
        clear_queue_data()
        if session.pop('is_folder_batch', False):
            session.modified = True
            return jsonify({'success': True, 'redirect': url_for('folderR')})
        return jsonify({'success': True, 'redirect': url_for('resultsf')})
    except Exception as e:
        app.logger.error(f'save_detections failed: {e}', exc_info=True)
        return jsonify({'success': False, 'error': str(e)})


@app.route("/save_as_is")
def save_as_is():
    #User clicked skip on the editor modal, so it saves original detections as is
    pending, queue, _ = load_queue_data()
    if not pending:
        return redirect(url_for('resultsf'))
    try:
        filepath = pending['image_path']
        img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
        if img is None:
            return next_or_results()
        if img.dtype != np.uint8:
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 1:
            img = cv2.cvtColor(img.squeeze(), cv2.COLOR_GRAY2BGR)

        detections_for_db = [
            {'label': CLS_NAMES.get(int(p['cls']), 'other'), 'confidence': float(p['conf']), 'polygon': p['points']}
            for p in pending['polygons']
        ]
        ann_path, img_w = _render_and_save(img, pending, detections_for_db)
        write_to_db_and_track(pending, detections_for_db, ann_path, img_w)
    except Exception as e:
        app.logger.error(f'save_as_is failed: {e}', exc_info=True)

    return next_or_folder_results()

#Saves every remaining image in the queue as is. It Loops through the full queue in a single request and writes each result directly to the database, then redirects to one of the results pages
@app.route("/skip_all_eye_detections")
def skip_all_eye_detections():
    pending, queue, queue_total = load_queue_data()
    
    if not pending:
        return redirect(url_for('resultsf'))

    while pending:
        try:
            filepath = pending['image_path']
            img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
            
            if img is not None:
                # Standardize image format
                if img.dtype != np.uint8:
                    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                if len(img.shape) == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                elif img.shape[2] == 1:
                    img = cv2.cvtColor(img.squeeze(), cv2.COLOR_GRAY2BGR)

                # Use the existing polygons from the pending object (the "as-is" data)
                detections_for_db = [
                    {
                        'label': CLS_NAMES.get(int(p['cls']), 'other'), 
                        'confidence': float(p['conf']), 
                        'polygon': p['points']
                    }
                    for p in pending.get('polygons', [])
                ]

                # Render heatmap/annotated image and write to SQLite
                ann_path, img_w = _render_and_save(img, pending, detections_for_db)
                write_to_db_and_track(pending, detections_for_db, ann_path, img_w)

        except Exception as e:
            app.logger.error(f'Skip All Error for {pending.get("image_path")}: {e}')

        # Advance the queue
        if queue:
            pending = queue[0]
            queue = queue[1:]
        else:
            pending = None
            queue = []

    # Final Cleanup
    clear_queue_data()
    
    # Logic to determine where to go next (Batch vs Single)
    if session.pop('is_folder_batch', False):
        session.modified = True
        return redirect(url_for('folderR'))
    
    return redirect(url_for('resultsf'))

def store_folder_image_ids(ids):
    key = session.get("queue_key") or str(uuid.uuid4())
    path = _session_path(key + "-folder_ids")
    with open(path, "w") as fh:
        json.dump(ids, fh)
    session["folder_ids_key"] = key
    session.modified = True

def load_folder_image_ids():
    key = session.get("folder_ids_key")
    if not key:
        return []
    path = _session_path(key + "-folder_ids")
    if not os.path.exists(path):
        return []
    with open(path) as fh:
        return json.load(fh)

def append_folder_image_id(image_id):
    ids = load_folder_image_ids()
    ids.append(image_id)
    key = session.get("folder_ids_key") or str(uuid.uuid4())
    path = _session_path(key + "-folder_ids")
    with open(path, "w") as fh:
        json.dump(ids, fh)
    session["folder_ids_key"] = key
    session.modified = True

def clear_folder_image_ids():
    key = session.pop("folder_ids_key", None)
    session.modified = True
    if key:
        path = _session_path(key + "-folder_ids")
        if os.path.exists(path):
            os.remove(path)

def write_to_db_and_track(pending, detections_for_db, ann_path, img_w):
    conn = sqlite3.connect('flytest.db')
    cur = conn.cursor()
    user = pending['user_account']
    id_row = cur.execute('SELECT id FROM users WHERE email = ?', (user,)).fetchone()
    user_id = str(id_row[0]) if id_row else None
    cur.execute(
        'INSERT INTO image_results (image_path, annotated_image, image_origin, image_date, result, created_by_id) VALUES (?, ?, ?, ?, ?, ?)',
        (pending['image_path'], ann_path, pending['image_origin'], pending['image_date'],
         json.dumps(detections_for_db), user_id)
    )
    image_id = cur.execute('SELECT last_insert_rowid()').fetchone()[0]
    processed = process_detections(
        pending['image_origin'], pending['image_date'],
        detections_for_db, image_id, pending['image_path'],
        pending['size'], pending['scale'], img_w, pending['img_set']
    )
    cur.executemany("""
        INSERT INTO object_results (
    image_id, 
    image_path, 
    date, 
    location, 
    object_id, 
    object_type, 
    polygon_net, 
    centroid, 
    imgwidth_ppx, 
    area_ppx2, 
    perimeter_ppx, 
    imgwidth_um, 
    area_um2, 
    perimeter_um, 
    confidence,
    bbox_width_px,
    bbox_height_px,
    major_axis_um,
    minor_axis_um,
    bbox_width_um,
    bbox_height_um,
    scale_source
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
    """, processed)
    conn.commit()
    conn.close()
    if session.get("is_folder_batch"):
        append_folder_image_id(image_id)
    return image_id

def next_or_folder_results():
    pending, queue, _ = load_queue_data()
    if queue:
        return redirect(url_for('next_image_folder'))
    clear_queue_data()
    if session.pop("is_folder_batch", False):
        session.modified = True
        return redirect(url_for('folderR'))
    return redirect(url_for('resultsf'))

#Function to upload folder of images
@app.route("/fuploadfolder", methods=["GET", "POST"])
def fuploadfolder():
    message = None
    countries = sorted([country.name for country in pycountry.countries])

    if request.method == "POST":
        files = request.files.getlist("files")
        image_origin = request.form.get("image_origin")
        image_date = request.form.get("image_date")
        size = request.form.get("image_size")
        scale = request.form.get("image_scale")
        img_set = request.form.get("type_of_image")
        user_account = str(session.get("user"))

        try:
            date_check = datetime.strptime(image_date, '%Y-%m-%d').date()
            today = datetime.today().date()
            if date_check > today:
                image_date = today.strftime('%Y-%m-%d')
        except (ValueError, TypeError):
            pass

        valid_files = [f for f in files if f and f.filename != ""]
        if not valid_files:
            message = "No files selected."
            return render_template("fuploadfolder.html", message=message, countries=countries)

        queue_entries = []
        for f in valid_files:
            filename = os.path.basename(f.filename)
            filepath = os.path.join("static/uploads", filename)
            f.save(filepath)
            if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
                app.logger.warning(f"Empty or missing file: {filepath}")
                continue
            file_detections, _, error = polygon_analyse_eyes(filepath)
            if error:
                app.logger.warning(f"Analysis error for {filepath}: {error}")
                continue
            entry = _build_pending_entry(filepath, image_origin, image_date, size, scale, img_set, user_account, file_detections)
            queue_entries.append(entry)

        if not queue_entries:
            message = "All uploaded files failed analysis."
            return render_template("fuploadfolder.html", message=message, countries=countries)

        store_queue_data(
            pending=queue_entries[0],
            queue=queue_entries[1:],
            queue_total=len(queue_entries)
        )
        store_folder_image_ids([])
        session["is_folder_batch"] = True
        session.modified = True

        return redirect(url_for("polygon_editor"))

    return render_template("fuploadfolder.html", message=message, countries=countries)

@app.route("/next_image_folder")
def next_image_folder():
    pending, queue, queue_total = load_queue_data()
    if not queue:
        clear_queue_data()
        if session.pop("is_folder_batch", False):
            session.modified = True
            return redirect(url_for('folderR'))
        return redirect(url_for('resultsf'))

    next_item = queue.pop(0)
    filepath = next_item['image_path']

    detections_raw, annotated_image, error = polygon_analyse_eyes(filepath)
    if error or detections_raw is None:
        app.logger.warning(f"Skipping {filepath}: {error}")
        update_queue_data(pending, queue)
        return redirect(url_for('next_image_folder'))

    next_item['polygons'] = [
        {'points': det['polygon'], 'cls': label_to_cls(det['label']), 'conf': det['confidence']}
        for det in detections_raw
    ]
    update_queue_data(next_item, queue)

    queue_index = queue_total - 1 - len(queue)
    current_filename = os.path.basename(filepath)

    return render_template(
        "newflyupload.html",
        message="Analysis complete.",
        detections=detections_raw,
        annotated_image=annotated_image,
        countries=sorted([c.name for c in pycountry.countries]),
        queue_index=queue_index,
        queue_total=queue_total,
        current_filename=current_filename
    )

@app.route("/odafolder", methods=["GET", "POST"])
def fuploadfolder_oda():
    #Works similary to uploading a folder
    countries = sorted([c.name for c in pycountry.countries])
    
    if request.method == "POST":
        # Setup directories as per your ODA logic
        BASE_DIR = "static/ODA"
        UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
        MASK_DIR = os.path.join(BASE_DIR, "masks")
        OUTPUT_DIR = os.path.join(BASE_DIR, "results")
        for d in [UPLOAD_DIR, MASK_DIR, OUTPUT_DIR]:
            os.makedirs(d, exist_ok=True)

        files = request.files.getlist("files") # Note: 'files' matches the HTML name
        user_account = str(session.get("user"))
        image_origin = request.form.get("image_origin")
        image_date = request.form.get("image_date")
        size = float(request.form.get("image_size", 1))
        scale = request.form.get("image_scale")
        img_set = request.form.get("type_of_image")

        valid_files = [f for f in files if f and f.filename != ""]
        if not valid_files:
            return render_template("odafolder.html", message="No files selected.", countries=countries)

        queue_entries = []

        for f in valid_files:
            # 1. Save File (handling relative paths from webkitdirectory)
            img_fn = os.path.basename(f.filename) 
            img_path = os.path.join(UPLOAD_DIR, img_fn)
            f.save(img_path)

            # 2. Generate Mask via Model
            img_cv = cv2.imread(img_path)
            if img_cv is None: continue
            
            results = oda_model(img_cv)
            mask_fn = os.path.splitext(img_fn)[0] + ".png"
            mask_path = os.path.join(MASK_DIR, mask_fn)
            
            mask_found = False
            confidence_score = 0
            raw_polygon = []

            for result in results:
                if result.masks is not None:
                    raw_polygon = result.masks.xy[0].tolist()
                    confidence_score = float(result.boxes.conf[0].item())
                    mask = result.masks.data[0].cpu().numpy()
                    mask = cv2.resize(mask, (result.orig_shape[1], result.orig_shape[0]))
                    cv2.imwrite(mask_path, (mask * 255).astype(np.uint8))
                    mask_found = True
            
            if not mask_found: continue

            # 3. Run Analysis
            heatmap_fn = f"{os.path.splitext(img_fn)[0]}_heatmap.png"
            plot_path = os.path.join(OUTPUT_DIR, heatmap_fn)
            stats = ODA_run(img_path, mask_path, True, True, plot_path)

            if stats:
                # Apply Scaling
                img_w = img_cv.shape[1]
                conversion_factor = size / img_w
                
                processed_stats = {
                    "ommatidia_count": int(stats["ommatidia_count"]),
                    "eye_area": int(stats["eye_area"]),
                    "ommatidial_diameter": float(stats["ommatidial_diameter"] * conversion_factor),
                    "ommatidial_diameter_SD": float(stats["ommatidial_diameter_SD"]),
                    "filename": img_fn,
                    "heatmap_fn": heatmap_fn
                }
                
                queue_entries.append({
                    "image_path": img_path,
                    "mask_path": mask_path,
                    "confidence": confidence_score,
                    "mask_net": raw_polygon,
                    "oda_stats": processed_stats,
                    "metadata": {
                        "origin": image_origin, "date": image_date, 
                        "scale": scale, "user": user_account, 
                        "size": size, "type_of_image": img_set
                    }
                })

        if not queue_entries:
            return render_template("odafolder.html", message="No valid eyes found.", countries=countries)

        # Stash in queue
        store_queue_data(pending=queue_entries[0], queue=queue_entries[1:], queue_total=len(queue_entries))

        # Redirect to the iterative viewer
        return redirect(url_for('save_oda_next'))

    return render_template("odafolder.html", countries=countries)

@app.route("/save_oda_fnext")
def save_oda_folder_next():
    pending, queue, queue_total = load_queue_data()
    
    if pending:
        # Save pending data to your DB here
        # db.save_oda_result(pending)
        pass

    if not queue:
        clear_queue_data()
        return redirect(url_for('oda_results'))

    # Move to next
    next_item = queue.pop(0)
    update_queue_data(next_item, queue)
    
    current_index = queue_total - len(queue) - 1

    return render_template("odafolder.html", 
                           oda_stats=next_item['oda_stats'], 
                           queue_index=current_index, 
                           queue_total=queue_total,
                           countries=sorted([c.name for c in pycountry.countries]))

@app.route("/folderR")
def folderR():
    ids = load_folder_image_ids()
    if not ids:
        return redirect(url_for('resultsf'))

    conn = sqlite3.connect("flytest.db")
    cur = conn.cursor()
    placeholders = ", ".join(["?"] * len(ids))
    cur.execute(f"""
        SELECT ir.id, ir.image_path, ir.annotated_image, ir.image_origin,
               ir.image_date, ir.result, ir.public, u.name
        FROM image_results ir
        LEFT JOIN users u ON u.id = CAST(ir.created_by_id AS INTEGER)
        WHERE ir.id IN ({placeholders})
        ORDER BY ir.id ASC
    """, ids)
    rows = cur.fetchall()
    conn.close()

    images = []
    for row in rows:
        id_, image_path, annotated_image, image_origin, image_date, result, public, owner_name = row
        detections = json.loads(result)
        images.append({
            "id": id_,
            "image_path": image_path,
            "annotated_image": annotated_image,
            "image_name": os.path.basename(image_path),
            "image_origin": image_origin,
            "image_date": image_date,
            "detections": detections,
            "public": public,
            "owner": owner_name or "Unknown"
        })

    return render_template("folderR.html", images=images)

#Functionality to export selected data
@app.route("/export_selected", methods=["POST"])
def export_selected():
    data = request.get_json()
    ids = [int(i) for i in data.get("image_ids", [])]

    if not ids:
        return jsonify({"error": "No IDs provided"}), 400

    conn = sqlite3.connect("flytest.db")
    cur = conn.cursor()
    placeholders = ", ".join(["?"] * len(ids))
    cur.execute(
        f"SELECT id, annotated_image, image_origin, image_date, result FROM image_results WHERE id IN ({placeholders})",
        ids
    )
    rows = cur.fetchall()
    conn.close()

    memory_file = BytesIO()
    with zipfile.ZipFile(memory_file, "w") as zf:
        export_lines = []
        for id_, annotated_image, image_origin, image_date, result in rows:
            image_name = annotated_image or ""
            if image_name and os.path.exists(image_name):
                zf.write(image_name, arcname=f"images/{os.path.basename(image_name)}")
            export_lines.append({
                "image_id": id_,
                "image": os.path.basename(image_name),
                "origin": image_origin,
                "date": image_date,
                "detections": json.loads(result)
            })
        zf.writestr("data.json", json.dumps(export_lines, indent=2))

    memory_file.seek(0)
    return send_file(memory_file, download_name="export.zip", as_attachment=True)

@app.route("/folderR_oda")
def folderR_oda():
    # Retrieve the list of ODA record IDs generated during the folder upload
    ids = load_folder_image_ids() 
    if not ids:
        return redirect(url_for('oda_results'))

    conn = sqlite3.connect("flytest.db")
    cur = conn.cursor()
    placeholders = ", ".join(["?"] * len(ids))
    
    # Querying the ODA results table specifically
    cur.execute(f"""
        SELECT id, filename, heatmap_path, image_origin, image_date, 
               ommatidia_count, eye_area, diameter, scale, created_by
        FROM oda_results 
        WHERE id IN ({placeholders})
        ORDER BY id ASC
    """, ids)
    rows = cur.fetchall()
    conn.close()

    images = []
    for row in rows:
        id_, filename, heatmap, origin, date, count, area, dia, scale, user = row
        images.append({
            "id": id_,
            "oda_stats": {
                "filename": filename,
                "heatmap_fn": os.path.basename(heatmap),
                "ommatidia_count": count,
                "eye_area": area,
                "ommatidial_diameter": dia
            },
            "metadata": {
                "origin": origin,
                "date": date,
                "scale": scale,
                "user": user
            }
        })

    return render_template("folderR_oda.html", images=images)

@app.route("/export_selected_oda", methods=["POST"])
def export_selected_oda():
    data = request.get_json()
    ids = [int(i) for i in data.get("image_ids", [])]

    if not ids:
        return jsonify({"error": "No IDs provided"}), 400

    conn = sqlite3.connect("flytest.db")
    cur = conn.cursor()
    placeholders = ", ".join(["?"] * len(ids))
    
    # Fetching ODA data
    cur.execute(f"""
        SELECT id, filename, heatmap_path, image_origin, image_date, 
               ommatidia_count, diameter, scale 
        FROM oda_results 
        WHERE id IN ({placeholders})
    """, ids)
    rows = cur.fetchall()
    conn.close()

    memory_file = BytesIO()
    with zipfile.ZipFile(memory_file, "w") as zf:
        export_data = []
        for row in rows:
            id_, filename, heatmap_path, origin, date, count, dia, scale = row
            
            # Add Heatmap Image to ZIP
            if heatmap_path and os.path.exists(heatmap_path):
                zf.write(heatmap_path, arcname=f"heatmaps/{os.path.basename(heatmap_path)}")
            
            # Add entry to JSON data
            export_data.append({
                "image_id": id_,
                "original_filename": filename,
                "origin": origin,
                "date": date,
                "ommatidia_count": count,
                "average_diameter": dia,
                "unit": scale
            })
            
        # Add the compiled data as a JSON file
        zf.writestr("oda_analysis_results.json", json.dumps(export_data, indent=2))

    memory_file.seek(0)
    return send_file(
        memory_file, 
        download_name="ODA_Export_Batch.zip", 
        as_attachment=True
    )

if __name__ == "__main__":
    app.run(debug=True, port=5000) #Set debug to False for finalisation
