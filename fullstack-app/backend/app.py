# backend/app.py
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from typing import List
import os

app = FastAPI()

# Autoriser CORS pour le frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080"],  # ton frontend Vite
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Charger le modèle YOLO
model = YOLO("best.pt")  # ton modèle local

@app.post("/analyze")
async def analyze(image: UploadFile = File(...)):
    # Sauvegarder l'image temporairement
    temp_path = f"temp_{image.filename}"
    with open(temp_path, "wb") as f:
        f.write(await image.read())

    # Faire la détection
    results = model(temp_path)[0]  # prends le premier résultat

    # Extraire les points (centres des boxes)
    landmarks: List[dict] = []
    if hasattr(results, "boxes") and results.boxes is not None:
        for box in results.boxes.xyxy:  # xyxy = [x1, y1, x2, y2]
            x_center = float((box[0] + box[2]) / 2)
            y_center = float((box[1] + box[3]) / 2)
            landmarks.append({"x": x_center, "y": y_center})

    # Supprimer le fichier temporaire
    os.remove(temp_path)

    return JSONResponse(content={"landmarks": landmarks})
