from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os

from app.fatigue_api import fatigue_router

app = FastAPI(title="Eye Health Platform")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "app", "static")       # ✅ For CSS, JS, images
TEMPLATES_DIR = os.path.join(BASE_DIR, "app", "templates") # ✅ For HTML files

# Serve static assets at /static/
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# ✅ Include all HTML + API routes from fatigue_router
app.include_router(fatigue_router)

# Server entry point
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
