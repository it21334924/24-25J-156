from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os

from app.exercise_api import exercise_router  
# from app.exercise_api_test import exercise_router

app = FastAPI(title="Eye Health Platform")

# ✅ Enable CORS (adjust origins in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "app", "templates")
app.mount("/templates", StaticFiles(directory=TEMPLATES_DIR), name="templates")

# ✅ Include the exercise router
app.include_router(exercise_router)

# ✅ Run server when executed directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
