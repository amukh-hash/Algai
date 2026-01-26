from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1.endpoints import data, xai, hybrid

app = FastAPI(title="Algo Trading Backend")

# CORS
origins = [
    "http://localhost:3000",
    "http://localhost:3001",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(data.router, prefix="/api/v1/data", tags=["data"])
app.include_router(xai.router, prefix="/api/v1/xai", tags=["xai"])
app.include_router(hybrid.router, prefix="/api/v1/hybrid", tags=["hybrid"])

@app.get("/")
def read_root():
    return {"message": "Welcome to the Algo Trading Backend"}
