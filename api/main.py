from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

try:
    from src.grid_model import get_sample_grid_serializable
except Exception:
    # allow running when python path is project root
    from grid_model import get_sample_grid_serializable


app = FastAPI(title="AeroNet Lite API - Phase 1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/grid")
def get_grid():
    """Return the full 10x10 grid as JSON."""
    return get_sample_grid_serializable()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=True)
