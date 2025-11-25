import asyncio
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from simulation import Simulation
from models import Config, StateResponse

DATA_DIR = Path(__file__).parent / "data"

app = FastAPI()
sim = Simulation(DATA_DIR)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount(
    "/static",
    StaticFiles(directory=DATA_DIR),
    name="static",
)


@app.get("/api/state", response_model=StateResponse)
def get_state():

    return sim.state()


@app.get("/api/config", response_model=Config)
def get_config():
    return sim.config


@app.post("/api/config", response_model=Config)
def update_config(cfg: Config):
    sim.set_config(cfg)
    return sim.config

@app.on_event("startup")
async def start_simulation_loop():
    async def loop():
        while True:
            sim.step()
            await asyncio.sleep(0.2)

    asyncio.create_task(loop())
