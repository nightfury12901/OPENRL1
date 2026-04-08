"""
FastAPI Application for the Legal Case Assistant OpenEnv Environment.
"""
from fastapi import FastAPI
from legal_env.models import LegalAction, LegalObservation, LegalEnvironmentState
from legal_env.server.legal_environment import LegalEnvironment

env = LegalEnvironment()
app = FastAPI(title="Legal Case Assistant", version="1.0.0")


@app.post("/reset", response_model=LegalObservation)
def reset():
    return env.reset()


@app.post("/step", response_model=LegalObservation)
def step(action: LegalAction):
    return env.step(action)


@app.get("/state", response_model=LegalEnvironmentState)
def state():
    return env.state


@app.get("/health")
def health():
    return {"status": "healthy"}
def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)