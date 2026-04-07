from fastapi import FastAPI
from data_cleaning_env.env import DataCleaningEnv, Action

app = FastAPI()
env = DataCleaningEnv()

@app.get("/")
def root():
    return {"status": "ok", "message": "Data Cleaning Environment is running!"}

@app.post("/reset")
def reset():
    obs = env.reset()
    return obs.dict()

@app.post("/step")
def step(action: Action):
    obs, reward, done, info = env.step(action)
    return {
        "observation": obs.dict(),
        "reward": reward.dict(),
        "done": done,
        "info": info
    }

@app.get("/state")
def state():
    return env.state()

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()