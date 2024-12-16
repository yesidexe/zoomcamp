from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
import uvicorn

app = FastAPI()

@app.get("/ping", response_class=PlainTextResponse)
async def ping():
    return "Pong"

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=9696)

# uvicorn app:app --reload --host 127.0.0.1 --port 9696
# curl http://127.0.0.1:9696/app

