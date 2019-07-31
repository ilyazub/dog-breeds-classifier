from pathlib import Path
from io import BytesIO

from responder import API
from aiohttp import ClientSession
import asyncio

from fastai.vision import (
    load_learner,
    defaults,
    open_image
)
from torch import device

defaults.device = device('cpu')
app = API()

export_file_url = 'https://drive.google.com/uc?id=1n8L9ZEaSJgfbQs9Vb9cUrHLHhFVTQiQI&export=download'
export_file_name = 'export.pkl'
path = Path(__file__).parent

async def get_bytes(url):
    async with ClientSession() as session:
        async with session.get(url) as response:
            return await response.read()

async def download_file(url, dest):
    if dest.exists():
        return
    data = await get_bytes(url)
    with open(dest, 'wb') as f:
        f.write(data)

async def setup_learner():
  await download_file(export_file_url, path / export_file_name)
  try:
      learn = load_learner(path, export_file_name)
      return learn
  except RuntimeError as e:
      if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
          print(e)
          message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
          raise RuntimeError(message)
      else:
          raise

loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learner = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()

def predict_image_from_bytes(bytes):
    img = open_image(BytesIO(bytes))
    _, _, losses = learner.predict(img)

    return {
        "predictions": sorted(
            zip(learner.data.classes, map(float, losses)),
            key=lambda p: p[1],
            reverse=True
        )
    }

@app.route("/upload")
async def upload(request, response):
    data = await request.form()
    bytes = await (data["file"].read())
    response.media = predict_image_from_bytes(bytes)


@app.route("/classify-url")
async def classify_url(request, response):
    bytes = await get_bytes(request.params["url"])
    response.media = predict_image_from_bytes(bytes)


@app.route("/")
def form(req, resp):
    resp.html = """
        <form action="/upload" method="post" enctype="multipart/form-data">
            Select image to upload:
            <input type="file" name="file">
            <input type="submit" value="Upload Image">
        </form>
        <p>Or submit a URL:</p>
        <form action="/classify-url" method="get">
            <input type="url" name="url">
            <input type="submit" value="Fetch and analyze image">
        </form>
    """


if __name__ == '__main__':
    app.run()
