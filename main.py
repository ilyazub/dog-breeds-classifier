from pathlib import Path
from io import BytesIO

from responder import API
from aiohttp import ClientSession

from fastai.vision import (
  load_learner,
  defaults,
  open_image
)
from torch import device

defaults.device = device('cpu')
api = API()

learner = load_learner(path=Path('./'))


async def get_bytes(url):
    async with ClientSession() as session:
        async with session.get(url) as response:
            return await response.read()


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


@api.route("/upload")
async def upload(request, response):
    data = await request.form()
    bytes = await (data["file"].read())
    response.media = predict_image_from_bytes(bytes)


@api.route("/classify-url")
async def classify_url(request, response):
    bytes = await get_bytes(request.params["url"])
    response.media = predict_image_from_bytes(bytes)


@api.route("/")
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
    api.run()
