import aiohttp
import asyncio
import json
import os
import time

MAX_DOWNLOAD_NUMBER = asyncio.Semaphore(10)
CAMERA_DATA_PATH = f'./camera.json'
SAVE_FOLDER = f'./Snapshot/'


def createFolder(path):
    if not os.path.exists(path):
        os.makedirs(path)


async def fetch_image_by_id(url, img_id):
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(verify_ssl=False)) as session:
        async with session.get(url) as response:
            try:
                image_data = await response.content.read()
                createFolder(f'{SAVE_FOLDER}{img_id}/')
                with open(f'{SAVE_FOLDER}{img_id}/{img_id}.jpg', 'wb') as f:
                    f.write(image_data)
                # print(f'Download complete - {img_id}.jpg')
            except Exception as e:
                print(f'Download failed - {url}', e)


async def fetch_all_images():
    sem = MAX_DOWNLOAD_NUMBER
    try:
        with open(CAMERA_DATA_PATH, 'r') as f:
            cameras = json.load(f)
        print(f'total {len(cameras)} cameras')
        for camera in cameras:
            url = camera['Url']
            camera_id = url.split('/')[-1]
            async with sem:
                await fetch_image_by_id(url, camera_id)
    except Exception as e:
        print('Loading failed', e)


def main():
    start = time.perf_counter()
    event_loop = asyncio.get_event_loop()
    future = asyncio.ensure_future(fetch_all_images())
    event_loop.run_until_complete(future)
    finish = time.perf_counter()
    print(f'All images finished in {round(finish - start,2)} seconds(s)')


if __name__ == '__main__':
    main()
