# Low-latency RTSP/WebRTC publisher

`rtspclientsink` is an RTSP **client**. It does not create an RTSP server, so
MediaMTX must be running and reachable before the DeepStream pipeline starts.

From the repository root, start both services on one Docker network:

```sh
docker compose up
```

The DeepStream service publishes to `rtsp://mediamtx:8554/deepstream-mosaic`.
Open the result locally at:

```text
http://127.0.0.1:8889/deepstream-mosaic
```

For a browser on another computer, set the Docker host's LAN IP before startup:

```sh
WEBRTC_ADDITIONAL_HOSTS=192.168.1.20 docker compose up
```

Then open `http://192.168.1.20:8889/deepstream-mosaic`.

## Existing DeepStream container

Do not publish to `127.0.0.1` unless MediaMTX is running in that same container.
Connect the existing container and MediaMTX to the same Docker network and use
the MediaMTX container name:

```sh
docker network create deepstream-video
docker run -d --name mediamtx --network deepstream-video \
  -p 8555:8554 -p 8889:8889 -p 8189:8189/udp \
  bluenviron/mediamtx:1.18.2
docker network connect deepstream-video e93387b1d6da
```

Inside the existing DeepStream container, run:

```sh
python3 deepstream_low_latency_rtsp_and_file.py \
  --publish-url rtsp://mediamtx:8554/deepstream-mosaic \
  file:///workspace/video4.mp4
```

The MP4's AAC audio is intentionally not decoded because this publisher emits
a video-only mosaic. This avoids the harmless `No decoder available for
audio/mpeg` warning on minimal DeepStream images.
