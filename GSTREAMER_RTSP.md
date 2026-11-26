# GStreamer video-to-RTSP pipeline

This pipeline uses `GstRtspServer` to publish one recorded clip at a time at:

```text
rtsp://localhost:8554/cam1
```

`gst_rtsp_server.py` reads and decodes the selected file through GStreamer `uridecodebin`, scales it while preserving portrait or landscape orientation, paces playback in real time, encodes H.264, and serves RTP over RTSP. If the D3D12 H.264 decoder is installed, the server downloads its frames from D3D12 memory; otherwise it uses a software-compatible conversion path. When a clip reaches its end, the server seeks back to the beginning so clients remain connected. The default output is limited to 1280 pixels on its longest side and 30 FPS to handle the supplied high-resolution clips smoothly.

Use the MSYS2 **UCRT64** terminal so Python, GStreamer, `gst-rtsp-server`, and OpenCV load the matching UCRT64 libraries. From the project folder, start a clip like this:

```bash
cd /c/Users/91639/OneDrive/Desktop/projects/video-ingest-project
python gst_rtsp_server.py assets/rtsp_inputs/dehradun_clip01.mp4
```

The server prints the RTSP endpoint. Stop it with `Ctrl+C`. To serve another clip, pass that clip's path as the positional argument. This endpoint serves the selected clip on a loop; it does not automatically make a playlist from every file.

## Verify the RTSP client and stream

The client check starts and stops its own RTSP server, decodes the RTSP stream using GStreamer, checks that frames continue arriving, and verifies stable frame dimensions. From a second UCRT64 terminal in the project folder, test one clip continuously for 30 seconds:

```bash
python test_gst_rtsp.py --source assets/rtsp_inputs/dehradun_clip01.mp4 --seconds 30
```

To check every file currently in `assets/rtsp_inputs` (a 3-second client check per file):

```bash
python test_gst_rtsp.py --all
```

Do not run the managed client check while a manually started server is already using port 8554. For a separate manual client, leave the server running and open another UCRT64 terminal with:

```bash
gst-play-1.0 rtsp://localhost:8554/cam1
```

## Laptop webcam object-detection demo

The webcam demo publishes the first Windows Media Foundation camera as a local-only RTSP stream, then runs the bundled NanoDet COCO model on that stream. It detects people and common COCO objects. This model does not detect hands or recognize gestures. The demo is preview-only and does not save video. Windows may require desktop-camera access in **Settings → Privacy & security → Camera**.

Stop the 1.1 file server with `Ctrl+C` to free CPU, then start the webcam server in **MSYS2 UCRT64**:

```bash
cd "/c/Users/Kaushik/Desktop/Traffic Detection"
MSYS_NO_PATHCONV=1 python gst_webcam_rtsp_server.py --device-index 0 --host 127.0.0.1 --port 8555 --mount /webcam --width 1280 --height 720 --fps 30
```

Leave it running. It prints `rtsp://127.0.0.1:8555/webcam`; the camera opens when the preview client connects. In a second **PowerShell** terminal, run:

```powershell
cd "C:\Users\Kaushik\Desktop\Traffic Detection"
.\.venv\Scripts\Activate.ps1
python scripts/live_rtsp_object_preview.py --source "rtsp://127.0.0.1:8555/webcam" --model models/object_detection_nanodet_2022nov.onnx --max-width 1280 --conf 0.35 --nms 0.6
```

Press `Q` or `Esc` in the preview to close it, then press `Ctrl+C` in the UCRT64 terminal to stop the RTSP server. If index 0 is not the laptop camera, try `--device-index 1`; `gst-device-monitor-1.0 Video/Source` can list available camera sources.

## Current verification

- All 20 files in `assets/rtsp_inputs` delivered decoded RTSP frames in the per-file check.
- `dehradun_clip02.mp4` (3840x2160, 60 FPS source) streamed continuously for 30 seconds, including a repeat after the source clip ended. The RTSP client received 711 frames at 1280x720 (about 24 FPS on this machine); the server output target is 30 FPS.
- `Muzaffarnagar_clip05` has no filename extension; GStreamer identified and streamed it by its file contents.

See [GSTREAMER_SETUP.md](GSTREAMER_SETUP.md) for the Windows installation and Python environment details. The older [RTSP_SETUP.md](RTSP_SETUP.md) documents the separate MediaMTX plus FFmpeg pipeline.
