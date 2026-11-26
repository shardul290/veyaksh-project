# Quick start

- Open the project folder in VS Code and use a PowerShell terminal.
- Create the environment: `py -3.11 -m venv .venv`
- Install dependencies: `.\.venv\Scripts\python.exe -m pip install -r requirements.txt`
- Use `models/yolo26s_traffic.onnx` for detection. The trained checkpoint is `models/yolo26s_traffic.pt`.
- For GStreamer RTSP, follow [GSTREAMER_SETUP.md](GSTREAMER_SETUP.md). Set `$env:GST_PYTHON = "C:\msys64\ucrt64\bin\python.exe"` if needed.
- Start the backend: `.\.venv\Scripts\python.exe -m uvicorn m6_backend.main:app --host 127.0.0.1 --port 8000`
- Open another terminal: `Start-Process "http://127.0.0.1:8000/"`
- Choose a local video path or RTSP URL, select matching camera zones, and click **Start analysis**.
- Keep the playback tab open. The first run of a local video prepares a cached playback copy.
- Review saved events in **Evidence trail** or **Event history**.
- Click **Stop** to end analysis. Press **Ctrl+C** in the backend terminal to stop the server.
- Videos, extracted frames, datasets, generated runs, and local databases are excluded from Git.
