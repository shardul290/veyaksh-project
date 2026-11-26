# GStreamer setup for video processing (Windows)

The video pipeline runs YOLO from the project's `.venv` and launches a small GStreamer/PyGObject decoder worker from MSYS2 UCRT64 Python. The two Python environments stay separate. The project's normal PyPI OpenCV wheel does not need a GStreamer backend.

## 1. Install MSYS2

If `C:\msys64\ucrt64\bin\python.exe` is missing, install [MSYS2](https://www.msys2.org/) using the default `C:\msys64` location. Open **MSYS2 UCRT64** from the Start menu; do not use the plain MSYS2 shell for these packages.

## 2. Update MSYS2 and install GStreamer

In the **MSYS2 UCRT64** terminal, run:

```bash
pacman -Suy
```

If the update closes the terminal, open **MSYS2 UCRT64** again and run `pacman -Suy` until it reports that the environment is up to date. MSYS2 uses rolling updates and recommends full upgrades. [MSYS2 update guide](https://www.msys2.org/docs/updating/)

Then install Python, PyGObject, GStreamer, the video plugins, and the FFmpeg decoder plugin:

```bash
pacman -S \
  mingw-w64-ucrt-x86_64-python \
  mingw-w64-ucrt-x86_64-python-gobject \
  mingw-w64-ucrt-x86_64-gstreamer \
  mingw-w64-ucrt-x86_64-gst-python \
  mingw-w64-ucrt-x86_64-gst-plugins-base \
  mingw-w64-ucrt-x86_64-gst-plugins-good \
  mingw-w64-ucrt-x86_64-gst-plugins-bad \
  mingw-w64-ucrt-x86_64-gst-libav
```

MSYS2's UCRT64 environment uses packages with the `mingw-w64-ucrt-x86_64-` prefix. [MSYS2 environments and package names](https://www.msys2.org/docs/environments/), [PyGObject package](https://packages.msys2.org/packages/mingw-w64-ucrt-x86_64-python-gobject), [GStreamer packages](https://packages.msys2.org/packages/mingw-w64-ucrt-x86_64-gstreamer)

## 3. Confirm the UCRT64 Python can load GStreamer

Still in **MSYS2 UCRT64**, run:

```bash
python -c 'import gi; gi.require_version("Gst", "1.0"); gi.require_version("GstVideo", "1.0"); from gi.repository import Gst, GstVideo; Gst.init(None); print("GStreamer:", Gst.version_string(), "GstVideo:", GstVideo.VideoInfo.__name__)'
```

It should print the GStreamer version and `GstVideo: VideoInfo`. Find the matching Windows interpreter path:

```bash
cygpath -w "$(command -v python)"
```

The default path is `C:\msys64\ucrt64\bin\python.exe`. Keep this UCRT64 installation separate from any MSVC GStreamer runtime; the processing command uses the UCRT64 Python and plugin libraries.

## 4. Run a 1.1 pilot from VS Code

Open a PowerShell terminal in VS Code at the project root. Set `$gstPython` to the Windows path printed above, then run:

```powershell
$gstPython = "C:\msys64\ucrt64\bin\python.exe"
Test-Path $gstPython
& $gstPython -c "import gi; gi.require_version('Gst','1.0'); gi.require_version('GstVideo','1.0'); from gi.repository import Gst, GstVideo; Gst.init(None); print(Gst.version_string(), GstVideo.VideoInfo.__name__)"
```

The first command should print `True`; the second should print the GStreamer version and `VideoInfo`. Then activate the project environment and run the comparison preview:

```powershell
.\.venv\Scripts\Activate.ps1
python -m m3_events.run_video_pipeline `
  --video data/incoming/1.1.mp4 `
  --model models/best.pt `
  --zones configs/zones/C01.json `
  --output-dir runs/1.1_gstreamer `
  --annotated-output runs/1.1_gstreamer/1.1_annotated_compact.mp4 `
  --device cpu --imgsz 640 --conf 0.15 --stride 1 `
  --decoder gstreamer --gstreamer-python $gstPython
```

The worker adds the UCRT64 DLL, typelib, and plugin directories based on `$gstPython`. It decodes and streams every frame with its timestamp; the model still runs on the project's `.venv`. The output settings keep 1.1's full frame sampling and compact overlay. Compare the generated preview and the elapsed time printed by the exporter before processing the remaining clips.

GStreamer can accelerate decoding only when a compatible hardware decoder plugin and device are available. `uridecodebin` selects from the installed decoders, so hardware acceleration is not guaranteed. YOLO inference remains on the device selected by `--device`; GStreamer will not move CPU inference onto a GPU. [GStreamer `uridecodebin`](https://gstreamer.freedesktop.org/documentation/playback/uridecodebin.html)
