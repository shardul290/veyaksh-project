"""Publish the local Windows webcam as a loopback H.264 RTSP stream."""

from __future__ import annotations

import argparse
import logging
import signal

import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstRtspServer", "1.0")
from gi.repository import GLib, Gst, GstRtspServer  # noqa: E402


LOG = logging.getLogger("gst_webcam_rtsp_server")


def make_pipeline(device_index: int, width: int, height: int, fps: int, bitrate: int) -> str:
    return (
        f"( mfvideosrc device-index={device_index} "
        "! queue leaky=downstream max-size-buffers=2 max-size-bytes=0 max-size-time=0 "
        "! videoconvert ! videoscale add-borders=true ! videorate "
        f"! video/x-raw,format=I420,width={width},height={height},framerate={fps}/1 "
        f"! x264enc tune=zerolatency speed-preset=ultrafast threads=2 bitrate={bitrate} "
        f"key-int-max={fps * 2} bframes=0 ! h264parse config-interval=-1 "
        "! rtph264pay name=pay0 pt=96 config-interval=1 )"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device-index", type=int, default=0,
                        help="Windows Media Foundation camera index (default: 0)")
    parser.add_argument("--host", default="127.0.0.1",
                        help="Bind address; loopback keeps the camera stream local (default)")
    parser.add_argument("--port", type=int, default=8555)
    parser.add_argument("--mount", default="/webcam")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--bitrate", type=int, default=2500,
                        help="H.264 target bitrate in kbit/s (default: 2500)")
    args = parser.parse_args()

    if args.device_index < 0 or args.port < 1 or args.port > 65535:
        parser.error("device-index must be nonnegative and port must be 1-65535")
    if min(args.width, args.height, args.fps, args.bitrate) < 1:
        parser.error("width, height, fps, and bitrate must be positive")
    if args.width < 2 or args.height < 2:
        parser.error("width and height must be at least 2 pixels")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    Gst.init(None)
    required = ("mfvideosrc", "videoconvert", "videoscale", "videorate", "x264enc",
                "h264parse", "rtph264pay")
    missing = [name for name in required if Gst.ElementFactory.find(name) is None]
    if missing:
        LOG.error("Missing GStreamer elements: %s", ", ".join(missing))
        return 1

    mount = "/" + args.mount.strip("/")
    factory = GstRtspServer.RTSPMediaFactory()
    factory.set_launch(make_pipeline(args.device_index, args.width, args.height,
                                     args.fps, args.bitrate))
    factory.set_shared(True)
    factory.set_eos_shutdown(False)

    def on_media_configure(_factory: GstRtspServer.RTSPMediaFactory,
                           _media: GstRtspServer.RTSPMedia) -> None:
        LOG.info("RTSP media configured for webcam index %d", args.device_index)

    factory.connect("media-configure", on_media_configure)
    server = GstRtspServer.RTSPServer()
    server.set_address(args.host)
    server.set_service(str(args.port))
    server.get_mount_points().add_factory(mount, factory)

    loop = GLib.MainLoop()
    if server.attach(None) == 0:
        LOG.error("Could not bind RTSP server to %s:%s", args.host, args.port)
        return 1

    print(f"Serving webcam index {args.device_index} as {args.width}x{args.height} at {args.fps} FPS",
          flush=True)
    print(f"RTSP endpoint: rtsp://{args.host}:{args.port}{mount}", flush=True)
    print("The webcam opens when a client connects. Press Ctrl+C to stop.", flush=True)
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda _signal, _frame: loop.quit())
    try:
        loop.run()
    finally:
        LOG.info("Stopping webcam RTSP server")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
