"""Decode a local video with GStreamer and stream timestamped BGR frames to stdout.

This helper intentionally uses only PyGObject/GStreamer. The main exporter can
therefore keep using its existing PyTorch/OpenCV virtual environment.
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstVideo", "1.0")
from gi.repository import Gst, GstVideo

FRAME_HEADER = struct.Struct("<QqII")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--video", type=Path, help="Local video file")
    source_group.add_argument("--uri", help="RTSP or other GStreamer-supported media URI")
    parser.add_argument("--realtime", action="store_true",
                        help="Synchronize appsink output to the video's timestamps")
    parser.add_argument("--output-width", type=int,
                        help="Optional preview width; height is supplied separately")
    parser.add_argument("--output-height", type=int)
    args = parser.parse_args()
    if args.video is not None and not args.video.is_file():
        parser.error(f"Video not found: {args.video}")
    if ((args.output_width is None) != (args.output_height is None)
            or (args.output_width is not None and (args.output_width < 2 or args.output_height < 2))):
        parser.error("output-width and output-height must be supplied together and be >=2")

    Gst.init(None)
    pipeline = Gst.Pipeline.new("traffic-detection-gstreamer-decoder")
    source = Gst.ElementFactory.make("uridecodebin", "source")
    queue = Gst.ElementFactory.make("queue", "video-queue")
    scale = Gst.ElementFactory.make("videoscale", "preview-scale") if args.output_width else None
    convert = Gst.ElementFactory.make("videoconvert", "bgr-converter")
    capsfilter = Gst.ElementFactory.make("capsfilter", "preview-caps") if args.output_width else None
    sink = Gst.ElementFactory.make("appsink", "frame-sink")
    required = [pipeline, source, queue, convert, sink]
    if args.output_width:
        required.extend((scale, capsfilter))
    if not all(required):
        raise RuntimeError("Missing GStreamer elements; install the base/good/bad plugin packages")

    source_uri = args.uri if args.uri else args.video.resolve().as_uri()
    source.set_property("uri", source_uri)
    queue.set_property("max-size-buffers", 2)
    queue.set_property("max-size-bytes", 0)
    queue.set_property("max-size-time", 0)
    sink.set_property("caps", Gst.Caps.from_string("video/x-raw,format=BGR"))
    if capsfilter:
        capsfilter.set_property("caps", Gst.Caps.from_string(
            f"video/x-raw,format=BGR,width={args.output_width},height={args.output_height}"))
    sink.set_property("emit-signals", False)
    sink.set_property("sync", args.realtime)
    sink.set_property("max-buffers", 2)
    sink.set_property("drop", False)
    sink.set_property("enable-last-sample", False)

    elements = [source, queue]
    if scale:
        elements.append(scale)
    elements.append(convert)
    if capsfilter:
        elements.append(capsfilter)
    elements.append(sink)
    for element in elements:
        pipeline.add(element)
    if scale:
        linked = queue.link(scale) and scale.link(convert) and convert.link(capsfilter) and capsfilter.link(sink)
    else:
        linked = queue.link(convert) and convert.link(sink)
    if not linked:
        raise RuntimeError("Could not link the GStreamer video conversion pipeline")

    link_errors: list[str] = []

    def link_video_pad(_decoder, pad) -> None:
        caps = pad.get_current_caps() or pad.query_caps(None)
        if not caps or caps.get_size() == 0:
            return
        if not caps.get_structure(0).get_name().startswith("video/"):
            return
        queue_sink = queue.get_static_pad("sink")
        if queue_sink.is_linked():
            return
        result = pad.link(queue_sink)
        if result != Gst.PadLinkReturn.OK:
            link_errors.append(str(result))

    source.connect("pad-added", link_video_pad)
    bus = pipeline.get_bus()
    state_result = pipeline.set_state(Gst.State.PLAYING)
    if state_result == Gst.StateChangeReturn.FAILURE:
        pipeline.set_state(Gst.State.NULL)
        raise RuntimeError("GStreamer could not start decoding this video")

    frame_index = 0
    try:
        output = sys.stdout.buffer
        while True:
            sample = sink.emit("try-pull-sample", 500 * Gst.MSECOND)
            if sample is not None:
                caps = sample.get_caps()
                video_info = GstVideo.VideoInfo.new_from_caps(caps)
                if video_info is None:
                    raise RuntimeError("GStreamer did not provide BGR frame layout metadata")
                width = int(video_info.width)
                height = int(video_info.height)
                stride = int(video_info.stride[0])
                offset = int(video_info.offset[0])
                row_bytes = width * 3
                buffer = sample.get_buffer()
                mapped_ok, mapped = buffer.map(Gst.MapFlags.READ)
                if not mapped_ok:
                    raise RuntimeError(f"Could not map decoded frame {frame_index}")
                try:
                    pixels = memoryview(mapped.data)
                    if stride < row_bytes or offset + stride * (height - 1) + row_bytes > len(pixels):
                        raise RuntimeError(f"Invalid BGR row layout in decoded frame {frame_index}")
                    pts = int(buffer.pts)
                    pts_ns = -1 if pts == Gst.CLOCK_TIME_NONE else pts
                    output.write(FRAME_HEADER.pack(frame_index, pts_ns, width, height))
                    if stride == row_bytes and offset == 0:
                        output.write(pixels[:row_bytes * height])
                    else:
                        packed = bytearray(row_bytes * height)
                        for row in range(height):
                            source_start = offset + row * stride
                            target_start = row * row_bytes
                            packed[target_start:target_start + row_bytes] = pixels[source_start:source_start + row_bytes]
                        output.write(packed)
                    output.flush()
                finally:
                    buffer.unmap(mapped)
                frame_index += 1
                continue

            if link_errors:
                raise RuntimeError(f"Could not link the decoded video stream: {link_errors[0]}")
            message = bus.pop_filtered(Gst.MessageType.ERROR | Gst.MessageType.EOS)
            if message is None:
                continue
            if message.type == Gst.MessageType.EOS:
                break
            error, debug = message.parse_error()
            raise RuntimeError(f"GStreamer decode error: {error}; {debug or ''}")
    finally:
        pipeline.set_state(Gst.State.NULL)

    if frame_index == 0:
        raise RuntimeError("GStreamer reached end-of-stream without decoding any video frames")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
