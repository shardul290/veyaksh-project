"use strict";

let liveSession = null;
let liveOptions = null;
let liveBusy = false;
let livePolling = false;
let liveFrameSession = null;
let liveInputToken = null;
let liveFrameBusy = false;
let liveLastCapture = -1;
let liveOwnedSource = null;
let liveSocket = null;
const liveVideo = $("#live-video");
const liveOverlay = $("#live-overlay");
let livePlaybackMeasure = {at: 0, frames: 0, fps: 0};
let liveTimeline = new LiveTiming.DetectionTimeline();
let liveClock = new LiveTiming.PresentationClock();
let liveBufferedFrames = [];
let liveCanvasPool = [];
let liveCapturedTs = -1;
let livePresentedTs = -1;
let livePresentedFrames = 0;
let livePaintedDetection = null;
let liveFeedKey = "";
let liveFlowPaused = false;
let livePlaybackError = "";
let liveVideoCallbackAt = 0;
const liveUploads = new LiveTiming.LatestFrameQueue(sendPlaybackFrame, () =>
  ["starting", "running"].includes(liveSession?.state) && liveSocket?.readyState === WebSocket.OPEN &&
  liveSocket.bufferedAmount <= 100000);
window.liveMonitoring = false;

const liveTime = seconds => {
  const value = Math.max(0, seconds || 0);
  return new Date(value * 1000).toISOString().slice(11, 19);
};

function liveZones() {
  const camera = $("#live-camera").value;
  const selector = $("#live-zones");
  selector.replaceChildren();
  for (const zone of liveOptions.zones.filter(item => item.cam_id === camera)) {
    const option = element("option", zone.name);
    option.value = zone.path;
    selector.append(option);
  }
}

function renderLive(session) {
  if (liveSession?.session_id === session.session_id &&
      session.detection_ts_s < liveSession.detection_ts_s) {
    session = {...session, detection_ts_s: liveSession.detection_ts_s, boxes: liveSession.boxes};
  }
  liveSession = session;
  liveTimeline.add(session);
  liveClock.observe(liveVideo.currentTime, session.detection_ts_s);
  if (liveFlowPaused && ["starting", "running"].includes(session.state) &&
      session.detection_ts_s >= liveCapturedTs - .25) {
    liveFlowPaused = false;
    liveVideo.play().catch(() => {livePlaybackError = "Click the video to resume.";});
  }
  const active = ["starting", "running", "stopping"].includes(session.state);
  window.liveMonitoring = active;
  liveInputToken = sessionStorage.getItem("live-token-" + session.session_id);
  const nativePlayback = session.browser_input && Boolean(liveInputToken);
  if (nativePlayback && active && session.input_ready && liveOwnedSource !== session.session_id) {
    liveOwnedSource = session.session_id;
    liveVideo.src = `/live-sessions/${session.session_id}/playback`;
    liveVideo.onloadedmetadata = () => {
      liveVideo.currentTime = Math.min(session.video_ts_s || 0, Math.max(0, liveVideo.duration - .1));
      liveVideo.play().catch(() => {
        livePlaybackError = "Click the video to begin playback.";
      });
    };
    liveSocket?.close();
    liveSocket = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/live-sessions/${session.session_id}/input`);
    liveSocket.onopen = () => {
      liveSocket.send(JSON.stringify({token: liveInputToken}));
      liveUploads.flush();
    };
    liveSocket.onmessage = message => {
      const status = JSON.parse(message.data);
      if (liveSession?.session_id === status.session_id) renderLive(status);
      liveUploads.flush();
    };
    liveSocket.onerror = () => { $("#live-error").textContent = "The frame connection was interrupted. Stop and restart the session."; };
  }
  if (!active) {liveVideo.pause();liveSocket?.close();liveSocket = null;liveUploads.clear();}
  $("#live-start").disabled = active || liveBusy;
  $("#live-stop").disabled = !active || session.state === "stopping" || liveBusy;
  for (const id of ["live-source", "live-camera", "live-zones", "live-confidence", "live-saved"]) {
    $("#" + id).disabled = active || liveBusy;
  }
  const names = {starting: "Preparing model and source", running: "Live analysis",
    stopping: "Stopping", stopped: "Stopped", completed: "Video completed", failed: "Session failed"};
  $("#live-state").textContent = names[session.state] || session.state;
  $("#live-state").dataset.state = session.state;
  $("#live-error").textContent = session.error || livePlaybackError;
  $("#live-model").textContent = session.model;
  $("#live-engine").textContent = session.backend
    ? `${session.decoder} · ${session.backend} · ByteTrack` : "GStreamer · YOLO26 · ByteTrack";
  $("#live-decoder-warning").textContent = session.decoder_warning || "";
  $("#live-clock").textContent = liveTime(nativePlayback ? livePresentedTs : session.presentation_ts_s ?? session.video_ts_s);
  const duration = nativePlayback && Number.isFinite(liveVideo.duration) ? liveVideo.duration : session.duration_s;
  $("#live-duration").textContent = duration ? "/ " + liveTime(duration) : "";
  $("#live-display-fps").textContent = (nativePlayback ? livePlaybackMeasure.fps : session.display_fps || 0).toFixed(1);
  $("#live-detection-fps").textContent = (session.detection_fps || 0).toFixed(1);
  if (!nativePlayback) $("#live-box-age").textContent = session.box_age_s == null ? "—" : Math.round(session.box_age_s * 1000) + " ms";
  $("#live-events-count").textContent = session.event_count || 0;
  $("#live-frame-note").textContent = session.state === "starting"
    ? session.preparation || "Connecting and warming the detector…"
    : active ? "Video and boxes share a buffered clock. Display delay adapts to detection speed."
    : "Session ended. Start a source to continue monitoring.";
  if (!nativePlayback && (active || session.has_frame) && liveFrameSession !== session.session_id) {
    liveFrameSession = session.session_id;
    $("#live-image").src = session.frame_url;
  }
  $("#live-empty").hidden = session.has_frame || nativePlayback && session.input_ready;
  $("#live-image").hidden = nativePlayback || !session.has_frame;
  liveVideo.hidden = !nativePlayback || !session.input_ready;
  liveOverlay.hidden = liveVideo.hidden;
  const feedKey = `${session.session_id}:${active}:${(session.recent_events || []).map(e => e.id).join(",")}`;
  if (feedKey === liveFeedKey) return;
  liveFeedKey = feedKey;
  const feed = $("#live-event-list");
  feed.replaceChildren();
  for (const event of session.recent_events || []) {
    const button = element("button", undefined, "live-event");
    button.type = "button";
    button.append(element("strong", typeName(event.type)),
      element("span", `${event.cam_id} · ${liveTime(event.video_ts_s)} · #${event.track_id}`));
    button.addEventListener("click", () => {
      showEvent(events.find(item => item.id === event.id) || {...event, status: "pending"});
    });
    feed.append(button);
  }
  if (!feed.children.length) feed.append(element("p", "Confirmed events appear here and in the evidence trail.", "muted"));
  renderEvidence();
}

async function pollLive() {
  if (livePolling || !liveSession) return;
  livePolling = true;
  try {
    const session = await api(`/live-sessions/${liveSession.session_id}`);
    renderLive(session);
  } catch (error) {
    $("#live-error").textContent = error.message;
    if (error.status === 404) {
      liveSession = null;
      window.liveMonitoring = false;
      $("#live-start").disabled = false;
      $("#live-stop").disabled = true;
      $("#live-state").textContent = "Backend restarted";
      for (const control of $("#live-form").elements) control.disabled = false;
      $("#live-stop").disabled = true;
      liveVideo.pause();
      liveSocket?.close();
    }
  } finally {
    livePolling = false;
  }
}

$("#live-form").addEventListener("submit", async event => {
  event.preventDefault();
  if (liveBusy) return;
  liveBusy = true;
  $("#live-start").disabled = true;
  $("#live-error").textContent = "";
  try {
    const session = await api("/live-sessions", {method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({source: $("#live-source").value, cam_id: $("#live-camera").value,
        zone_config: $("#live-zones").value, confidence: Number($("#live-confidence").value),
        browser_input: !/^rtsps?:\/\//i.test($("#live-source").value.trim())})});
    $("#camera-video").pause();
    clearPlayback();
    selectedCamera = null;
    sessionStorage.setItem("live-token-" + session.session_id, session.input_token);
    liveLastCapture = -1;
    liveTimeline = new LiveTiming.DetectionTimeline();
    liveClock = new LiveTiming.PresentationClock();
    liveCanvasPool.push(...liveBufferedFrames.map(item => item.canvas));
    liveBufferedFrames = [];
    liveCapturedTs = livePresentedTs = -1;
    livePresentedFrames = 0;
    livePaintedDetection = null;
    liveFlowPaused = false;
    liveUploads.clear();
    livePlaybackError = "";
    livePlaybackMeasure = {at: performance.now(), frames: 0, fps: 0};
    liveOwnedSource = null;
    liveFrameSession = null;
    liveBusy = false;
    renderLive(session);
  } catch (error) {
    $("#live-error").textContent = error.message;
    $("#live-start").disabled = false;
  } finally {
    liveBusy = false;
  }
});

$("#live-stop").addEventListener("click", async () => {
  if (liveBusy || !liveSession) return;
  liveBusy = true;
  $("#live-stop").disabled = true;
  try {
    const session = await api(`/live-sessions/${liveSession.session_id}/stop`, {method: "POST"});
    liveBusy = false;
    renderLive(session);
  } catch (error) {
    $("#live-error").textContent = error.message;
  } finally {
    liveBusy = false;
    await pollLive();
  }
});

$("#live-camera").addEventListener("change", liveZones);
$("#live-saved").addEventListener("change", () => {
  const value = $("#live-saved").value;
  if (!value) return;
  $("#live-source").value = value;
  const filename = value.split("/").pop();
  const camera = {"1.1.mp4": "C01", "2.2.mp4": "C02"}[filename];
  if (camera) {
    $("#live-camera").value = camera;
    liveZones();
  }
});

async function initializeLive() {
  if (historyPage) return;
  try {
    liveOptions = await api("/live-options");
    for (const camera of liveOptions.cameras) {
      const option = element("option", `${camera.cam_id} · ${camera.name} · ${nodeName(camera.node_id)}`);
      option.value = camera.cam_id;
      $("#live-camera").append(option);
    }
    liveZones();
    for (const source of liveOptions.sources) {
      const option = element("option", source.name);
      option.value = source.path;
      $("#live-saved").append(option);
    }
    $("#live-model").textContent = liveOptions.model;
    $("#live-start").disabled = false;
    const sessions = await api("/live-sessions");
    if (sessions.length) renderLive(sessions.find(s => ["starting", "running", "stopping"].includes(s.state)) || sessions[0]);
    setInterval(pollLive, 1000);
  } catch (error) {
    $("#live-error").textContent = error.message;
  }
}
initializeLive();

async function sendPlaybackFrame(frame) {
  if (!liveSession?.browser_input || !liveInputToken) return;
  if (frame.ts <= liveLastCapture || !liveFlowPaused && frame.ts - liveLastCapture < .09) return;
  liveFrameBusy = true;
  const session = liveSession;
  const timestamp = frame.ts;
  liveLastCapture = timestamp;
  try {
    const blob = await new Promise(resolve => frame.canvas.toBlob(resolve, "image/jpeg", .88));
    if (!blob) throw new Error("The browser could not capture a video frame.");
    const header = new ArrayBuffer(8);
    new DataView(header).setFloat64(0, timestamp, true);
    if (liveSocket?.readyState === WebSocket.OPEN && liveSession?.session_id === session.session_id) {
      liveSocket.send(new Blob([header, blob]));
    }
  } catch (error) {
    $("#live-error").textContent = error.message;
  } finally {
    liveFrameBusy = false;
  }
}

function captureLiveFrame(timestamp) {
  if (liveVideo.hidden || liveVideo.paused || liveVideo.readyState < 2 ||
      timestamp - liveCapturedTs < .039) return;
  liveCapturedTs = timestamp;
  // 32 frames at 960x540 use about 64 MiB. Neither video nor inference queues grow indefinitely.
  const canvas = liveCanvasPool.pop() || document.createElement("canvas");
  const width = Math.min(960, liveVideo.videoWidth);
  const height = Math.round(liveVideo.videoHeight * width / liveVideo.videoWidth);
  if (canvas.width !== width || canvas.height !== height) {canvas.width = width;canvas.height = height;}
  canvas.getContext("2d", {alpha: false}).drawImage(liveVideo, 0, 0, width, height);
  const frame = {ts: timestamp, canvas};
  liveBufferedFrames.push(frame);
  while (liveBufferedFrames.length > 32) liveCanvasPool.push(liveBufferedFrames.shift().canvas);
  liveUploads.offer(frame);
  const latest = liveTimeline.latest ?? 0;
  if (frame.ts - latest >= .7) {
    liveFlowPaused = true;
    liveVideo.pause();
  }
}

if (liveVideo.requestVideoFrameCallback) {
  const onFrame = (_now, metadata) => {
    liveVideoCallbackAt = performance.now();
    captureLiveFrame(metadata.mediaTime);
    liveVideo.requestVideoFrameCallback(onFrame);
  };
  liveVideo.requestVideoFrameCallback(onFrame);
}
// Browsers can throttle video-frame callbacks when the decoding element is
// occluded or scrolled offscreen. Continue sampling its current decoded frame.
setInterval(() => {
  if (performance.now() - liveVideoCallbackAt > 100) captureLiveFrame(liveVideo.currentTime);
}, 40);

function paintLiveOverlay() {
  if (!liveVideo.hidden && liveSession) {
    const width = 960;
    const height = Math.round(width * liveVideo.videoHeight / liveVideo.videoWidth) || 540;
    if (liveOverlay.width !== width || liveOverlay.height !== height) {
      liveOverlay.width = width;
      liveOverlay.height = height;
    }
    const ctx = liveOverlay.getContext("2d");
    const now = performance.now();
    const frames = livePresentedFrames;
    if (now - livePlaybackMeasure.at >= 1000) {
      livePlaybackMeasure = {at: now, frames,
        fps: Math.max(0, (frames - livePlaybackMeasure.frames) * 1000 / (now - livePlaybackMeasure.at))};
    }
    $("#live-display-fps").textContent = livePlaybackMeasure.fps.toFixed(1);
    const target = liveClock.advance(now, liveVideo.currentTime, liveTimeline.latest, liveBufferedFrames[0]?.ts);
    const frame = target == null ? null : liveBufferedFrames.findLast(item => item.ts <= target + .001);
    if (!frame) {requestAnimationFrame(paintLiveOverlay);return;}
    const changed = frame.ts !== livePresentedTs;
    if (changed) {livePresentedFrames++;livePresentedTs = frame.ts;}
    if (!changed && livePaintedDetection === liveTimeline.latest) {requestAnimationFrame(paintLiveOverlay);return;}
    livePaintedDetection = liveTimeline.latest;
    ctx.drawImage(frame.canvas, 0, 0, width, height);
    liveOverlay.dataset.frameTime = frame.ts.toFixed(3);
    liveOverlay.dataset.detectionTime = liveTimeline.latest.toFixed(3);
    $("#live-clock").textContent = liveTime(frame.ts);
    $("#live-box-age").textContent = Math.round(Math.max(0, liveVideo.currentTime - frame.ts) * 1000) + " ms";
    const zone = liveOptions?.zones.find(item => item.path === $("#live-zones").value);
    if (zone) {
      ctx.font = "14px Segoe UI";
      ctx.lineWidth = 1.5;
      for (const box of liveTimeline.boxesAt(frame.ts)) {
        const [x1, y1, x2, y2] = box.bbox;
        const x = x1 * width / zone.frame_size.width;
        const y = y1 * height / zone.frame_size.height;
        const w = (x2 - x1) * width / zone.frame_size.width;
        const h = (y2 - y1) * height / zone.frame_size.height;
        const unsafe = box.alert || box.path?.status === "wrong_path";
        ctx.strokeStyle = unsafe ? "#ff5b65" : box.class === "pedestrian" ? "#6bdd91" : "#68c7ed";
        ctx.strokeRect(x, y, w, h);
        const label = `${box.track_id == null ? "Detecting" : "#" + box.track_id} ${box.class}${unsafe ? " | ALERT" : ""}`;
        const textWidth = ctx.measureText(label).width + 8;
        ctx.fillStyle = "rgba(8,20,34,.85)";
        ctx.fillRect(x, Math.max(0, y - 21), textWidth, 21);
        ctx.fillStyle = ctx.strokeStyle;
        ctx.fillText(label, x + 4, Math.max(15, y - 5));
      }
    }
  }
  requestAnimationFrame(paintLiveOverlay);
}
$(".live-stage").addEventListener("click", () => {
  if (liveVideo.paused && liveSession && ["starting", "running"].includes(liveSession.state)) liveVideo.play().catch(() => {});
});
liveVideo.addEventListener("playing", () => {livePlaybackError = "";});
liveVideo.addEventListener("ended", async () => {
  while (liveFrameBusy || liveUploads.busy) await new Promise(resolve => setTimeout(resolve, 50));
  if (liveSocket?.readyState === WebSocket.OPEN) {
    liveSocket.send("end");
  } else if (liveSession && liveInputToken) {
    api(`/live-sessions/${liveSession.session_id}/end`, {method: "POST",
      headers: {"X-Playback-Token": liveInputToken}}).catch(error => {$("#live-error").textContent = error.message;});
  }
});
requestAnimationFrame(paintLiveOverlay);
setInterval(() => liveUploads.flush(), 80);
