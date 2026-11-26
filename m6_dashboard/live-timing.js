/* Timestamp-based presentation shared by the live view and its regression tests. */
(function (root) {
  "use strict";
  class DetectionTimeline {
    constructor() { this.samples = []; }
    get latest() { return this.samples.at(-1)?.ts ?? null; }
    add(status) {
      const ts = status.detection_ts_s;
      if (!Number.isFinite(ts)) return;
      const sample = {ts, boxes: status.boxes || []};
      const index = this.samples.findIndex(item => item.ts === ts);
      if (index >= 0) this.samples[index] = sample;
      else this.samples.push(sample);
      this.samples.sort((a, b) => a.ts - b.ts);
      while (this.samples.length > 40 || this.latest - this.samples[0].ts > 3) this.samples.shift();
    }
    boxesAt(ts) {
      if (!this.samples.length) return [];
      const rightIndex = this.samples.findIndex(item => item.ts >= ts);
      const right = this.samples[rightIndex];
      const left = rightIndex < 0 ? this.samples.at(-1) : this.samples[rightIndex - 1];
      if (right && Math.abs(right.ts - ts) < .001) return right.boxes;
      if (!left || !right || right.ts - left.ts > .7) {
        const nearest = [left, right].filter(Boolean).sort((a,b) => Math.abs(a.ts-ts)-Math.abs(b.ts-ts))[0];
        return nearest && Math.abs(nearest.ts - ts) <= .04 ? nearest.boxes : [];
      }
      const ratio = (ts - left.ts) / (right.ts - left.ts);
      const next = new Map(right.boxes.filter(box => box.track_id != null).map(box => [box.track_id, box]));
      const boxes = [];
      for (const box of left.boxes) {
        const other = box.track_id == null ? null : next.get(box.track_id);
        if (other && other.class === box.class) {
          boxes.push({...box, bbox: box.bbox.map((value, i) => value + (other.bbox[i] - value) * ratio)});
        } else if (ts - left.ts <= .04) boxes.push(box);
      }
      if (right.ts - ts <= .04) {
        const shown = new Set(boxes.filter(box => box.track_id != null).map(box => box.track_id));
        boxes.push(...right.boxes.filter(box => box.track_id == null || !shown.has(box.track_id)));
      }
      return boxes;
    }
  }

  class PresentationClock {
    constructor() { this.time = null; this.at = null; this.delay = .35; this.lags = []; this.last = null; }
    observe(sourceTime, detectionTime) {
      if (!Number.isFinite(detectionTime) || detectionTime === this.last) return;
      this.last = detectionTime;
      this.lags.push(Math.max(0, sourceTime - detectionTime));
      if (this.lags.length > 16) this.lags.shift();
      const ordered = [...this.lags].sort((a,b) => a-b);
      this.delay = Math.min(1.2, Math.max(.25, ordered[Math.floor((ordered.length - 1) * .9)] + .15));
    }
    advance(now, sourceTime, latestDetection, oldestFrame) {
      const elapsed = this.at == null ? 0 : Math.max(0, Math.min(.1, (now - this.at) / 1000));
      this.at = now;
      if (latestDetection == null || oldestFrame == null || latestDetection < oldestFrame) return null;
      const target = Math.min(latestDetection, Math.max(oldestFrame, sourceTime - this.delay));
      if (this.time == null || this.time < oldestFrame) this.time = target;
      else if (target >= this.time) this.time = Math.min(target, this.time + elapsed * 1.08);
      return Math.min(this.time, latestDetection);
    }
  }
  class LatestFrameQueue {
    constructor(send, ready) {this.send = send;this.ready = ready;this.pending = null;this.busy = false;}
    offer(frame) {this.pending = frame;return this.flush();}
    async flush() {
      if (this.busy || !this.pending || !this.ready()) return;
      const frame = this.pending;
      this.pending = null;
      this.busy = true;
      try {await this.send(frame);}
      finally {this.busy = false;await this.flush();}
    }
    clear() {this.pending = null;}
  }
  const api = {DetectionTimeline, PresentationClock, LatestFrameQueue};
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.LiveTiming = api;
})(typeof window !== "undefined" ? window : globalThis);
