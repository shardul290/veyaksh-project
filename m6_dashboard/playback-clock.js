/* Pure media-time scheduler, shared by browser playback and Node tests. */
(function(root) {
  "use strict";
  class PlaybackClock {
    constructor(events) {
      this.events = events.map((event, index) => ({...event, event_index:index}))
        .sort((a,b) => a.video_ts_s-b.video_ts_s || a.event_index-b.event_index);
      this.last = -0.000001;
      this.seen = new Set();
    }
    seek(time) {
      if (time < this.last) this.seen.clear();
      // Seeking skips intervening events. An event exactly at the landing
      // timestamp may fire when playback resumes, rather than while paused.
      this.last = time - 0.000001;
    }
    advance(time, playing=true) {
      if (!playing || !Number.isFinite(time)) return [];
      if (time < this.last - .1) { this.seen.clear(); this.last=-0.000001; }
      const due = this.events.filter(e => e.video_ts_s > this.last && e.video_ts_s <= time + 0.000001 && !this.seen.has(e.event_index));
      for (const event of due) this.seen.add(event.event_index);
      this.last = time;
      return due;
    }
  }
  if (typeof module !== "undefined" && module.exports) module.exports=PlaybackClock;
  else root.PlaybackClock=PlaybackClock;
})(typeof globalThis !== "undefined" ? globalThis : this);
