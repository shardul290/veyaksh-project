"use strict";
const assert = require("node:assert/strict");
const {test} = require("node:test");
const {DetectionTimeline, PresentationClock, LatestFrameQueue} = require("../m6_dashboard/live-timing.js");
const box = (id, x, name="car") => ({track_id:id, class:name, bbox:[x, 20, x+40, 60]});
test("moving boxes are interpolated at the presented video timestamp", () => {
  const t = new DetectionTimeline();
  t.add({detection_ts_s:1, boxes:[box(7,100)]});
  t.add({detection_ts_s:1.4, boxes:[box(7,200)]});
  assert.deepEqual(t.boxesAt(1.2)[0].bbox,[150,20,190,60]);
  assert.equal(t.boxesAt(1.7).length,0);
});
test("unconfirmed detections remain visible on their frame without invented trajectories", () => {
  const t = new DetectionTimeline();
  t.add({detection_ts_s:1, boxes:[box(null,100), box(2,300)]});
  t.add({detection_ts_s:1.4, boxes:[box(null,200)]});
  assert.equal(t.boxesAt(1).length,2);
  assert.equal(t.boxesAt(1.2).length,0);
  assert.equal(t.boxesAt(1.4).length,1);
});
test("ID reuse with a different class and long result gaps never interpolates", () => {
  const t = new DetectionTimeline();
  t.add({detection_ts_s:1, boxes:[box(1,100)]});
  t.add({detection_ts_s:1.4, boxes:[box(1,200,"pedestrian")]});
  t.add({detection_ts_s:3, boxes:[box(1,300,"pedestrian")]});
  assert.equal(t.boxesAt(1.2).length,0);
  assert.equal(t.boxesAt(2).length,0);
});
test("out of order results are sorted and history stays bounded", () => {
  const t = new DetectionTimeline();
  for(let i=100;i>=0;i--) t.add({detection_ts_s:i/10, boxes:[]});
  assert.equal(t.latest,10);
  assert.ok(t.samples.length <= 31);
});
test("presentation cannot advance beyond a completed detection", () => {
  const c = new PresentationClock();
  c.observe(1.4,1);
  let value=c.advance(0,1.4,1,0);
  for(let now=40;now<3000;now+=40) {
    value=c.advance(now,5,1,0);
    assert.ok(value<=1);
  }
  assert.equal(value,1);
});
test("slow inference adapts delay, stays monotonic, and recovers after old frames are discarded", () => {
  const c = new PresentationClock();
  c.observe(1.8,1);
  assert.ok(c.delay >= .9 && c.delay <= 1.2);
  const first=c.advance(1000,1.8,1,.2);
  const second=c.advance(1040,1.9,1,.2);
  assert.ok(second>=first);
  assert.equal(c.advance(1080,4,2,2.5),null);
  assert.ok(c.advance(1120,4,3,2.5)>=2.5);
});
test("an in-flight encode drains the newest pending frame after playback pauses", async () => {
  const sent=[];
  let finish;
  const firstDone=new Promise(resolve => {finish=resolve;});
  const q=new LatestFrameQueue(async frame => {sent.push(frame);if(frame===1) await firstDone;},()=>true);
  const draining=q.offer(1);
  q.offer(2);q.offer(3);q.offer(4);
  assert.equal(q.pending,4);
  finish();await draining;
  assert.deepEqual(sent,[1,4]);
  assert.equal(q.busy,false);
});
test("frames captured before the socket opens flush automatically when ready", async () => {
  let ready=false;
  const sent=[];
  const q=new LatestFrameQueue(async frame => {sent.push(frame);},()=>ready);
  await q.offer(1);await q.offer(2);
  assert.deepEqual(sent,[]);
  ready=true;await q.flush();
  assert.deepEqual(sent,[2]);
});
