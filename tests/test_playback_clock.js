"use strict";
const assert=require("node:assert/strict");
const {test}=require("node:test");
const PlaybackClock=require("../m6_dashboard/playback-clock.js");
const event=(time,id=1)=>({video_ts_s:time,track_id:id});
test("does not alert before timestamp; alerts once at timestamp",()=>{
  const c=new PlaybackClock([event(10)]);
  assert.equal(c.advance(9.99).length,0);assert.equal(c.advance(10).length,1);
  assert.equal(c.advance(10).length,0);assert.equal(c.advance(10.5).length,0);
});
test("pausing never consumes or dispatches events",()=>{
  const c=new PlaybackClock([event(10)]);
  c.advance(9);assert.equal(c.advance(10,false).length,0);assert.equal(c.advance(10,true).length,1);
});
test("playback-rate changes and dropped frames preserve crossed alerts",()=>{
  const c=new PlaybackClock([event(10),event(10.1,2),event(12,3)]);
  c.advance(9);assert.deepEqual(c.advance(11).map(e=>e.track_id),[1,2]);
  assert.equal(c.advance(12).length,1);
});
test("seeking forward skips past alerts",()=>{
  const c=new PlaybackClock([event(5),event(10),event(15)]);
  c.advance(2);c.seek(12);assert.equal(c.advance(12).length,0);assert.equal(c.advance(15).length,1);
});
test("seeking onto event while paused waits for play",()=>{
  const c=new PlaybackClock([event(10)]);c.seek(10);
  assert.equal(c.advance(10,false).length,0);assert.equal(c.advance(10,true).length,1);
});
test("backward replay re-arms popups",()=>{
  const c=new PlaybackClock([event(10)]);c.advance(10);c.seek(0);
  assert.equal(c.advance(9).length,0);assert.equal(c.advance(10).length,1);
});
test("loop starts a new popup cycle",()=>{
  const c=new PlaybackClock([event(0),event(1)]);assert.equal(c.advance(0).length,1);
  c.advance(2);assert.equal(c.advance(0).length,1);assert.equal(c.advance(1).length,1);
});
test("source event indexes survive chronological sorting",()=>{
  const c=new PlaybackClock([event(20),event(10)]);
  assert.equal(c.advance(10)[0].event_index,1);assert.equal(c.advance(20)[0].event_index,0);
});
