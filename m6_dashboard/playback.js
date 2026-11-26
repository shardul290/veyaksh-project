"use strict";
// Non-modal alerts leave the monitored video running. Only Review opens a dialog.
let playbackSession=null, playbackGeneration=0, frameRequest=null, fallbackRequest=null;
const monitor=document.querySelector("#camera-video");
const overlay=document.querySelector("#detection-overlay");
const analysisStatus=document.querySelector("#analysis-status");
const playbackAlerts=document.querySelector("#playback-alerts");
const offsetLabel=value=>`${Math.floor(value/60).toString().padStart(2,"0")}:${(value%60).toFixed(2).padStart(5,"0")}`;

function clearPlaybackAlerts() {
  for(const video of playbackAlerts.querySelectorAll("video"))video.pause();
  playbackAlerts.replaceChildren();
}

function clearPlayback() {
  playbackGeneration++;
  playbackSession=null;
  document.querySelector("#next-event").disabled=true;
  document.querySelector("#playback-time").textContent="00:00.00";
  if(frameRequest!==null && monitor.cancelVideoFrameCallback) monitor.cancelVideoFrameCallback(frameRequest);
  if(fallbackRequest!==null) cancelAnimationFrame(fallbackRequest);
  frameRequest=fallbackRequest=null;
  overlay.getContext("2d").clearRect(0,0,overlay.width,overlay.height);
  clearPlaybackAlerts();
  analysisStatus.textContent="Select a recording to load its analysis.";
}

async function loadPlaybackAnalysis(camera, recording) {
  clearPlayback();
  const generation=playbackGeneration;
  analysisStatus.textContent="Loading synchronized detections…";
  try {
    const analysis=await api(`/cameras/${encodeURIComponent(camera.cam_id)}/analysis`);
    if(generation!==playbackGeneration)return false;
    if(analysis.recording!==recording.path)throw new Error("Analysis belongs to a different recording.");
    const frames=new Map();
    for(const row of analysis.detections){if(!frames.has(row.ts))frames.set(row.ts,[]);frames.get(row.ts).push(row);}
    const times=[...frames.keys()].sort((a,b)=>a-b);
    playbackSession={analysis,clock:new PlaybackClock(analysis.events),frames,times,generation};
    document.querySelector("#next-event").disabled=!analysis.events.length;
    analysisStatus.textContent=`Analysis ready · ${analysis.events.length} events · alerts follow video time`;
    if(analysis.coverage && !analysis.coverage.complete && analysis.coverage.source_duration_s-analysis.coverage.last_detection_s>1)
      analysisStatus.textContent+=` · coverage unverified after ${offsetLabel(analysis.coverage.last_detection_s)}`;
    paintDetections(monitor.currentTime);
    return true;
  }catch(error){
    if(generation===playbackGeneration)analysisStatus.textContent=error.status===404&&error.message==="Not Found"
      ? "Server update needed: restart the backend, then select this camera again to show detection boxes."
      : `Raw recording · ${error.message}`;
    return false;
  }
}

function paintDetections(mediaTime) {
  document.querySelector("#playback-time").textContent=offsetLabel(mediaTime);
  const rect=overlay.getBoundingClientRect(), dpr=window.devicePixelRatio||1;
  const width=Math.max(1,Math.round(rect.width*dpr)),height=Math.max(1,Math.round(rect.height*dpr));
  if(overlay.width!==width||overlay.height!==height){overlay.width=width;overlay.height=height;}
  const ctx=overlay.getContext("2d");ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,rect.width,rect.height);
  if(!playbackSession)return;
  const {analysis,frames,times}=playbackSession;
  let low=0,high=times.length;
  while(low<high){const mid=(low+high)>>1;if(times[mid]<=mediaTime+.00001)low=mid+1;else high=mid;}
  const ts=times[low-1];
  if(ts===undefined||mediaTime-ts>.25)return;
  const source=analysis.frame_size, scale=Math.min(rect.width/source.width,rect.height/source.height);
  const dx=(rect.width-source.width*scale)/2,dy=(rect.height-source.height*scale)/2;
  ctx.font="9px system-ui, sans-serif";ctx.lineWidth=1;
  for(const row of frames.get(ts)) {
    const unsafe=row.path?.status==="wrong_path" || analysis.events.some(e=>e.track_id===row.track_id&&mediaTime>=e.video_ts_s&&mediaTime-e.video_ts_s<=2);
    const color=unsafe?"#ff5858":row.class==="pedestrian"?"#48df86":"#71c9ef";
    const [x1,y1,x2,y2]=row.bbox, x=x1*scale+dx,y=y1*scale+dy,w=(x2-x1)*scale,h=(y2-y1)*scale;
    ctx.strokeStyle=color;ctx.globalAlpha=.85;ctx.strokeRect(x,y,w,h);ctx.globalAlpha=1;
    const displayClass=row.class==="pedestrian"?"ped":row.class;
    const suffix=unsafe?(row.class==="pedestrian"?" · WP":" · alert"):row.path?.status==="right_path"?" · right":"";
    const label=`#${row.track_id??"?"} ${displayClass}${suffix}`, textWidth=ctx.measureText(label).width;
    const left=Math.max(0,Math.min(rect.width-textWidth-4,x)), top=Math.max(0,y-14);
    ctx.fillStyle="rgba(8,18,29,.18)";ctx.fillRect(left,top,textWidth+4,13);
    ctx.fillStyle=color;ctx.fillText(label,left+2,top+10);
  }
}

function showPlaybackAlert(event, session) {
  const card=element("article",undefined,"playback-alert");card.setAttribute("role","status");
  const heading=element("div",undefined,"playback-alert-heading"), close=element("button","×");
  close.type="button";close.setAttribute("aria-label","Dismiss playback alert");
  close.addEventListener("click",()=>{card.querySelector("video")?.pause();card.remove();});
  heading.append(element("strong",typeName(event.type)),close);
  const detail=element("p",`${session.analysis.cam_id} · ${offsetLabel(event.video_ts_s)} · #${event.track_id} ${event.class}`);
  const mini=element("video");mini.muted=true;mini.playsInline=true;mini.controls=true;mini.preload="metadata";
  mini.src=mediaUrl(`${session.analysis.recording}#t=${Math.max(0,event.video_ts_s-1).toFixed(3)}`);
  mini.addEventListener("timeupdate",()=>{if(mini.currentTime>=event.video_ts_s+3)mini.pause();});
  mini.addEventListener("error",()=>{mini.hidden=true;});
  const saved=element("p","Saving to event trail…","playback-save-status"), review=element("button","Review evidence");review.disabled=true;
  card.append(heading,detail,mini,saved,review);playbackAlerts.prepend(card);
  mini.play().catch(()=>{});
  while(playbackAlerts.children.length>3){const old=playbackAlerts.lastElementChild;old.querySelector("video")?.pause();old.remove();}
  beep();
  let attempts=0;
  async function persist() {
    attempts++;
    try {
      const result=await api(`/cameras/${encodeURIComponent(session.analysis.cam_id)}/playback-events`,{
        method:"POST",keepalive:true,headers:{"Content-Type":"application/json"},
        body:JSON.stringify({analysis_id:session.analysis.analysis_id,event_index:event.event_index})});
      saved.textContent=`Saved to event trail · #${result.id}`;review.disabled=false;
      review.onclick=()=>showEvent(result);
      events=[result,...events.filter(e=>e.id!==result.id)].sort((a,b)=>b.id-a.id);render();
      refreshSummary().catch(()=>{});
    }catch(error){
      saved.textContent=`Save failed · ${error.message}`;
      if(attempts<3)setTimeout(persist,1000*attempts);
      else {const retry=element("button","Retry saving");retry.onclick=()=>{retry.remove();attempts=0;persist();};saved.append(retry);}
    }
  }
  persist();
}

function playbackTick(mediaTime) {
  document.querySelector("#playback-time").textContent=offsetLabel(mediaTime);
  if(!playbackSession)return;
  paintDetections(mediaTime);
  if(monitor.seeking)return;
  for(const event of playbackSession.clock.advance(mediaTime,!monitor.paused&&!monitor.ended))showPlaybackAlert(event,playbackSession);
}
function schedulePlaybackFrame() {
  if(frameRequest!==null||fallbackRequest!==null||monitor.paused)return;
  if(monitor.requestVideoFrameCallback){frameRequest=monitor.requestVideoFrameCallback((_now,metadata)=>{frameRequest=null;playbackTick(metadata.mediaTime);schedulePlaybackFrame();});}
  else {fallbackRequest=requestAnimationFrame(()=>{fallbackRequest=null;playbackTick(monitor.currentTime);schedulePlaybackFrame();});}
}
monitor.addEventListener("play",schedulePlaybackFrame);
monitor.addEventListener("play",()=>{if(selectedCamera)document.querySelector("#recording-info").textContent=`Playing ${nodeName(selectedCamera)} · ${playbackSession?"alerts follow the video clock":"raw recording"}`;});
monitor.addEventListener("seeking",()=>{
  clearPlaybackAlerts();
  overlay.getContext("2d").clearRect(0,0,overlay.width,overlay.height);
});
monitor.addEventListener("timeupdate",()=>{if(!monitor.requestVideoFrameCallback)playbackTick(monitor.currentTime);});
monitor.addEventListener("seeked",()=>{playbackSession?.clock.seek(monitor.currentTime);paintDetections(monitor.currentTime);if(!monitor.paused)schedulePlaybackFrame();});
monitor.addEventListener("pause",()=>paintDetections(monitor.currentTime));
window.addEventListener("resize",()=>paintDetections(monitor.currentTime));

document.querySelector("#next-event").addEventListener("click",()=>{
  if(!playbackSession)return;
  const upcoming=playbackSession.clock.events.find(e=>e.video_ts_s>monitor.currentTime+.2)||playbackSession.clock.events[0];
  if(!upcoming)return;
  monitor.currentTime=Math.max(0,upcoming.video_ts_s-2);
  monitor.play().catch(()=>{});
});
