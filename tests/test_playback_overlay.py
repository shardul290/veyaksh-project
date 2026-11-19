import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from m3_events.playback import draw_detection, merge_display_tracks, render_video


class OverlayTests(unittest.TestCase):
    def test_unassigned_detections_stay_visible_without_a_path_claim(self):
        untracked={"frame":0,"ts":0,"track_id":None,"class":"pedestrian","bbox":[1,2,3,4]}
        tracked={"frame":0,"ts":0,"track_id":7,"class":"pedestrian","bbox":[5,6,7,8]}
        decision={**tracked,"path":{"status":"right_path","zone_id":"FOOT","reason":"on_footpath"}}
        rows=merge_display_tracks([untracked,tracked],[decision])
        self.assertEqual(len(rows),2)
        self.assertEqual(rows[0]["path"]["status"],"unknown")
        self.assertEqual(rows[0]["path"]["reason"],"untracked_detection")
        self.assertEqual(rows[1]["path"]["status"],"right_path")

    def test_pedestrian_green_and_unsafe_red_with_thin_outline(self):
        row={"track_id":3,"class":"pedestrian","bbox":[40,50,100,110]}
        safe=np.full((120,160,3),200,dtype=np.uint8)
        draw_detection(safe,row)
        self.assertGreater(int(safe[80,40,1]),int(safe[80,40,2]))
        # The outline never fills the subject or the scene behind it.
        np.testing.assert_array_equal(safe[80,60],[200,200,200])
        wrong=np.full((120,160,3),200,dtype=np.uint8)
        draw_detection(wrong,row,alert=True)
        self.assertGreater(int(wrong[80,40,2]),int(wrong[80,40,1]))
        np.testing.assert_array_equal(wrong[80,60],[200,200,200])

    def test_render_preserves_every_source_frame_and_final_rule_color(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/"source.mp4";output=root/"overlay.mp4"
            writer=cv2.VideoWriter(str(source),cv2.VideoWriter_fourcc(*"mp4v"),10,(160,120))
            for _ in range(12):writer.write(np.full((120,160,3),200,dtype=np.uint8))
            writer.release()
            tracks=root/"tracks.jsonl";events=root/"events.jsonl"
            rows=[{"ts":i/10,"frame":i,"track_id":3,"class":"pedestrian","bbox":[40,50,100,110]} for i in range(12)]
            tracks.write_text("".join(json.dumps(r)+"\n" for r in rows))
            events.write_text(json.dumps({"track_id":3,"video_ts_s":.6})+"\n")
            render_video(source,tracks,events,output)
            capture=cv2.VideoCapture(str(output));frames=[]
            while True:
                ok,frame=capture.read()
                if not ok:break
                frames.append(frame)
            capture.release()
            self.assertEqual(len(frames),12)
            self.assertGreater(int(frames[2][80,40,1]),int(frames[2][80,40,2]))
            self.assertGreater(int(frames[8][80,40,2]),int(frames[8][80,40,1]))
            self.assertFalse((root/"overlay.rendering.mp4").exists())


if __name__=="__main__":unittest.main()
