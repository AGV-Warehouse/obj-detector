import pyrealsense2 as rs
import numpy as np
import cv2

pipeline=rs.pipeline()
config=rs.config()

config.enable_stream(rs.stream.color,640,480,rs.format.bgr8,30)
config.enable_stream(rs.stream.depth,640,480,rs.format.z16,30)

pipeline.start(config)

align=rs.align(rs.stream.color)

dist=1

print("starting")

try:
	while True:
		frames=pipeline.wait_for_frames()
		frames=align.process(frames)
		color_frame=frames.get_color_frame()
		depth_frame=frames.get_depth_frame()
		if not color_frame or not depth_frame:
			continue
		color=np.asanyarray(color_frame.get_data())
		depth=np.asanyarray(depth_frame.get_data())

		depth_m=depth*depth_frame.get_units()
		mask=(depth_m>0)&(depth_m<dist)
		mask=(mask*255).astype(np.uint8)

		kernel=np.ones((5,5),np.uint8)
		mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE, kernel)
		mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN, kernel)
		contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

		for cnt in contours:
			area=cv2.contourArea(cnt)
			if area<500:
				continue

			x,y,w,h=cv2.boundingRect(cnt)
			cv2.rectangle(color,(x,y),(x+w,y+h),(0,255,0),2)

			cx=x+w//2
			cy=y+h//2
			dist=depth_frame.get_distance(cx,cy)

			cv2.putText(color,f"{dist:.2f}m", (x,y-10), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)

		cv2.imshow("mask",mask)
		cv2.imshow("result",color)

		if cv2.waitKey(1)==27:
			break

finally:
	pipeline.stop()
	cv2.destroyAllWindows()
