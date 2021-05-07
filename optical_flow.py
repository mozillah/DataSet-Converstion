import numpy as np
import cv2
import time
import argparse
from datetime import datetime,timedelta
help_message = '''
USAGE: optical_flow.py [<video_source>]
Keys:
 1 - toggle HSV flow visualization
 2 - toggle glitch
'''
count = 0
def draw_flow(img, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (x2, y2) in lines:      
	    cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis

def draw_hsv(flow):
    h, w = flow.shape[:2]
    fx, fy = flow[:,:,0], flow[:,:,1]
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx*fx+fy*fy)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[...,0] = ang*(180/np.pi/2)
    hsv[...,1] = 255
    hsv[...,2] = np.minimum(v*4, 255)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr

def save_recording(video_name_,fps,width,height):
    now = datetime.now()
    video_name = video_name_[:-4]+str(now.year) + str(format(now.month,'02d')) + str(format(now.day,'02d')) + str(format(now.hour,'02d')) + str(format(now.minute,'02d')) + str(format(now.second,'02d')) + ".mp4"  
    out = cv2.VideoWriter(video_name,cv2.VideoWriter_fourcc('M','J','P','G'), fps, (width,height))
    return out

def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow = -flow
    flow[:,:,0] += np.arange(w)
    flow[:,:,1] += np.arange(h)[:,np.newaxis]
    res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
    return res

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", help="path to the video file")
    ap.add_argument("-a", "--min-area", type=int, default=10, help="minimum area size")
    ap.add_argument("-f", "--frames-length", type=int, default=1000, help="frame array length")
    ap.add_argument("-t", "--timespan", type=int, default=60, help="timespan to record video")
    
    args = vars(ap.parse_args())
    # if the video argument is None, then we are reading from webcam
    if args.get("video", None) is None:
        cam = VideoStream(src=0).start()
        time.sleep(2.0)
    # otherwise, we are reading from a video file
    else:
        cam = cv2.VideoCapture(args["video"])

    frame_width = int(cam.get(3))
    frame_height = int(cam.get(4))
    fps = cam.get(cv2.CAP_PROP_FPS)
    codec=cam.get(cv2.CAP_PROP_FOURCC)
    print(f"fps =={fps} , codec=={codec}")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    print(fourcc)
    print(f"fourcc=={codec}")
    output_video = str(int(time.time()))+"test.mp4"
    if args["video"][-4:] != ".smp":
        output_video = args["video"][:-4]+'finaltrack_output.mp4'
    print(output_video)
    out=save_recording(video_name_=output_video,fps=fps,width=frame_width,height=frame_height)
    # out = cv2.VideoWriter(output_video,cv2.VideoWriter_fourcc('M','J','P','G'), fps,(frame_width, frame_height))
    # cam = cv2.VideoCapture(fn)
    ret, prev = cam.read()
    show_hsv = True
    show_glitch = True
    frameCount = 0
    if ret:
        prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
        cur_glitch = prev.copy()
    else:
        blank_image = np.zeros((frame_height,frame_width,3), np.uint8)
        prevgray = cv2.cvtColor(blank_image, cv2.COLOR_BGR2GRAY)
        cur_glitch = prev.copy()
        
    while True:
        ret, img = cam.read()
        if ret:
            vis = img.copy()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prevgray, gray,None, 0.5, 5, 15, 3, 5, 1.1, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
            prevgray = gray
            cv2.imshow('flow', draw_flow(gray, flow))
            if show_hsv:
                gray1 = cv2.cvtColor(draw_hsv(flow), cv2.COLOR_BGR2GRAY)
                thresh = cv2.threshold(gray1, 25, 255, cv2.THRESH_BINARY)[1]
                thresh = cv2.dilate(thresh, None, iterations=2)
                (cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                # loop over the contours
                for c in cnts:
                # if the contour is too small, ignore it
                    (x, y, w, h) = cv2.boundingRect(c)
                    if cv2.contourArea(c) < args["min_area"]:
                        continue
                    # if w > 100 and h > 100 and w < 900 and h < 680:
                    cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 4)
                    cv2.putText(vis,str(time.time()), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),1)
                    # text = "Occupied"
                    timestamp = datetime.now()
                # cv2.imshow('Image', vis)
                # cv2.imshow('Image2', draw_hsv(flow))
                # if show_glitch:
                #     cur_glitch = warp_flow(cur_glitch, flow)
                #     cv2.imshow('glitch', cur_glitch)
            ch = 0xFF & cv2.waitKey(5)
            try:
                if ((datetime.now() - timestamp).seconds < args["timespan"]):
                    print(timestamp)
                    frameCount+=1
                    print(frameCount)
                    out.write(img)
            except NameError:
                print("Ignore")
                #  a = 10 # nope
        if frameCount > args["frames_length"] or not ret:
            out.release()
            cv2.waitKey(1000)
            out=save_recording(video_name_=output_video,fps=fps,width=frame_width,height=frame_height)
            frameCount = 0
            if not ret:
                exit()
        if ch == 27:
            break
    
    out.release()  
    cv2.destroyAllWindows() 		
