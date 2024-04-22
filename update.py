import hydra
import torch
import cv2
import csv
import os
from datetime import datetime
from random import randint
from sort import *
from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.utils import DEFAULT_CONFIG, ROOT, ops
from ultralytics.yolo.utils.checks import check_imgsz
from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box
vehicle_speeds = {}
vehicle_avg_speeds = {}

tracker = None


# Intersection lines setup
lines = {
    'horizontal_front': (1300, 700, 2500, 700),
    'horizontal_back': (1300, 1500, 2500, 1500),
    'vertical_left': (1300, 750, 1300, 900),
    'vertical_right': (2500, 700, 2500, 1500)
}
line_crossings = {key: 0 for key in lines.keys()}



def compute_speed(tracker, fps, scale_factor):
    """
    Compute speed based on the displacement of the centroid and the given scale factor.

    :param tracker: Instance of the tracker.
    :param fps: Frames per second of the video.
    :param scale_factor: Conversion factor from pixel to real-world units (e.g., meters/pixel).
    :return: Speed in km/h.
    """
    if len(tracker.centroidarr) < 2:
        return 0

    x1, y1 = tracker.centroidarr[-1]
    x2, y2 = tracker.centroidarr[-2]
    dist_pixels = ((x2 - x1)**2 + (y2 - y1)**2)**0.5
    dist_meters = dist_pixels * scale_factor
    speed_kmh = (dist_meters * fps) * 3.6

    return speed_kmh

def resize_image(img, width, height):
    orig_height, orig_width = img.shape[:2]
    ratio_w = width / orig_width
    ratio_h = height / orig_height
    ratio = min(ratio_w, ratio_h)
    new_width = int(orig_width * ratio)
    new_height = int(orig_height * ratio)
    resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    return resized_img

def init_tracker():
    global tracker
    
    sort_max_age = 5 
    sort_min_hits = 2
    sort_iou_thresh = 0.2
    tracker =Sort(max_age=sort_max_age,min_hits=sort_min_hits,iou_threshold=sort_iou_thresh)

rand_color_list = []


def line_intersect(x1, y1, x2, y2, x3, y3, x4, y4):
    """ Check if line (x1, y1) to (x2, y2) intersects with line (x3, y3) to (x4, y4) """
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if den == 0:
        return False  # lines are parallel
    
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / den
    
    if 0 <= t <= 1 and 0 <= u <= 1:
        # Calculate the exact point of intersection
        intersect_x = x1 + t * (x2 - x1)
        intersect_y = y1 + t * (y2 - y1)
        # Return True if it's a valid crossing point
        return True
    return False

def check_line_crossing(track, line_name, line):
    """ Check and count line crossing with direction validation """
    if len(track.centroidarr) < 2:
        return False  # Not enough points to determine crossing
    
    x1, y1 = track.centroidarr[-2]
    x2, y2 = track.centroidarr[-1]
    line_x1, line_y1, line_x2, line_y2 = line
    
    if line_intersect(x1, y1, x2, y2, line_x1, line_y1, line_x2, line_y2):
        if line_name.startswith('horizontal'):
            return (y2 - y1) > 0 if line_name.endswith('front') else (y2 - y1) < 0
        elif line_name.startswith('vertical'):
            return (x2 - x1) > 0 if line_name.endswith('left') else (x2 - x1) < 0
    return False

def update_line_crossings(tracks, lines, line_crossings):
    for track in tracks:
        if len(track.centroidarr) >= 2:  # Ensure there are enough points to check crossing
            for line_name, line in lines.items():
                if check_line_crossing(track, line_name, line):
                    line_crossings[line_name] += 1

                
def draw_boxes(img, bbox, identities=None, categories=None, names=None, offset=(0, 0), tracks=None, fps=60, scale_factor=0.01, lines=None, line_crossings=None):
    
    # Define a list of visually appealing colors
    color_palette = [(255, 69, 0), (255, 165, 0), (255, 215, 0), (0, 128, 0), (0, 0, 255), (138, 43, 226), (128, 0, 128), (255, 20, 147), (0, 255, 255), (255, 255, 0)]
    # You can add more colors if needed
    
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        id = int(identities[i]) if identities is not None else 0
        class_name = names[int(categories[i])] if categories is not None and names is not None else ''
        box_center = (int((box[0]+box[2])/2),(int((box[1]+box[3])/2)))
        label = f"{class_name} - {id}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 1)
        color = color_palette[int(categories[i]) % len(color_palette)] if categories is not None and len(color_palette) > 0 else (255, 255, 255)
        # Adjust the thickness of the rectangle
        thickness = 2
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        cv2.rectangle(img, (x1, y1 - 30), (x1 + w, y1), color, -1)
        cv2.putText(img, class_name, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
        # if tracks is not None:
        #     track = tracks[i]
        #     speed_kmh = compute_speed(track, fps, scale_factor)
        #     speed_text = f"{speed_kmh:.2f}km/h"
        #     cv2.putText(img, speed_text, (x2 + 5, y2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        # Draw lines and display crossing counts
        line_color = (0, 0, 255)  # Green for lines
        text_color = (0, 255, 0)  # White for text
        for line_name, line in lines.items():
            cv2.line(img, (line[0], line[1]), (line[2], line[3]), line_color, 8)
            midpoint = ((line[0] + line[2]) // 2, (line[1] + line[3]) // 2)
            cv2.putText(img, f"{line_crossings[line_name]}", (midpoint[0] + 10, midpoint[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 4, text_color, 8)
    
    return img


def random_color_list():
    global rand_color_list
    rand_color_list = []
    for i in range(0,5005):
        r = randint(0, 255)
        g = randint(0, 255)
        b = randint(0, 255)
        rand_color = (r, g, b)
        rand_color_list.append(rand_color)
    #......................................
        


class DetectionPredictor(BasePredictor):
    
    def get_annotator(self, img):
        return Annotator(img, line_width=self.args.line_thickness, example=str(self.model.names))

    def preprocess(self, img):
        img = torch.from_numpy(img).to(self.model.device)
        img = img.half() if self.model.fp16 else img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.0
        return img

    def postprocess(self, preds, img, orig_img):
        preds = ops.non_max_suppression(preds,
                                        self.args.conf,
                                        self.args.iou,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det)

        for i, pred in enumerate(preds):
            shape = orig_img[i].shape if self.webcam else orig_img.shape
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], shape).round()

        return preds

    def write_results(self, idx, preds, batch):

        p, im, im0 = batch
        log_string = ""
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        self.seen += 1
        im0 = im0.copy()
        if self.webcam:  # batch_size >= 1
            log_string += f'{idx}: '
            frame = self.dataset.count
        else:
            frame = getattr(self.dataset, 'frame', 0)
        # tracker
        self.data_path = p
    
        save_path = str(self.save_dir / p.name)  # im.jpg
        self.txt_path = str(self.save_dir / 'labels' / p.stem) + ('' if self.dataset.mode == 'image' else f'_{frame}')
        log_string += '%gx%g ' % im.shape[2:]  # print string
        self.annotator = self.get_annotator(im0)
        
        det = preds[idx]
        self.all_outputs.append(det)
        if len(det) == 0:
            return log_string
        for c in det[:, 5].unique():
            n = (det[:, 5] == c).sum()  # detections per class
            log_string += f"{n} {self.model.names[int(c)]}{'s' * (n > 1)}, "
    
    
        # #..................USE TRACK FUNCTION....................
        dets_to_sort = np.empty((0,6))
        
        for x1,y1,x2,y2,conf,detclass in det.cpu().detach().numpy():
            dets_to_sort = np.vstack((dets_to_sort, 
                        np.array([x1, y1, x2, y2, conf, detclass])))
        
        tracked_dets = tracker.update(dets_to_sort)
        tracks =tracker.getTrackers()
        update_line_crossings(tracks, lines, line_crossings)
        
        for track in tracks:
            speed_kmh = compute_speed(track, fps=30, scale_factor=0.01)
            if track.id not in vehicle_speeds:
                vehicle_speeds[track.id] = []
            vehicle_speeds[track.id].append(speed_kmh)
            [cv2.line(im0, (int(track.centroidarr[i][0]),
                        int(track.centroidarr[i][1])), 
                        (int(track.centroidarr[i+1][0]),
                        int(track.centroidarr[i+1][1])),
                        rand_color_list[track.id], thickness=3) 
                        for i,_ in  enumerate(track.centroidarr) 
                            if i < len(track.centroidarr)-1 ] 
        

        if len(tracked_dets)>0:
            bbox_xyxy = tracked_dets[:,:4]
            identities = tracked_dets[:, 8]
            categories = tracked_dets[:, 4]
            # draw_boxes(im0, bbox_xyxy, identities, categories, self.model.names)
            YOUR_VIDEO_FPS = 60  # Change this to your actual FPS
            SCALE_FACTOR = 0.01  # Change this to your real-world scale (e.g., meters/pixel)
            draw_boxes(im0, tracked_dets[:,:4], identities=tracked_dets[:, 8], categories=tracked_dets[:, 4], names=self.model.names, lines=lines, line_crossings=line_crossings, fps=YOUR_VIDEO_FPS, scale_factor=SCALE_FACTOR)
            
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        return log_string

def write_avg_speeds_to_csv(output_dir, vehicle_avg_speeds):
    now = datetime.now()
    time_stamp = now.strftime('%Y-%m-%d_%H-%M-%S')
    filename = os.path.join(output_dir, f'average_speeds_{time_stamp}.csv')
    
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['Vehicle ID', 'Average Speed (km/h)']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for vehicle_id, avg_speed in vehicle_avg_speeds.items():
            writer.writerow({'Vehicle ID': vehicle_id, 'Average Speed (km/h)': avg_speed})

            
@hydra.main(version_base=None, config_path=str(DEFAULT_CONFIG.parent), config_name=DEFAULT_CONFIG.name)
def predict(cfg):
    init_tracker()
    random_color_list()
    
    cfg.model = cfg.model or "yolov8n.pt"
    cfg.imgsz = check_imgsz(cfg.imgsz, min_dim=2)
    cfg.source = cfg.source if cfg.source is not None else ROOT / "assets"
    predictor = DetectionPredictor(cfg)
    predictor()

    # Compute average speeds for each vehicle
    for vehicle_id, speeds in vehicle_speeds.items():
        vehicle_avg_speeds[vehicle_id] = sum(speeds) / len(speeds)

    # Assuming the logs are saved in a 'logs' folder within the current run's directory
    output_dir = os.path.join(os.getcwd(), "logs")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    write_avg_speeds_to_csv(output_dir, vehicle_avg_speeds)

if __name__ == "__main__":
    predict()
