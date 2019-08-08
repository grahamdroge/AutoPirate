# Document handling tool to use on common documents
# used around Red Berry Innovations

# import libraries
import cv2
import sys
from PIL import Image, ImageOps
import imutils
from typing import List, Dict, Tuple
import numpy as np
import matplotlib.pyplot as plt
import datetime
import argparse
import logging
from shapedetector import ShapeDetector
import pytesseract
import re
from pdf2image import convert_from_path
from math import *
import os
from skimage.measure import compare_mse
from skimage.transform import probabilistic_hough_line
import pickle

from py2neo import Node, Relationship, Graph
graph = Graph("bolt://localhost:7687",auth=("neo4j","redberry"))
graph.delete_all()


# Mapping from image text to
# color masks
Line_Map = {
    "ORG":    [105,255,255],
    "BLK":    [0,0,0],
    "VIO":    [150,255,255],
    "GRN":    [67,255,167],
    "BLU":    [0,0,255],
    "GRY":    [0,0,187],
    "BRN":    [98,255,128],
    "WHT":    [0,0,84],
    "RED":    [120,255,255],
    "YEL":    [54,95,67],
}


# Output annotations
LINES = List[int]
LINE_COLORS = List[int]
CONTOUR = np.ndarray
CONTOURS = List[CONTOUR]
CENTERS = List[int]
IMAGE = np.ndarray
META  = Dict[str, str]
TEXT_MAP = Dict[str, list]
RECT = List[Tuple]


# ALLDATA Wiring diagram class
class ADWiringDiagram():

    def __init__(self, filename: str = "", border: bool = False):

        # Check filename argument
        if filename == "":
            filename = input("Enter diagram path: ")

        self.filename       = filename
        self.image_format   = filename.split(".")[-1]

        # Open image file
        if self.image_format.lower() == "pdf":
            self.orig_image = convert_from_path(filename)[0]
        else:
            self.orig_image = Image.open(filename)


        # Initialize image variables for RGB,GRAY and
        # other image characteristics
        self.image_rgb      = self.orig_image.convert('RGB')
        self.image_data     = np.array(self.image_rgb)
        self.temp_image     = self.image_data
        self.image_gray     = cv2.cvtColor(self.image_data, cv2.COLOR_BGR2GRAY)

        self.image_shape    = self.image_data.shape
        self.border         = None
        self.border_image   = np.array([], np.uint8)

        self.circle_centers = []

        # If border argument given then trim off border
        # and outside pixels
        if border:
            self.trim_meta()

    def bw_filter(self, img: IMAGE, rtype: str = "gray") -> IMAGE:
        # Function that returns a thresholded image where the
        # threshold is black and white only
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, thresh = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY_INV)

        if rtype == "rgb":
            rgb = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
            return rgb
        elif rtype == "gray":
            return thresh

    def color_filter(self, img: IMAGE, color: str, rtype: str = "gray") -> IMAGE:
        # Function that filters out the pixels that match a specfic color
        # range.  Uses hsv color space for simpler color range masking

        if color == "ECU":
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.inRange(gray,246,248)
            return gray
        else:
            hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            # Compute ranges
            offset = np.array([10,0,0])
            ref = np.array(Line_Map[color])

            lower = ref - offset
            upper = ref + offset

            # Apply masks to hsv images
            mask = cv2.inRange(hsv_img, lower, upper)
            res = cv2.bitwise_and(hsv_img, hsv_img, mask=mask)

            if rtype == "rgb":
                return cv2.cvtColor(res, cv2.COLOR_HSV2BGR)
            elif rtype == "gray":
                h, s, gray = cv2.split(res)
                return gray


    def trim_meta(self) -> None:

        # Retreive black/white image and calculate edges
        bw_img = self.bw_filter(self.image_gray,rtype = "gray")
        bw_img = cv2.bilateralFilter(bw_img, 11, 17, 17)
        edged = cv2.Canny(bw_img, 30, 200)

        # Find the contours, parse and sort them
        cnts = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]

        for c in cnts:
            # approximate the contour
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.015 * peri, True)

            # if our approximated contour has four points, then
            # we can assume that we have found our border
            if len(approx) == 4:
                self.border = approx
                break

        # Calculate region boundries and region itself
        upper_left = approx[0][0] + np.array([0,5])
        bottom_right = approx[2][0] + np.array([0,-5])
        region = np.append(upper_left, bottom_right)

        # Grab only the inside of the border
        self.border_image = np.array(self.image_rgb.crop(region))


    def get_lines(self, img: IMAGE) -> (LINES, LINES, list):
        # Function that grabs all the lines from
        # the image
        Threshold1 = 150
        Threshold2 = 350
        FilterSize = 3

        space_threshold = 25
        filt_indices = []

        # Erode smaller lines
        kernel = np.ones((5,5), np.uint8)
        erode = cv2.erode(img, kernel, iterations=1)

        # # Dilate back to thicken the lines
        # dilate = cv2.dilate(erode, kernel, iterations=2)

        # # bilateral filter to sharpen edges
        # median = cv2.bilateralFilter(dilate,9,75,75)

        # Computes edges
        # edges = cv2.Canny(img, Threshold1, Threshold2, FilterSize)

        # Grabs the lines
        # lines = cv2.HoughLinesP(edges,rho = 1,theta = 1*np.pi/180,threshold = 100,minLineLength = 100,maxLineGap = 50)
        temp_lines = probabilistic_hough_line(erode,threshold=100,line_length=100,line_gap=50)
        lines = []
        for l in temp_lines:
            lines.append( np.array(l).reshape((1,4)).astype(np.int32) )
        lines = np.array(lines)

        # Loop through lines and discard those that represent the other
        # side of image line
        for i in range(len(lines) - 1):
            start = tuple(lines[i].ravel()[0:2])
            end = tuple(lines[i].ravel()[2:4])

            deg = line2_angle( (start,end) )

            if i in filt_indices:
                continue

            for i2 in range(len(lines) - 1):
                if i2 == i or i2 in filt_indices:
                    continue

                start_ = tuple(lines[i2].ravel()[0:2])
                end_ = tuple(lines[i2].ravel()[2:4])

                deg2 = line2_angle( (start_,end_) )

                deg_diff = abs( deg - deg2 )
                if ( ( (start[0] - start_[0])**2 + (start[1] - start_[1])**2 )**(1/2) ) < space_threshold and deg_diff < 10:
                    filt_indices.append(i2)
                    continue

        filt_lines = np.delete(lines, filt_indices, 0)

        def centerScore(elem):
            start, end = elem.ravel().reshape((2,2))
            xoff, yoff = ( (end - start) / 2 ).astype(int)
            center = ( start + np.array([xoff,yoff]) )
            cscore = center.sum()
            return cscore
            # return elem.ravel()[0]

        temp_lines = sorted(filt_lines, key=centerScore)
        thresh = 60

        temp_indices = []
        for index in range(len(temp_lines) - 4):
            for i in [1,2,3,4]:
                total_diff1 = np.abs( (temp_lines[index] - temp_lines[index+i]) ).sum()
                if total_diff1 < thresh:
                    temp_indices.append(index)

        final_lines = np.delete(temp_lines,temp_indices, 0)
        return lines, final_lines

    def get_contours(self, img: IMAGE) -> (CONTOURS, CENTERS):
        # Given an image, this function gets all of the contours
        # and returns an image of just the contours, the contours
        # themselves and the center of the contours
        ret, thresh = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY_INV)
        thresh = cv2.medianBlur(thresh, 9)
        edged = cv2.Canny(thresh, 30, 200)

        # Find the contours, parse and sort them
        cnts = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key = cv2.contourArea, reverse = True)

        centers = []
        approx_cnts = []
        # blank = np.zeros(img.shape, np.uint8)

        # Loop over contours
        for c in cnts:
            # Get center of contour
            M = cv2.moments(c)
            if M["m00"] == 0:
                continue
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            # Draw middle of contour
            centers.append((cX, cY))
            # cv2.circle(blank, (cX, cY), 7, (255, 255, 255), -1)

            # Approximate the contours
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.015 * peri, True)
            approx_cnts.append(approx)

        # Draw the contour
        # cv2.drawContours(blank, cnts, -1, (255, 0, 0), 3)

        return (approx_cnts, centers)


    def plot_contour(self, img: IMAGE, cnt: CONTOUR) -> IMAGE:
        # Function to plot individual contour and its
        # center
        blank = np.zeros(img.shape, np.uint8)

        # M = cv2.moments(cnt)
        # cX = int(M["m10"] / M["m00"])
        # cY = int(M["m01"] / M["m00"])

        # center = (cX, cY)
        # cv2.circle(blank, center, 7, (255, 255, 255), -1)
        cv2.drawContours(blank, [cnt], -1, (255,0,0), 3)

        return blank


    def get_bound_rects(self,cnts: CONTOURS) -> (RECT):
        # Function that takes in contours and calculates the
        # corresponding bounding rectangle
        # temp = img.copy()
        params = []

        for c in cnts:
            x,y,w,h = cv2.boundingRect(c)
            params.append((x,y,w,h))
            # cv2.rectangle(temp, (x,y), (x+w, y+h), (0,0,255), 2)

        return params


    def clear_contours(self, img: IMAGE, contours: CONTOURS) -> IMAGE:
        # Given a list of contours, this function clears them from
        # the image and returns the new image
        temp = img.copy()

        for c in contours:
            cv2.fillPoly(temp, pts=[c], color=(255,255,255))

        return temp

    def plot_lines(self,lines: LINES, se: bool = False) -> IMAGE:
        # Function that takes lines for get_lines and
        # plots them on a blank image
        blank = np.zeros(self.image_shape, np.uint8)

        for i in range(lines.shape[0]):
            x1, y1, x2, y2 = lines[i][0]
            cv2.line(blank, (x1, y1), (x2, y2), (255, 255, 255), 3)
            # cv2.putText(blank, "{}".format(i), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)
            if se:
                cv2.circle(blank, (x1, y1), 7, (255, 0, 0), -1)
                cv2.circle(blank, (x2, y2), 7, (0, 255, 0), -1)


        return blank

    def extract_text(self, img: IMAGE) -> (TEXT_MAP,LINE_COLORS):
        # Function to extract text boxes, line colors and decipher
        # text using ocr engine

        # Grab digit images and load them into array
        # Compute and store histograms for each image
        # num_digits = len([name for name in os.listdir('digits') if "image_" in name]) + 1
        # digits = [np.array(Image.open("digits/image_"+str(i)).convert('L')) for i in range(1,num_digits)]
        digits = pickle.load(open("digits.p",'rb'))

        # Shape detector to tell the shape of contour
        sd = ShapeDetector()

        # Filter out ecu boxes
        ecu_img = self.color_filter(img, color="ECU", rtype='rgb')
        ecu_img = cv2.medianBlur(ecu_img, 5)

        # Enlarge the boxes so we can overwrite the dashed or
        # solid lines
        kernel = np.ones((9,9), np.uint8)
        _ecu_img = cv2.dilate(ecu_img, kernel, iterations=3)

        # Grab the contours and then fill them with white to
        # erase them from the image
        ecu_contours,centers = self.get_contours(_ecu_img)
        no_boxes = self.clear_contours(img, ecu_contours)
        ecu_rects = self.get_bound_rects(ecu_contours)

        # Convert to gray and filter out the colored wires
        # so it is just text
        # no_boxes = cv2.cvtColor(no_boxes, cv2.COLOR_BGR2GRAY)
        just_text = self.bw_filter(no_boxes, rtype='rgb')

        # Add circle boxes for later relationship building
        kernel2 = np.ones((5,5), np.uint8)
        erode = cv2.erode(just_text, kernel2, iterations=1)
        median = cv2.medianBlur(erode, 5)
        temp_contours, circle_centers = self.get_contours(median)
        circle_centers = list(set(circle_centers))

        # Loop through and grab only circle contours
        circle_centers = []
        for c in temp_contours:
            if sd.detect(c) == "circle":
                M = cv2.moments(c)
                if M["m00"] == 0:
                    continue
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                circle_centers.append((cX,cY))

        self.circle_centers = circle_centers

        # Dilate text to get a rectangular contour
        # (Thanksgiving Approach)
        enlarged = cv2.dilate(just_text, kernel, iterations=5)

        # Grab text contours and convert to rectangular
        # boundries
        text_contours, centers = self.get_contours(enlarged)
        params = self.get_bound_rects(text_contours)

        # Decipher/Filter text and store mappings to
        # center and bounding rectangle
        line_colors = []
        ecu_map = {}
        decoded_text = []
        brgb = Image.fromarray(img, 'RGB')
        for i, _roi in enumerate(params):

            roi = list(_roi)
            roi[2] += roi[0]
            roi[3] += roi[1]
            roi = tuple(roi)
            ar = float(_roi[2]) / _roi[3]

            text_roi = brgb.crop(roi)
            text = pytesseract.image_to_string(text_roi)

            if lc_check(text):
                temp = text.split('/')
                if temp[0] == "BLK":
                    c = temp[1]
                else:
                    c = temp[0]

                line_colors.append(c)
            elif len(text) == 0 and (.65 < ar < 1.25):
                text_roi = text_roi.resize((100,80)).convert('L')

                best_score = 100000
                best_index = 0
                for i2,d in enumerate(digits):
                    mse = compare_mse(d,np.array(text_roi))
                    if mse < best_score:
                        best_score = mse
                        best_index = i2 + 1
                
                if _roi[0] < (200):
                    ecu_map[i] = (_roi, "IN-"+str(best_index))
                elif _roi[0] > (img.shape[1] - 200):
                    ecu_map[i] = (_roi, "OUT-"+str(best_index))

            else:
                # Check if text passes filter test
                if filter_text(text) != "":
                    if text in decoded_text:
                        continue
                    else:
                        decoded_text.append(text)

                    (closest_ecu, text_near) = text2_ecu(ecu_rects, _roi, text)
                    if not any(closest_ecu):
                        pass
                    else:
                        ecu_map[i] = (closest_ecu, text_near)

        return ecu_map, list(set(line_colors))


    def build_relationships(self, img: IMAGE, mapping: Dict) -> Dict:
        # Function that takes in wire lines and builds the
        # relationship beween ecus/modules

        # Get line information
        scratch, lines = self.get_lines(img)

        # Get line image
        lines_plot = self.plot_lines(lines)

        # Find corners
        lines_plot = cv2.cvtColor(lines_plot, cv2.COLOR_BGR2GRAY)
        corners = cv2.goodFeaturesToTrack(lines_plot, 10000, 0.5, 40)

        # Add the black circle boxes to list of boxes
        boxes2 = corners2_boxes(np.array(self.circle_centers), circles=True)
        corners_filt = []
        for b in boxes2:
            for i,c in enumerate(corners):
                if is_inside(c.ravel(), b):
                     corners_filt.append(i)

        corners = np.delete(corners, corners_filt, 0)

        # Convert corners to bounding boxes
        boxes1 = corners2_boxes(corners)
        boxes = boxes1

        boxes.extend(boxes2)

        box_votes, line_votes = voting(lines, boxes)
        start_lines = get_starts(box_votes)

        used_indices = []
        concat_lines = []
        for line_index in start_lines:

            if line_index in used_indices:
                continue

            start, end = lines[line_index].ravel().reshape((2,2))

            b1, b2 = line_votes[line_index]
            b = b2 if len(box_votes[b1]) != 1 else b1
            if is_inside(tuple(start), boxes[b]):
                initial_start = tuple(start)
            else:
                initial_start = tuple(end)

            used_indices.append(line_index)
            box1, box2 = line_votes[line_index]
            last_box = box2 if len(box_votes[box1]) > 1 else box1

            last_line = None
            constructing = True

            multi_line_indices = []
            multi_line = False
            multi_line_box = None
            while constructing:
                line = lines[line_index].ravel().reshape((2,2))
                box1, box2 = line_votes[line_index]
                box = box2 if box1 == last_box else box1
                last_box = box

                if len(box_votes[box]) == 1:
                    # At the end of path
                    end_index, se = box_votes[box][0]
                    end = tuple(lines[end_index].reshape((2,2))[se])
                    if initial_start != end:
                        concat_lines.append( (initial_start,end) )

                    if multi_line:
                        if len(multi_line_indices) > 0:
                            last_line = line_index
                            line_index = multi_line_indices.pop(0)
                            used_indices.append(line_index)
                            last_box = multi_line_box
                        else:
                            constructing = False
                    else:
                        constructing = False

                elif len(box_votes[box]) > 2:
                    next_lines = box_votes[box]
                    for l in next_lines:
                        if l[0] != last_line:
                            multi_line_indices.append(l[0])
                    multi_line_indices = list(set(multi_line_indices))
                    last_line = line_index
                    line_index = multi_line_indices.pop(0)
                    used_indices.append(line_index)
                    multi_line = True
                    multi_line_box = box

                else:
                    next_lines = box_votes[box]
                    last_line = line_index
                    line_index = next_lines[0][0] if next_lines[1][0] == last_line else next_lines[1][0]
                    used_indices.append(line_index)


        # return concat_lines

        # Map relationships for starts and ends
        connections = {}

        for line in concat_lines:
            start, end = line
            start_text = False
            end_text = False
            for k,v in mapping.items():
                box = v[0]
                text = v[1]
                if "OUT" == text.split("-")[0] or "IN" == text.split("-")[0]:
                    exs = (80,30)
                else:
                    exs = (50,50)
                if is_connected(start, box, exs):
                    start_text = text
                elif is_connected(end, box, exs):
                    end_text = text

            if start_text and end_text:
                if start_text in connections.keys():
                    connections[start_text].append(end_text)
                else:
                    connections[start_text] = [end_text]
                if end_text in connections.keys():
                    connections[end_text].append(start_text)
                else:
                    connections[end_text] = [start_text]

        return connections

    def pipeline(type: str):
        # Function that performs a set of operations stopping
        # at the type CHECKPOINT

        if self.border_image.any():
            img = self.border_image.copy()
        else:
            img = self.image_rgb.copy()

        # Grab ECU mappings and the line colors in image
        ecu_map, line_colors = self.extract_text(img)

        # Loop over line colors and build relationships
        # for each
        last_connections = {}
        all_connections = {}
        for l in line_colors:
            if l in Line_Map.keys():
                cimg = self.color_filter(img,color=l,rtype="gray")
                connections = self.build_relationships(cimg,ecu_map)
                all_connections.update(connections)

        

        # Take all of the connections and perform any pre-processing
        # before storing in graph database
        # store_connections(all_connections)

#### HELPER FUNCTIONS ####

def store_connections(connections: dict) -> None:
    # Function to store the ecu connections in the
    # graph database

    for k,v in connections.items():
        key = k.replace("\n\n","\n").replace("  "," ")
        num_conn = len(v)

        if (graph.nodes.match("ECU", name=key).first() == None):
            graph.create(Node("ECU", name=key, connections=num_conn))

    used_nodes = []
    for k,v in connections.items():
        key = k.replace("\n\n","\n").replace("  "," ")
        node1 = graph.nodes.match("ECU", name=key).first()
        used_nodes.append(key)
        for n2 in v:
            end = n2.replace("\n\n","\n").replace("  "," ")
            if end in used_nodes:
                continue

            node2 = graph.nodes.match("ECU", name=end).first()
            graph.create(Relationship(node1,"CONNECTED-TO",node2))


def rC(rect,pt):
    logic = rect[0] < pt[0] < rect[0]+rect[2] and rect[1] < pt[1] < rect[1]+rect[3]
    return logic

def is_connected(point: tuple, box: tuple, exs: tuple) -> bool:
    # Checks whether the start or end point is connected
    # to a given ecu.

    # Build small bounding box
    ex_x = exs[0]         # expand coefficient
    ex_y = exs[1]
    x = point[0]
    y = point[1]

    tl_x = x - ex_x
    tl_y = y - ex_y
    tr_x = x + ex_x
    tr_y = y - ex_y
    bl_x = x - ex_x
    bl_y = y + ex_y
    br_x = x + ex_x
    br_y = y + ex_y

    if ( rC(box,(tl_x,tl_y)) or rC(box,(tr_x,tr_y)) or rC(box,(bl_x,bl_y)) or rC(box,(br_x,br_y))  ):
        return True
    else:
        return False



def text2_ecu(ecu_rects: tuple, text_rect: tuple, text: str) -> tuple:
    # Function that takes the text bounding rectangle and
    # maps that to the closest ecu contours.  A relationship
    # is formed between the two

    # Loop through both bounding rectangles
    best_index = None
    short_dist = 1000
    best_param = (0,0,0,0)
    threshold = 200

    # Get 4 midpoints for text rectangle
    top = (text_rect[0] + int(text_rect[2]/2), text_rect[1])
    left = (text_rect[0], text_rect[1] + int(text_rect[3]/2))
    bot = (text_rect[0] + int(text_rect[2]/2), text_rect[1] + text_rect[3])
    right = (text_rect[0] + text_rect[2], text_rect[1] + int(text_rect[3]/2))

    for i,er in enumerate(ecu_rects):
        # Get 4 midpoints of ecu rectangle
        points = [(er[0] + int(er[2]/2), er[1]),
        (er[0], er[1] + int(er[3]/2)),
        (er[0] + int(er[2]/2), er[1] + er[3]),
        (er[0] + er[2], er[1] + int(er[3]/2)) ]

        for p in points:
            # Calculate distance between points, check if any beat the shortest
            # distance. If so keep it
            dist1 = ( (top[0] - p[0])**2 + (top[1] - p[1])**2 )**(1/2)
            dist2 = ( (left[0] - p[0])**2 + (left[1] - p[1])**2 )**(1/2)
            dist3 = ( (bot[0] - p[0])**2 + (bot[1] - p[1])**2 )**(1/2)
            dist4 = ( (right[0] - p[0])**2 + (right[1] - p[1])**2 )**(1/2)

            shortest = min(dist1,dist2,dist3,dist4)

            if shortest < short_dist and shortest < threshold:
                best_index = i
                short_dist = shortest
                best_param = er

    if best_index == None:
        return (best_param,"")

    return (best_param, text)


def lc_check(text: str) -> bool:
    # Function that checks if text is part of line
    # color description
    if ( bool(re.match(r"[A-Z]{3}/[A-Z]{3}$", text)) or
         bool(re.match(r"[A-Z]{3}$",text)) or
         bool(re.match(r"[A-Z]{3}[/,\n]{2}[A-Z]{3}$", text)) or
         bool(re.match(r"[A-Z]{3}[\n, /]{2}[A-Z]{3}$", text)) ):
        return True
    else:
       return False

def show_image(img: IMAGE) -> IMAGE:
    # Function that displays the image
    # from a numpy array
    if len(img.shape) < 3:
        temp_image = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        temp_image = Image.fromarray(temp_image, 'RGB')
        temp_image.show()
    else:
        temp_image = Image.fromarray(img, 'RGB')
        temp_image.show()


def is_inside(point: tuple, area: np.ndarray) -> bool:
    # Function that determines if the point is inside
    # the given area
    x = point[0]
    y = point[1]

    area_rs = area.reshape((4,2))
    max_x, max_y = np.amax(area_rs, axis=0)
    min_x, min_y = np.amin(area_rs, axis=0)

    if ( max_x >= x >= min_x ) and ( max_y >= y >= min_y ):
        return True
    else:
        return False


def voting(lines: LINES, boxes: list) -> list:
    # Function that constructs a dictionary that that stores what lines
    # correspond to what boxes
    box_votes = {}
    line_votes = {}
    discard_lines = []

    for i,l in enumerate(lines):
        line_count = []
        start, end = l.ravel().reshape((2,2))
        s = False
        e = False
        for i2,b in enumerate(boxes):
            if is_inside(start, b) and not s:
                line_count.append(i2)
                s = True
            elif is_inside(end, b) and not e:
                line_count.append(i2)
                e = True
            if s and e:
                break
        if len(line_count) == 2:
            line_votes[i] = line_count
        else:
            discard_lines.append(i)


    for i,b in enumerate(boxes):
        box_count = []
        for i2,l in enumerate(lines):
            if i2 in discard_lines:
                continue
            start, end = l.ravel().reshape((2,2))
            if is_inside(start,b):
                box_count.append((i2,0))
            elif is_inside(end,b):
                box_count.append((i2,1))
        if len(box_count) != 0:
            box_votes[i] = box_count

    return (box_votes, line_votes)

def get_starts(box_votes: dict) -> dict:
    # Function that returns lines that can be considered starts
    # or ends, no corners.  Provides starting place to build lines
    se = []

    for k,v in box_votes.items():
        if len(v) == 1:
            line_index = v[0][0]
            se.append(line_index)

    return se


def filter_text(text: str) -> str:
    # Function that filters the text returned from
    # pytesseract.

    # Blacklisted characters that shouldnt be in
    # text
    blacklist = ["&","}","{","[","]"]

    if text == None:
        return ""

    if len(text) < 10:
        return ""


    # Compute simple statistics of text to filter
    # out unwanted text
    num_spaces = text.count(' ')
    num_nl = text.count('\n')
    space_ratio = num_spaces / len(text)

    accum_word = 0
    new_text = text.replace("\n\n","\n").replace(' ','\n')
    num_words = len(new_text.split("\n"))
    for n in new_text:
        accum_word += len(n)

    if (accum_word / num_words) <= 3:
        return ""

    if num_nl == 0:
        return ""

    nl_ratio = len(text) / num_nl

    if nl_ratio < 4:
        return ""

    # No lowercase letters or blacklisted characters
    for c in text:
        if c.islower():
            return ""
        elif c in blacklist:
            return ""

    # If text contains non-ascii characters then ignore
    if not is_ascii(text):
        return ""

    return text


def line2_angle(line: tuple) -> int:
    # Function that computes the angle from
    # a given line tuple ( (x1,y1), (x2,y2) )
    start = line[0]
    end = line[1]

    if ( end[0] - start[0] ) == 0:
        deg = 90
    else:
        slope = abs( ( end[1] - start[1] ) / ( end[0] - start[0] ) )
        deg = atan(slope) * 180/pi

    return deg


def corners2_boxes(corners: np.ndarray, circles: bool=False) -> list:
    # Function that takes in the corner x,y points
    # and converts them to 2D boxes.  Enables you to
    # follow contours and check if they meet a corner
    # area

    if circles:
        ex = 55
        corners = np.array(corners)
    else:
        ex = 23

    boxes = []            # Tuple of box coordinates.  top left-x,top left-y, width, height

    for c in corners:
        x,y = c.ravel()
        tl_x = x - ex
        tl_y = y - ex
        tr_x = x + ex
        tr_y = y - ex
        bl_x = x - ex
        bl_y = y + ex
        br_x = x + ex
        br_y = y + ex

        box =[
            [[ tl_x, tl_y ]],
            [[ tr_x, tr_y ]],
            [[ br_x, br_y ]],
            [[ bl_x, bl_y ]]
        ]
        box = np.array(box, dtype=np.int32)
        boxes.append(box)

    return boxes



def is_ascii(s):
    # Function that returns true if all characters
    # are ASCII characters
    return all(ord(c) < 128 for c in s)
