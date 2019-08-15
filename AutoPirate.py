# Document handling tool to use on common documents
# used around Red Berry Innovations

# import libraries
import cv2
import sys
from PIL import Image, ImageOps
import imutils
import numpy as np
from shapedetector import ShapeDetector
import pytesseract
import re
from pdf2image import convert_from_path
from math import *
import os
from skimage.measure import compare_ssim
from skimage.transform import probabilistic_hough_line
from skimage.morphology import skeletonize
import pickle
import networkx as nx

# from py2neo import Node, Relationship, Graph
# graph = Graph("bolt://localhost:7687",auth=("neo4j","redberry"))
# graph.delete_all()

G = nx.Graph()
circle_centers = []
border = None


def bw_filter(img,rtype="gray"):
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

def line_filter(img, color,rtype="gray"):
    # Function that filters out the pixels that match a specfic color
    # range.  Uses hsv color space for simpler color range masking

    if color == "ECU":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.inRange(gray,246,248)
        return gray
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh = cv2.inRange(gray,15,240)
        median = cv2.medianBlur(thresh,5)
        return median


def trim_meta(img):

    # Retreive black/white image and calculate edges
    bw_img = bw_filter(img,rtype = "gray")
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
            border = approx
            break

    # Calculate region boundries and region itself
    upper_left = approx[0][0] + np.array([0,5])
    bottom_right = approx[2][0] + np.array([0,-5])
    region = np.append(upper_left, bottom_right)

    # Grab only the inside of the border
    return np.array(Image.fromarray(img,'RGB').crop(region))


def get_lines(img):
    # Function that grabs all the lines from
    # the image

    # Grayscale image for skeletonize function
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Erode smaller lines
    kernel = np.ones((5,5), np.uint8)
    erode = cv2.erode(img, kernel, iterations=1)

    ret, thresh = cv2.threshold(erode,1,255,cv2.THRESH_BINARY)
    thresh = (thresh / 255).astype(np.uint8)
    skeleton = skeletonize(thresh).astype(np.uint8) * 255

    temp_lines = probabilistic_hough_line(skeleton,threshold=45,line_length=30,line_gap=50)
    lines = []
    for l in temp_lines:
        lines.append( np.array(l).reshape((1,4)).astype(np.int32) )
    lines = np.array(lines)

    return lines

def get_contours(img):
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


def plot_contour(img,cnt):
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


def get_bound_rects(cnts) :
    # Function that takes in contours and calculates the
    # corresponding bounding rectangle
    # temp = img.copy()
    params = []

    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        params.append((x,y,w,h))
        # cv2.rectangle(temp, (x,y), (x+w, y+h), (0,0,255), 2)

    return params


def clear_contours(img,contours):
    # Given a list of contours, this function clears them from
    # the image and returns the new image
    temp = img.copy()

    for c in contours:
        cv2.fillPoly(temp, pts=[c], color=(255,255,255))

    return temp

def plot_lines(shape,lines,se=False):
    # Function that takes lines for get_lines and
    # plots them on a blank image
    blank = np.zeros(shape, np.uint8)

    for i in range(lines.shape[0]):
        x1, y1, x2, y2 = lines[i][0]
        cv2.line(blank, (x1, y1), (x2, y2), (255, 255, 255), 3)
        # cv2.putText(blank, "{}".format(i), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)
        if se:
            cv2.circle(blank, (x1, y1), 7, (255, 0, 0), -1)
            cv2.circle(blank, (x2, y2), 7, (0, 255, 0), -1)


    return blank

def extract_text(img):
    # Function to extract text boxes, line colors and decipher
    # text using ocr engine
    global G

    # Grab digit images and load them into array
    # Compute and store histograms for each image
    digits = pickle.load(open("digits.p",'rb'))
    digits2 = pickle.load(open("digits2.p",'rb'))

    # Shape detector to tell the shape of contour
    sd = ShapeDetector()

    # Filter out ecu boxes
    ecu_img = line_filter(img, color="ECU", rtype='rgb')
    ecu_img = cv2.medianBlur(ecu_img, 5)

    # Enlarge the boxes so we can overwrite the dashed or
    # solid lines
    kernel = np.ones((9,9), np.uint8)
    _ecu_img = cv2.dilate(ecu_img, kernel, iterations=3)

    # Grab the contours and then fill them with white to
    # erase them from the image
    ecu_contours,centers = get_contours(_ecu_img)
    no_boxes = clear_contours(img, ecu_contours)
    ecu_rects = get_bound_rects(ecu_contours)

    # Convert to gray and filter out the colored wires
    # so it is just text
    just_text = bw_filter(no_boxes, rtype='rgb')

    # Add circle boxes for later relationship building
    kernel2 = np.ones((5,5), np.uint8)
    erode = cv2.erode(just_text, kernel2, iterations=1)
    median = cv2.medianBlur(erode, 5)
    temp_contours, circle_centers = get_contours(median)
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

    circle_centers = circle_centers

    # Dilate text to get a rectangular contour
    # (Thanksgiving Approach)
    enlarged = cv2.dilate(just_text, kernel, iterations=5)

    # Grab text contours and convert to rectangular
    # boundries
    text_contours, centers = get_contours(enlarged)
    params = get_bound_rects(text_contours)

    # Decipher/Filter text and store mappings to
    # center and bounding rectangle
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
            pass

        elif len(text) == 0 and (.65 < ar < 1.25):
            text_roi = text_roi.resize((100,80)).convert('L')

            best_score = 0
            best_index = 0
            for i2,d in enumerate(digits):
                cmp_score = compare_ssim(d,np.array(text_roi),data_range=d.max()-d.min())
                cmp_score2 = compare_ssim(digits2[i2],np.array(text_roi),data_range=d.max()-d.min())
                score = max(cmp_score,cmp_score2)
                if score > best_score:
                    best_score = score
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

    # Create nodes for each ecu and outward/inward
    for k,v in ecu_map.items():
        val = v[1].replace("\n\n","\n")
        G.add_node(val)

    return ecu_map


def build_relationships(img,mapping):
    # Function that takes in wire lines and builds the
    # relationship beween ecus/modules
    global G

    # Get line information
    lines = get_lines(img)

    # Get line image
    lines_plot = plot_lines(img.shape,lines)

    # Find corners
    lines_plot = cv2.cvtColor(lines_plot, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(lines_plot, 10000, 0.5, 40)

    # Add the black circle boxes to list of boxes
    boxes2 = corners2_boxes(np.array(circle_centers), circles=True)

    # Convert corners to bounding boxes
    boxes1 = corners2_boxes(corners)
    boxes = boxes1.copy()

    box_votes, line_votes = voting(lines, boxes)
    start_lines = get_starts(box_votes)

    used_indices = []
    used_boxes = []
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

        while constructing:
            line = lines[line_index].ravel().reshape((2,2))
            box1, box2 = line_votes[line_index]
            box = box2 if box1 == last_box else box1
            last_box = box

            if len(box_votes[box]) == 1:
                # At the end of path
                end_index, se = box_votes[box][0]
                end = tuple(lines[end_index].reshape((2,2))[se])
                concat_lines.append( (initial_start,end) )
                constructing = False

            else:
                next_lines = box_votes[box]
                last_line = line_index
                line_index = next_lines[0][0] if next_lines[1][0] == last_line else next_lines[1][0]
                used_indices.append(line_index)


    # Add each of the junction boxes to the node mapping
    # as well as adding them to nodes in the graph
    count = len(mapping)
    junction_params = get_bound_rects(boxes2)
    for i,b in enumerate(junction_params):
        node_name = 'JUNCTION-{}'.format(i)
        val = ( (b), node_name )
        G.add_node(node_name)
        mapping.update({count: val})
        count += 1

    # Map relationships for starts and ends
    for line in concat_lines:
        start, end = line
        start_text = False
        end_text = False
        for k,v in mapping.items():
            box = v[0]
            text = v[1].replace("\n\n","\n")
            if "OUT" == text.split("-")[0] or "IN" == text.split("-")[0]:
                exs = (80,30)
            elif "JUNCTION" == text.split("-")[0]:
                exs = (10,10)
            else:
                exs = (50,50)
            if is_connected(start, box, exs):
                start_text = text
            elif is_connected(end, box, exs):
                end_text = text
            
            if start_text and end_text:
                G.add_edge(start_text,end_text)
                break

    print("\t[+] Obtained graph with [ {} ] nodes and [ {} ] edges".format(G.number_of_nodes(),G.number_of_edges()),end="\n\n")
    return G.copy()

def pipeline(directory):
    # Function that performs a set of operations stopping
    # at the type CHECKPOINT
    global G

    images = os.listdir(directory)

    graphs = []
    for i in images:

        print("[*] Parsing image file {}".format(i))
        
        # Open image depending on image format
        full_path = os.path.join(directory,i)
        if i.split('.')[1].lower() == "pdf":
            img = convert_from_path(full_path)[0].convert('RGB')
        else:
            img = Image.open(full_path).convert('RGB')

        # Trim border
        img = np.array(img)
        border_image = trim_meta(img)

        # Grab ECU mappings and the line colors in image
        print("\t|---> [*] Extracting Text")
        ecu_map = extract_text(border_image)

        # Build relationships based off lines
        lines_img = line_filter(border_image,color='lines')
        lines_img = cv2.cvtColor(lines_img, cv2.COLOR_GRAY2RGB)

        print("\t|---> [*] Building Relationships")
        graph = build_relationships(lines_img,ecu_map.copy())
        graphs.append(graph.copy())

        G.clear()

    # Concatenate page continuation lines
    print("[*] Composing continued page graphs")
    F = None
    for i in range(0,len(graphs)-1):
        print(5*" "+"|---> Page {} <--> Page {}".format(i,i+1))
        g,h = (F,graphs[i+1]) if F != None else (graphs[i],graphs[i+1])

        out_nodes = [n for n in g.nodes if "OUT-" in n]
        next_in_nodes = [n for n in h.nodes if "IN-" in n]
        # next_in_nodes = next_in_nodes[:-1]
        if len(out_nodes) != len(next_in_nodes):
            print("")
            print("[-] Length of OUT[{}] and IN[{}] nodes does not match".format(len(out_nodes),len(next_in_nodes)))
            print("    returning all graphs")
            return graphs

        new_names = ["CONNECT-JUNCT-{}-{}".format(i,n) for n in range(0,len(next_in_nodes))]
        mapping_out = dict(list(zip(out_nodes,new_names)))
        mapping_in = dict(list(zip(next_in_nodes,new_names)))

        G_ = nx.relabel_nodes(g,mapping_out)
        H_ = nx.relabel_nodes(h,mapping_in)
        F = nx.compose(G_,H_)

    return F
    # Take all of the connections and perform any pre-processing
    # before storing in graph database
    # store_connections(all_connections)

#### HELPER FUNCTIONS ####

def store_connections(connections):
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

def is_connected(point,box,exs):
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



def text2_ecu(ecu_rects,text_rect,text):
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


def lc_check(text):
    # Function that checks if text is part of line
    # color description
    if ( bool(re.match(r"[A-Z]{3}/[A-Z]{3}$", text)) or
         bool(re.match(r"[A-Z]{3}$",text)) or
         bool(re.match(r"[A-Z]{3}[/,\n]{2}[A-Z]{3}$", text)) or
         bool(re.match(r"[A-Z]{3}[\n, /]{2}[A-Z]{3}$", text)) ):
        return True
    else:
       return False

def show_image(img):
    # Function that displays the image
    # from a numpy array
    if len(img.shape) < 3:
        temp_image = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        temp_image = Image.fromarray(temp_image, 'RGB')
        temp_image.show()
    else:
        temp_image = Image.fromarray(img, 'RGB')
        temp_image.show()


def is_inside(point,area):
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


def voting(lines,boxes):
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


def filter_text(text):
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


def line2_angle(line):
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


def corners2_boxes(corners,circles=False):
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
