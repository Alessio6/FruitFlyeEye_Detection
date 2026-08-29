#Original offline program that detects fly head parts and allows the user to edit detected meshes with an editor.
from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
import json
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import copy

class PolygonEditor:
    def __init__(self, imgpath, masksdata, origimg):
        self.window = tk.Toplevel()
        self.window.title(f"Edit Segmentation - {Path(imgpath).name}")
        self.window.geometry("1200x800")
        
        #Bring to the front of the screen
        self.window.lift()
        self.window.attributes('-topmost', True)
        self.window.after(100, lambda: self.window.attributes('-topmost', False))
        self.window.focus_force()
        
        self.imgpath = imgpath
        self.origimg = origimg.copy()
        self.masksdata = copy.deepcopy(masksdata)
        
        #Convert masks to polygons
        self.polygons = []
        self.masks_to_polygons()
        
        #UI state
        self.selpolygon = None
        self.selpoint = None
        self.dragging = False
        self.scalefactor = 1.0
        self.photo = None
        self.finished = False
        
        #Store the original
        self.origpolygons = copy.deepcopy(self.polygons)
        
        #Setup UI
        self.setup_ui()
        self.window.update()
        self.redraw()
        self.window.after(100, self.fit_to_window)
    
    def masks_to_polygons(self):
        if self.masksdata['masks'] is None:
            return
            
        h = self.origimg.shape[0]
        w = self.origimg.shape[1]
        
        for i in range(len(self.masksdata['masks'])):
            mask = self.masksdata['masks'][i]
            
            #Get mask shape
            maskh = mask.shape[0]
            maskw = mask.shape[1]
            
            #Resize if needed
            if maskh != h or maskw != w:
                maskresize = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
            else:
                maskresize = mask
            
            maskbin = (maskresize > 0.5).astype(np.uint8)
            
            #Make sure 2D
            if len(maskbin.shape) > 2:
                maskbin = maskbin.squeeze()
            
            #Find contours
            contours = cv2.findContours(maskbin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = contours[0]
            
            if contours:
                #Get biggest contour
                biggestcontour = max(contours, key=cv2.contourArea)
                
                #Simplifies
                epsilon = 0.005 * cv2.arcLength(biggestcontour, True)
                approx = cv2.approxPolyDP(biggestcontour, epsilon, True)
                
                #Convert to a list
                points = []
                for pt in approx:
                    x = int(pt[0][0])
                    y = int(pt[0][1])
                    points.append((x, y))
                
                self.polygons.append({
                    'points': points,
                    'class': int(self.masksdata['classes'][i]),
                    'conf': float(self.masksdata['confs'][i])
                })
    
    def setup_ui(self):
        mainframe = tk.Frame(self.window)
        mainframe.pack(fill=tk.BOTH, expand=True)
        
        canvasframe = tk.Frame(mainframe)
        canvasframe.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        vscrollbar = tk.Scrollbar(canvasframe, orient=tk.VERTICAL)
        vscrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        hscrollbar = tk.Scrollbar(canvasframe, orient=tk.HORIZONTAL)
        hscrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.canvas = tk.Canvas(canvasframe, 
                               xscrollcommand=hscrollbar.set,
                               yscrollcommand=vscrollbar.set,
                               bg='gray')
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        vscrollbar.config(command=self.canvas.yview)
        hscrollbar.config(command=self.canvas.xview)
        
        controlframe = tk.Frame(mainframe, width=200, bg='lightgray')
        controlframe.pack(side=tk.RIGHT, fill=tk.Y)
        controlframe.pack_propagate(False)
        
        self.infolabel = tk.Label(controlframe, text="", justify=tk.LEFT,
                                   bg='lightgray', font=('Arial', 9))
        self.infolabel.pack(padx=10, pady=10)
        
        tk.Button(controlframe, text="Reset", command=self.reset, 
                 bg='orange', fg='white', font=('Arial', 10, 'bold')).pack(pady=5)
        
        tk.Button(controlframe, text="Finish", command=self.finish,
                 bg='green', fg='white', font=('Arial', 12, 'bold')).pack(pady=20)
        
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<Button-3>", self.on_right_click)  #Right click to delete point
        self.canvas.bind("<Double-Button-1>", self.on_double_click)  #Double click to add point
        self.canvas.bind("<MouseWheel>", self.on_mousewheel)
        self.canvas.bind("<Button-4>", self.on_mousewheel)
        self.canvas.bind("<Button-5>", self.on_mousewheel)
        self.window.protocol("WM_DELETE_WINDOW", self.finish)
    
    def redraw(self):
        self.canvas.delete("all")
        
        #Copy image
        dispimg = self.origimg.copy()
        h = dispimg.shape[0]
        w = dispimg.shape[1]
        
        #Draw polygons
        for i in range(len(self.polygons)):
            polydata = self.polygons[i]
            points = polydata['points']
            cls = polydata['class']
            conf = polydata['conf']
            
            #Colours
            if cls == 0:
                col = (0, 255, 0)  #Green
                lab = "Left Eye"
            elif cls == 1:
                col = (0, 255, 255)  #Yellow
                lab = "Right Eye"
            elif cls == 2:
                col = (255, 0, 0)  #Red
                lab = "Top Eye"
            elif cls == 3:
                col = (255, 255, 0)  #Orange
                lab = "Antenna"
            else:
                col = (255, 0, 255)  #Purple
                lab = "Something Else"
            
            #Draw filled polygon
            if len(points) >= 3:
                pts = np.array(points, np.int32)
                overlay = dispimg.copy()
                cv2.fillPoly(overlay, [pts], col)
                cv2.addWeighted(overlay, 0.4, dispimg, 0.6, 0, dispimg)
                
                #Draw outline
                thick = 2 if i == self.selpolygon else 1
                cv2.polylines(dispimg, [pts], True, col, thick)
                
                #Draw label
                m = cv2.moments(pts)
                if m["m00"] != 0:
                    cx = int(m["m10"] / m["m00"])
                    cy = int(m["m01"] / m["m00"])
                    
                    labtext = f"{lab} {conf:.2f}"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    
                    #Scale the font based on the size of the image
                    imgsize = max(h, w)
                    if imgsize < 500:
                        fontscale = 0.4
                        thick2 = 1
                    elif imgsize < 1000:
                        fontscale = 0.7
                        thick2 = 2
                    elif imgsize < 2000:
                        fontscale = 1.0
                        thick2 = 2
                    else:
                        fontscale = 1.5
                        thick2 = 3
                    
                    textsize = cv2.getTextSize(labtext, font, fontscale, thick2)
                    tw = textsize[0][0]
                    th = textsize[0][1]
                    
                    cv2.rectangle(dispimg, (cx-5, cy-th-5), (cx+tw+5, cy+5), col, -1)
                    cv2.putText(dispimg, labtext, (cx, cy), font, fontscale, (0, 0, 0), thick2)
        
        for i in range(len(self.polygons)):
            polydata = self.polygons[i]
            points = polydata['points']
            
            #Point colour
            ptcol = (255, 0, 0) if i == self.selpolygon else (255, 255, 255)
            
            for j in range(len(points)):
                x = points[j][0]
                y = points[j][1]
                rad = 8 if (i == self.selpolygon and j == self.selpoint) else 5
                cv2.circle(dispimg, (x, y), rad, ptcol, -1)
                cv2.circle(dispimg, (x, y), rad, (0, 0, 0), 2)
        
        #Scale image
        neww = int(w * self.scalefactor)
        newh = int(h * self.scalefactor)
        dispimg = cv2.resize(dispimg, (neww, newh))
        
        #Convert to the correct type
        dispimgrgb = cv2.cvtColor(dispimg, cv2.COLOR_BGR2RGB)
        pilimg = Image.fromarray(dispimgrgb)
        
        #Keep reference
        self.photo = ImageTk.PhotoImage(pilimg)
        
        #Display
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        self.canvas.config(scrollregion=(0, 0, neww, newh))
        
        #Update info
        if self.selpolygon is not None:
            poly = self.polygons[self.selpolygon]
            clsname = (
                "Left Eye"      if poly['class'] == 0 else
                "Right Eye"     if poly['class'] == 1 else
                "Left Antenna"  if poly['class'] == 2 else
                "Right Antenna"     if poly['class'] == 3 else
                "Unnamed"
            )
            infotext = f"Selected:\n{clsname}\nConf: {poly['conf']:.2f}\nPoints: {len(poly['points'])}"
        else:
            infotext = f"Total Polygons: {len(self.polygons)}"
        
        self.infolabel.config(text=infotext)
    
    def canvas_to_img_coords(self, canvasx, canvasy):
        x = int(self.canvas.canvasx(canvasx) / self.scalefactor)
        y = int(self.canvas.canvasy(canvasy) / self.scalefactor)
        return x, y
    
    def on_click(self, event):
        imgx, imgy = self.canvas_to_img_coords(event.x, event.y)
        
        #Checking if clicking on point
        mindist = 15 / self.scalefactor
        clickedpoly = None
        clickedpt = None
        
        for i in range(len(self.polygons)):
            polydata = self.polygons[i]
            for j in range(len(polydata['points'])):
                x = polydata['points'][j][0]
                y = polydata['points'][j][1]
                dist = np.sqrt((imgx - x)**2 + (imgy - y)**2)
                if dist < mindist:
                    clickedpoly = i
                    clickedpt = j
                    mindist = dist
        
        if clickedpt is not None:
            #Clicked on point
            self.selpolygon = clickedpoly
            self.selpoint = clickedpt
            self.dragging = True
        else:
            #Check if inside polygon
            for i in range(len(self.polygons)):
                polydata = self.polygons[i]
                pts = np.array(polydata['points'], np.int32)
                if cv2.pointPolygonTest(pts, (imgx, imgy), False) >= 0:
                    self.selpolygon = i
                    self.selpoint = None
                    break
            else:
                self.selpolygon = None
                self.selpoint = None
        
        self.redraw()
    
    def on_drag(self, event):
        if self.dragging and self.selpoint is not None:
            imgx, imgy = self.canvas_to_img_coords(event.x, event.y)
            
            #Clamp to bounds
            h = self.origimg.shape[0]
            w = self.origimg.shape[1]
            imgx = max(0, min(w - 1, imgx))
            imgy = max(0, min(h - 1, imgy))
            
            #Update point
            self.polygons[self.selpolygon]['points'][self.selpoint] = (imgx, imgy)
            self.redraw()
    
    def on_release(self, event):
        self.dragging = False
    
    def on_right_click(self, event):
        imgx, imgy = self.canvas_to_img_coords(event.x, event.y)
        
        #Find closest point
        mindist = 15 / self.scalefactor
        clickedpoly = None
        clickedpt = None
        
        for i in range(len(self.polygons)):
            polydata = self.polygons[i]
            for j in range(len(polydata['points'])):
                x = polydata['points'][j][0]
                y = polydata['points'][j][1]
                dist = np.sqrt((imgx - x)**2 + (imgy - y)**2)
                if dist < mindist:
                    clickedpoly = i
                    clickedpt = j
                    mindist = dist
        
        #Delete point if found and polygon has more than 3 points
        if clickedpt is not None:
            if len(self.polygons[clickedpoly]['points']) > 3:
                del self.polygons[clickedpoly]['points'][clickedpt]
                self.selpolygon = clickedpoly
                self.selpoint = None
                self.redraw()
            else:
                messagebox.showwarning("Cannot Delete", "Polygon must have at least 3 points")
    
    def on_double_click(self, event):
        imgx, imgy = self.canvas_to_img_coords(event.x, event.y)
        
        #Find closest edge
        if self.selpolygon is not None:
            polydata = self.polygons[self.selpolygon]
            points = polydata['points']
            
            mindist = 15 / self.scalefactor
            bestedge = None
            bestpos = None
            
            #Check each of the edges
            for i in range(len(points)):
                p1 = points[i]
                p2 = points[(i + 1) % len(points)]
                
                #Calculate distance from point to line segment
                x1 = p1[0]
                y1 = p1[1]
                x2 = p2[0]
                y2 = p2[1]
                
                #Vector from p1 to p2
                dx = x2 - x1
                dy = y2 - y1
                
                #If it has no 0 length
                if dx == 0 and dy == 0:
                    dist = np.sqrt((imgx - x1)**2 + (imgy - y1)**2)
                else:
                    #Parameter t for projection onto the line
                    t = max(0, min(1, ((imgx - x1) * dx + (imgy - y1) * dy) / (dx * dx + dy * dy)))
                    
                    #Closest point on segment
                    closestx = x1 + t * dx
                    closesty = y1 + t * dy
                    
                    #Distance to segment
                    dist = np.sqrt((imgx - closestx)**2 + (imgy - closesty)**2)
                
                if dist < mindist:
                    mindist = dist
                    bestedge = i
                    bestpos = (imgx, imgy)
            
            #Insert point after bestedge
            if bestedge is not None:
                self.polygons[self.selpolygon]['points'].insert(bestedge + 1, bestpos)
                self.selpoint = bestedge + 1
                self.redraw()
    
    def on_mousewheel(self, event):
        if event.num == 5 or event.delta < 0:
            #Zoom out
            self.scalefactor = max(0.1, self.scalefactor * 0.9)
        else:
            #Zoom in
            self.scalefactor = min(5.0, self.scalefactor * 1.1)
        
        self.redraw()
    
    def fit_to_window(self):
        #Fit the image to the size of the window
        self.canvas.update()
        canvasw = self.canvas.winfo_width()
        canvash = self.canvas.winfo_height()
        
        imgh = self.origimg.shape[0]
        imgw = self.origimg.shape[1]
        
        scalew = canvasw / imgw
        scaleh = canvash / imgh
        self.scalefactor = min(scalew, scaleh) * 0.95
        
        self.redraw()
    
    def reset(self):
        if messagebox.askyesno("Reset", "Reset all changes?"):
            self.polygons = copy.deepcopy(self.origpolygons)
            self.selpolygon = None
            self.selpoint = None
            self.redraw()
    
    def finish(self):
        self.finished = True
        self.window.destroy()
    
    def get_edited_masks(self):
        #Convert polygons back to masks
        if not self.polygons:
            return None, None, None
        
        h = self.origimg.shape[0]
        w = self.origimg.shape[1]
        masks = []
        classes = []
        confs = []
        
        for polydata in self.polygons:
            mask = np.zeros((h, w), dtype=np.uint8)
            pts = np.array(polydata['points'], np.int32)
            cv2.fillPoly(mask, [pts], 255)
            
            masks.append(mask)
            classes.append(polydata['class'])
            confs.append(polydata['conf'])
        
        return np.array(masks), np.array(classes), np.array(confs)
    
    def wait_for_finish(self):
        #Wait for the user
        self.window.wait_window()


def calccover(imgpath, masks):
    img = cv2.imread(str(imgpath), cv2.IMREAD_UNCHANGED)
    if img is None:
        return 0.0
    
    if len(img.shape) == 2:
        h = img.shape[0]
        w = img.shape[1]
    else:
        h = img.shape[0]
        w = img.shape[1]
    
    totalpix = h * w
    
    #Count the amount of mask pixels
    maskpix = 0
    for mask in masks:
        maskpix = maskpix + np.sum(mask > 0)
    
    pct = (maskpix / totalpix) * 100
    return pct


def detecteyes(modelpath, imgfiles, outfolder, confthresh=0.25):
    print("")
    print("Loading model")
    model = YOLO(modelpath)
    
    outpath = Path(outfolder)
    outpath.mkdir(parents=True, exist_ok=True)
    
    print("Found " + str(len(imgfiles)) + " images")
    print("Confidence: " + str(confthresh))
    
    #Convert greyscale
    tempfolder = outpath / "temp_rgb"
    tempfolder.mkdir(parents=True, exist_ok=True)
    
    converted = []
    print("")
    print("Converting greyscale images")
    for file1 in imgfiles:
        if file1.suffix.lower() == '.tif':
            img = cv2.imread(str(file1), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            
            if img.dtype != np.uint8:
                img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            
            imgrgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            tempfile = tempfolder / (file1.stem + ".jpg")
            cv2.imwrite(str(tempfile), imgrgb)
            converted.append((tempfile, file1))
        else:
            converted.append((file1, file1))
    
    print("")
    print("Running detection")
    
    needtemp = False
    for c in converted:
        if c[0] != c[1]:
            needtemp = True
            break
    
    if needtemp:
        source = str(tempfolder)
    else:
        source = [str(c[0]) for c in converted]
    
    results = model.predict(
        source=source,
        conf=confthresh,
        save=False,
        retina_masks=True,
    )
    
    #Make output directory
    annotdir = outpath / 'detections'
    annotdir.mkdir(parents=True, exist_ok=True)
    
    print("")
    print("Results:")
    
    summary = []
    
    root = tk.Tk()
    root.withdraw()
    
    #Process each of the results
    for i in range(len(results)):
        result = results[i]
        
        #Find the matching file
        resultpath = Path(result.path)
        file2 = None
        for convtemp, convorig in converted:
            if convtemp.name == resultpath.name:
                file2 = convorig
                break
        
        if file2 is None:
            continue
        
        name = file2.name
        print("")
        print(name)
        
        #Load original
        img = cv2.imread(str(file2), cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        
        if img.dtype != np.uint8:
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        #Make sure it's 3 channel
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif len(img.shape) == 3 and img.shape[2] == 1:
            img = cv2.cvtColor(img.squeeze(), cv2.COLOR_GRAY2BGR)
        
        if result.masks is not None:
            masksdata = {
                'masks': result.masks.data.cpu().numpy(),
                'classes': result.boxes.cls.cpu().numpy(),
                'confs': result.boxes.conf.cpu().numpy()
            }
            
            editor = PolygonEditor(file2, masksdata, img)
            editor.wait_for_finish()
            
            editmasks, editclasses, editconfs = editor.get_edited_masks()
            
            if editmasks is not None:
                h = img.shape[0]
                w = img.shape[1]
                lefteyes = 0
                righteyes = 0
                topeyes = 0
                antennae = 0
                allmasks = []
                
                for j in range(len(editmasks)):
                    mask = editmasks[j]
                    cls = editclasses[j]
                    conf = editconfs[j]
                    
                    maskbin = (mask > 0).astype(np.uint8)
                    allmasks.append(maskbin)
                    
                    #Get colour
                    if int(cls) == 0:
                        lefteyes = lefteyes + 1
                        col = (0, 255, 0)  #Green
                        lab = "Left Eye"
                    elif int(cls) == 1:
                        righteyes = righteyes + 1
                        col = (0, 255, 255)  #Yellow
                        lab = "Right Eye"
                    elif int(cls) == 2:
                        topeyes = topeyes + 1
                        col = (255, 0, 0)  #Red
                        lab = "Top Eye"
                    elif int(cls) == 3:
                        antennae = antennae + 1
                        col = (255, 255, 0)  #Orange
                        lab = "Antenna"
                    else:
                        righteyes = righteyes + 1
                        col = (255, 0, 255)  #Purple
                        lab = "Something Else"
                    
                    maskbool = maskbin > 0
                    
                    #Apply colour
                    for c in range(3):
                        img[:, :, c] = np.where(maskbool, 
                                               img[:, :, c] * 0.6 + col[c] * 0.4, 
                                               img[:, :, c])
                    
                    #Draw contour
                    contours = cv2.findContours(maskbin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    contours = contours[0]
                    cv2.drawContours(img, contours, -1, col, 6)
                    
                    #Add label
                    if contours:
                        m = cv2.moments(contours[0])
                        if m["m00"] != 0:
                            cx = int(m["m10"] / m["m00"])
                            cy = int(m["m01"] / m["m00"])
                            
                            labtext = lab + " " + str(round(conf, 2))
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            
                            #Change the font size depending on the image size
                            imgsize = max(h, w)
                            if imgsize < 500:
                                fontscale = 0.4
                                thick = 1
                            elif imgsize < 1000:
                                fontscale = 0.7
                                thick = 2
                            elif imgsize < 2000:
                                fontscale = 1.0
                                thick = 2
                            else:
                                fontscale = 1.5
                                thick = 3
                            
                            textsize = cv2.getTextSize(labtext, font, fontscale, thick)
                            tw = textsize[0][0]
                            th = textsize[0][1]
                            
                            cv2.rectangle(img, (cx-10, cy-th-10), (cx+tw+10, cy+10), col, -1)
                            cv2.putText(img, labtext, (cx, cy), font, fontscale, (0, 0, 0), thick)
                
                #Calculate amount of coverage
                pct = calccover(file2, allmasks)
                
                #Save summary
                sumentry = {
                    'filename': name,
                    'left_eyes': lefteyes,
                    'right_eyes': righteyes,
                    'top_eyes': topeyes,
                    'antennae': antennae,
                    'total_eyes': len(editmasks),
                    'coverage_percentage': round(pct, 2),
                    'confidences': editconfs.tolist()
                }
                summary.append(sumentry)
                
                print("  Left: " + str(lefteyes) + ", Right: " + str(righteyes) + ", Top: " + str(topeyes) + ", Antennae: " + str(antennae))
                print("  Coverage: " + str(round(pct, 2)) + "%")
        else:
            #No detections
            sumentry = {
                'filename': name,
                'left_eyes': 0,
                'right_eyes': 0,
                'total_eyes': 0,
                'coverage_percentage': 0.0,
                'confidences': []
            }
            summary.append(sumentry)
            print("  No eyes detected")
        
        #Save the image
        outname = file2.stem + "_segmented.jpg"
        cv2.imwrite(str(annotdir / outname), img)
    
    root.destroy()
    
    #Save summary
    sumfile = outpath / "segmentation_summary.json"
    with open(sumfile, 'w') as f:
        json.dump(summary, f, indent=2)
    
    #Clean temp
    import shutil
    if tempfolder.exists():
        shutil.rmtree(tempfolder)
    
    print("")
    print("Summary:")
    print("Processed " + str(len(imgfiles)) + " images")
    
    totaleyes = 0
    for s in summary:
        totaleyes = totaleyes + s['total_eyes']
    print("Detected " + str(totaleyes) + " eyes total")
    
    if summary:
        avgcov = 0
        for s in summary:
            avgcov = avgcov + s['coverage_percentage']
        avgcov = avgcov / len(summary)
        print("Average coverage " + str(round(avgcov, 2)) + "%")
    
    print("Segmented images saved to " + str(annotdir))
    print("Summary saved to " + str(sumfile))
    
    return summary

def main():
    print("Starting")
    
    root = tk.Tk()
    root.withdraw()
    
    imgfiles = filedialog.askopenfilenames(
        title="Select images",
        filetypes=[
            ("Image files", "*.tif *.jpg *.jpeg *.png *.tiff"),
            ("TIFF files", "*.tif *.tiff"),
            ("JPEG files", "*.jpg *.jpeg"),
            ("PNG files", "*.png"),
            ("All files", "*.*")
        ]
    )
    
    root.destroy()
    
    if not imgfiles:
        print("No files selected")
        return
    
    imgfiles = [Path(f) for f in imgfiles]
    
    script = Path(__file__).parent
    modelpath = script / "fly_training_seg" / "fly_eye_seg" / "weights" / "best.pt"
    outfolder = script / "fly_detections_seg" 
    
    print("")
    print("Model " + str(modelpath))
    print("Images " + str(len(imgfiles)) + " files")
    print("Output " + str(outfolder))
    
    if not modelpath.exists():
        print("Model wasn't found")
        return
    
    detecteyes(modelpath, imgfiles, outfolder)
    
    print("")
    print("Detection finished")

if __name__ == "__main__":
    main()
