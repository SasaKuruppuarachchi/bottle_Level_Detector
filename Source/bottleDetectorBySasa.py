
# CODE REVIEW
##- Sasanka Kuruppuarachchi
##Project - "https://github.com/SasaKuruppuarachchi/bottle_Level_Detector"

#import packages numpy, cv2, Matplotlib, sklearn.cluster, 
import numpy as np 
import os 
import cv2 
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering

# gaussian_kernel(size,sigma) is the function for noce filtering using gausian filter
def gaussian_kernel(size,sigma = 1):
    size = int(size)
    x,y = np.mgrid[-size:size+1,-size:size+1]
    normal = 1/(2.0*np.pi*sigma**2)
    g = np.exp(-((x**2+y**2)/(2.0*sigma**2)))*normal
    return g

# defining the canny detector function 
   
# here weak_th and strong_th are thresholds for 
# double thresholding step
def contour_detectorSasa(img, weak_th = None, strong_th = None, num_clusters = 3):
    _contours =  []
       
    # Noise reduction step - image smoothing
    img = cv2.filter2D(img,-1,gaussian_kernel(size,sigma = 1)) 
       
    # Calculating the gradients 
    gx = cv2.Sobel(np.float32(img), cv2.CV_64F, 1, 0, 3) 
    gy = cv2.Sobel(np.float32(img), cv2.CV_64F, 0, 1, 3) 
    
    # Conversion of Cartesian coordinates to polar  
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees = True) 
    cv2.imshow('cannyO',mag) 
    # setting the minimum and maximum thresholds  
    # for double thresholding 
    mag_max = np.max(mag) 
    if not weak_th:weak_th = mag_max * 0.1
    if not strong_th:strong_th = mag_max * 0.5
      
    # getting the dimensions of the input image   
    height, width = img.shape 
     
    # get weak and strong ids
    weak_ids = np.zeros_like(img) 
    strong_ids = np.zeros_like(img)               
    ids = np.zeros_like(img)
    
    row = []
    # double thresholding step 
    for i_x in range(width):
        #row = []
        for i_y in range(height): 
            coord = []
            grad_mag = mag[i_y, i_x] 
              
            if grad_mag<weak_th: 
                mag[i_y, i_x]= 0
            elif strong_th>grad_mag>= weak_th: 
                mag[i_y, i_x]= 1
                
                coord.append(i_x)
                coord.append(i_y)
                row.append(coord)
                
            else: 
                mag[i_y, i_x]= 2
                coord.append(i_x)
                coord.append(i_y)
                row.append(coord)
                
    cv2.imshow('canny1',mag)
    #clustering
    cluster = AgglomerativeClustering(n_clusters= num_clusters, affinity='euclidean', linkage='single')
    cluster.fit_predict(row)
    group1 = []
    group2 = []
    group3 = []
    group4 = []
    group5 = []
    
    n = len(row)
    for i in range (n):
        #print(row[i], cluster.labels_[i])
        if cluster.labels_[i] == 0:
            group1.append(row[i])
        elif cluster.labels_[i] == 1:
            group2.append(row[i])
        elif cluster.labels_[i] == 2:
            group3.append(row[i])
        elif cluster.labels_[i] == 3:
            group4.append(row[i])
        elif cluster.labels_[i] == 4:
            group5.append(row[i])

    

    ctr1 = np.array(group1).reshape((-1,1,2)).astype(np.int32)
    ctr2 = np.array(group2).reshape((-1,1,2)).astype(np.int32)
    ctr3 = np.array(group3).reshape((-1,1,2)).astype(np.int32)
    ctr4 = np.array(group4).reshape((-1,1,2)).astype(np.int32)
    ctr5 = np.array(group5).reshape((-1,1,2)).astype(np.int32)
    _contours.append(ctr1)
    _contours.append(ctr2)
    _contours.append(ctr3)
    
    return _contours


def fluid_level_detect(file_name, tresh_min, tresh_max , num_clusters = 3):
    image = cv2.imread(file_name)
    im_bw = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    

    (thresh, im_bw) = cv2.threshold(im_bw, tresh_min, tresh_max, 0)
    cv2.imwrite('bw_'+file_name, im_bw)
    cv2.imshow('thresholded',im_bw)

  

    count = 0
    fluidLevel = 0
    X = 0
    Y = 0
    W = 0
    H = 0
    fluidLevelEst = 0
    fluidPrecentage = 0
    fluidPercentageReal = 0
    text = ""
    a = 0
    #contours, hierarchy = cv2.findContours(im_bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours1 = contour_detectorSasa(im_bw, tresh_min,tresh_max, num_clusters)
    
    for c in contours1:
        
        rect = cv2.boundingRect(c)
        if rect[2] < 50 or rect[3] < 10: continue
        if rect[2] > 300 or rect[3] > 300: continue

        #print cv2.contourArea(c)
        x,y,w,h = rect
        cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),2)
        print("x =",rect[0])
        print("y =",rect[1])
        print("w =",rect[2])
        print("h =",rect[3])
        
        hwRatio = rect[3]/rect[2]
        print("h/w = ",hwRatio)
        fluidLevelEst = 34.645 * hwRatio
        print("Fluid level est = ",fluidLevelEst,"%")

        #experimental - draw actual min area contour
        rect1 = cv2.minAreaRect(c)
        xy,wh,a = rect1
        Hreal = 0
        Wreal = 0
        if abs(a)<45:
            Hreal = wh[1]
            Wreal = wh[0]
        else:
            Hreal = wh[0]
            Wreal = wh[1]
        print("")
        print("Real Fluid level accounted for the tilt of the camera")
        print("Real H = ", Hreal)
        print("Real W = ", Wreal)
        print("Real H/W = ",Hreal/Wreal)
        fluidPercentageReal = Hreal/Wreal*33.178
        print("Real Fluid level = ", fluidPercentageReal, "%")
        #cv2.putText(image,"o",(int(xy[0]),int(xy[1])),0,0.6,(0,0,255))
        box = cv2.boxPoints(rect1)
        box = np.int0(box)
        image = cv2.drawContours(image,[box],0,(0,0,0),1)
        #this gives the most accurate estimate for fluid level as it accounts for the camera tilt
        
        
        if rect[3] > H:
            fluidPrecentage = round(fluidPercentageReal)
            text = "Level = " + str(fluidPrecentage) + "%"
            X = x
            Y = y
            W = w
        H = rect[3]
        count +=1
        
    
    cv2.putText(image,text,(X+W+10,Y),0,0.6,(0,0,255))
    
    #show 80% rect
    Ha = int(-80/fluidPrecentage *Hreal)
    cv2.putText(image,"80%",(X-50,Y+H+Ha),0,0.6,(0,255,0))
    cv2.rectangle(image,(X,Y+H),(X+W,Y+Ha+H),(0,255,0),2)
    cv2.imshow("Result",image)
    print ("num of contours detected ", count)
    
    #cv2.drawContours(image, contours, -1, (0,255,0), 3)
    #bounding_boxes = [cv2.boundingRect(contour) for contour in contours]
    #cv2.imwrite('cnt_'+file_name, image)
    #cv2.imshow('detection',image)

    return count,fluidPrecentage, abs(a)


size = 5
path = (r"C:\Users\Asus\dev\bottle_Level_Detector\Source")
if __name__ == '__main__':
    filename = 'testCases/14.jpg'
    filename = os.path.join(path,filename)
    cnt, level, tilt = fluid_level_detect(filename, 75, 100)
    print("")
  
    #80 100
    if cnt > 1:
        print("**** Retry with lower min threshhold *****")
        cv2.destroyAllWindows()
        fluid_level_detect(filename, 60, 100,2)

    if 90 - tilt > 6:
        print("**** Retry with lower min threshhold *****")
        cv2.destroyAllWindows()
        fluid_level_detect(filename, 60, 100,2)

    if level < 10:
        print("**** Retry with lower clustering *****")
        cv2.destroyAllWindows()
        fluid_level_detect(filename, 75, 100,2)

cv2.waitKey(0)


