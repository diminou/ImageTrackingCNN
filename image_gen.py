import numpy as np
import cv2
from scipy import interpolate

def create_image():
    #Create background
    img = np.full((2560,2560), np.random.randint(230,255),dtype = np.uint8)
    nb_speckles = np.random.randint(5e3,7e3)
    i=0
    while(i<nb_speckles):
        #random ellipse parameters
        center_val = (np.random.randint(0,2560),np.random.randint(0,2560))
        axis_len = (np.random.randint(8,18),np.random.randint(8,18))
        color_val = np.random.randint(0,90)
        #draw the ellipse
        cv2.ellipse(img,center_val,axis_len,0,0,360,color_val,-1)
        i += 1
    #resize image
    img = cv2.resize(img,(0,0),fx=0.1,fy=0.1,interpolation = 3)
    returmg


# The shape function is loosly defined and is only valid for reactangular 4-nodes quad elements.
# It can be transfered to generic 4-nodes quad by revising the calculation of eps and nu.
def four_node_quad(posnodes, dispnodes, X, Y):
    width = posnodes[1][0] - posnodes[0][0]
    height = width = posnodes[3][1] - posnodes[0][1]
    mult = 0.25
    eps = X/(width/2) - (posnodes[0][0] + 1)
    nu = Y/(height/2) - (posnodes[0][1] + 1)
    #definition of each node shape function value
    shp_f1 = mult * (1-eps) * (1-nu)
    shp_f2 = mult * (1+eps) * (1-nu)
    shp_f3 = mult * (1+eps) * (1+nu)
    shp_f4 = mult * (1-eps) * (1+nu)
    # computation of displacement at target point
    res_disp = [np.sum([shp_f1*dispnodes[0][0], shp_f2*dispnodes[1][0], shp_f3*dispnodes[2][0], shp_f4*dispnodes[3][0]],
                       axis=0),
                np.sum([shp_f1*dispnodes[0][1], shp_f2*dispnodes[1][1], shp_f3*dispnodes[2][1], shp_f4*dispnodes[3][1]],
                        axis=0)]
    return res_disp


def distort_image(img, method='rand_disp', met_interp='linear', disps=None):
    #assume initial node positions from image dims
    img_dims = np.array(img.shape)
    posnodes = [[0,0], [img_dims[0],0], [img_dims[0],img_dims[1]], [0,img_dims[1]]]
    if(method=="rand_disp"):
        #draw displacement values in pixels
        disps = np.round(np.random.randn(4,2)*5,1)
    #compute displacement of each pixel of the image
    x = np.arange(0, img_dims[0], 1)
    y = np.arange(0, img_dims[1], 1)
    xv, yv = np.meshgrid(x, y)
    mat_fullfield = four_node_quad(posnodes, disps, xv, yv)
    #compute new pixels positions
    xv_new = xv + mat_fullfield[0]
    yv_new = yv + mat_fullfield[1]
    #interpolate new image
    img_def = interpolate.griddata(points=np.array([xv_new.flatten(), yv_new.flatten()]).transpose(), 
                                   values=img.flatten(), 
                                   xi=np.array([xv.flatten(), yv.flatten()]).transpose(),
                                   method=met_interp,
                                   fill_value=0)
    img_def = img_def.reshape(256,256).astype('uint8')
    return {'image':img_def, 'disp':mat_fullfield}
