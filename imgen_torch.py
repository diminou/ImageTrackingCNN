import numpy as np
import cv2
from scipy import interpolate
from scipy import ndimage
import scipy
from torch.utils.data import Dataset, DataLoader
import torch

def CreateImage(rescaled_imgsize, scaling_factor):
    #Create background
    big_imgsize = int(rescaled_imgsize / scaling_factor)
    full_imgsize = big_imgsize
    img = np.full((full_imgsize, full_imgsize),
        np.random.randint(230,255),dtype = np.uint8)

    nb_speckles = np.random.randint(1e3,2e3)

    i=0
    while(i<nb_speckles):
        #random ellipse parameters
        center_val = (np.random.randint(0,full_imgsize),np.random.randint(0,full_imgsize))
        axis_len = (np.random.randint(8,18),np.random.randint(8,18))
        color_val = np.random.randint(0,90)
        
        #draw the ellipse
        cv2.ellipse(img,center_val,axis_len,0,0,360,color_val,-1)
        i += 1

    #resize image
    img = cv2.resize(img,(0,0),fx=scaling_factor, fy=scaling_factor,interpolation = 3)
     
    return img


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


def deform_matrix(seed, shape, sigmas, smoothing_sigmas, base_maxes):
    x_location = np.random.randint(shape[0])
    y_location = np.random.randint(shape[1])
    x_displacement = np.random.uniform(-sigmas[0], sigmas[0])
    y_displacement = np.random.uniform(-sigmas[1], sigmas[1])
    dummy_xs = np.zeros(shape)
    dummy_ys = np.zeros(shape)
    dummy_xs[x_location, y_location] = x_displacement
    dummy_ys[x_location, y_location] = y_displacement
    smooth_xs = scipy.ndimage.gaussian_filter(dummy_xs, smoothing_sigmas[0])
    smooth_ys = scipy.ndimage.gaussian_filter(dummy_ys, smoothing_sigmas[1])
    base_max = np.random.uniform(base_maxes[0], base_maxes[1])
    smooth_xs = smooth_xs * (base_max / np.abs(smooth_xs).max())
    smooth_ys = smooth_ys * (base_max / np.abs(smooth_ys).max())
    return (smooth_xs, smooth_ys)


def shift_elements(original, deformations):
    x_def, y_def = deformations
    def shift_func(image):
        return (image[0] - x_def[image], image[1] - y_def[image])
    return ndimage.geometric_transform(original, shift_func)


def generate_pair(seed, displacement_sigmas, smoothing_sigmas, basemaxes, imsize=64, scaling_factor=0.1):
    orig = CreateImage(imsize, scaling_factor)
    deformations = deform_matrix(seed, orig.shape, displacement_sigmas, smoothing_sigmas, basemaxes)
    shifted = shift_elements(orig, deformations)
    return ((orig, shifted), deformations)


def generate_stacked(seed, displacement_sigmas, smoothing_sigmas, basemaxes, imsize=64, scaling_factor=0.1):
    ors, deform = generate_pair(seed, displacement_sigmas, smoothing_sigmas, basemaxes, imsize=imsize, scaling_factor=scaling_factor)
    o, rs = ors
    dx, dy = deform
    return (np.stack([o, rs], axis=0), np.stack([dx, dy], axis=0))


class ImageSequence(Dataset):

    def __init__(self, batch_size,
                 displacement_sigmas, smoothing_sigmas,
                 imsize=64, scaling_factor=0.1,
                 basemaxes=(3.0, 10.0),
                 maxlen=1000):
        self.batch_size = batch_size
        self.maxlen = maxlen
        self.displacement_sigmas = displacement_sigmas
        self.smoothing_sigmas = smoothing_sigmas
        self.imsize = imsize
        self.basemaxes = basemaxes
        self.scaling_factor = scaling_factor

    def __len__(self):
        return self.maxlen

    def __getitem__(self, idx):
        inp, outp = generate_stacked(np.random.randint(1000),
                                     self.displacement_sigmas,
                                     self.smoothing_sigmas,
                                     self.basemaxes,
                                     self.imsize,
                                     self.scaling_factor)
        return (torch.tensor(inp.reshape((2, self.imsize, self.imsize)) / 255., dtype=torch.float32),
                torch.tensor(outp.reshape((2, self.imsize, self.imsize)), dtype=torch.float32))


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
    img_def = img_def.reshape(256, 256).astype('uint8')
    return {'image':img_def, 'disp':mat_fullfield}
