from scipy import misc
import imageio

import matplotlib.pyplot as plt
import numpy as np
import math


folderName = 'line/'
imageName = 'pentagon.png'
method = 0 # 0 - sequential, 1 - threads, 2 - distributed
#----------------------------------------------------------------------------------------#
# Step 1: read image
color = 255

img = imageio.imread(imageName)

print('image shape: ', img.shape)
print(img[64,64,0],img[64,64,1],img[64,64,2],img[64,64,3])

plt.imshow(img, )

plt.savefig(folderName + "image.png",bbox_inches='tight')

plt.close()


def sequential(x_max,y_max,img,color,theta_dim,theta_max,r_dim,r_max):
    hough_space = np.zeros((r_dim, theta_dim))

    for x in range(x_max):
        for y in range(y_max):
            if img[x, y, 0] == color: continue
            for itheta in range(theta_dim):
                theta = 1.0 * itheta * theta_max / theta_dim
                r = x * math.cos(theta) + y * math.sin(theta)
                ir = r_dim * (1.0 * r) / r_max
                hough_space[int(ir), itheta] = hough_space[int(ir), itheta] + 1
    return hough_space


import asyncio



async def threads(x,y_max,img,color,theta_dim,theta_max,r_dim,r_max,hough_space):
    for y in range(y_max):
        if img[x, y, 0] == color: continue
        for itheta in range(theta_dim):
            theta = 1.0 * itheta * theta_max / theta_dim
            r = x * math.cos(theta) + y * math.sin(theta)
            ir = r_dim * (1.0 * r) / r_max
            hough_space[int(ir), itheta] = hough_space[int(ir), itheta] + 1


async def parallel(x_max,y_max,img,color,theta_dim,theta_max,r_dim,r_max):
    # Schedule three calls *concurrently*:
    hough_space = np.zeros((r_dim, theta_dim))
    await asyncio.gather(*[threads(x,y_max,img,color,theta_dim,theta_max,r_dim,r_max,hough_space) for x in range(x_max)])
    return hough_space

from mpi4py import MPI

def seqforx(x_start,x_end,y_max,img,color,theta_dim,theta_max,r_dim,r_max):
    hough_space = np.zeros((r_dim, theta_dim))
    for x in range(x_start,x_end):
        for y in range(y_max):
            if img[x, y, 0] == color: continue
            for itheta in range(theta_dim):
                theta = 1.0 * itheta * theta_max / theta_dim
                r = x * math.cos(theta) + y * math.sin(theta)
                ir = r_dim * (1.0 * r) / r_max
                hough_space[int(ir), itheta] = hough_space[int(ir), itheta] + 1
    if MPI.COMM_WORLD.Get_rank() == 0: return hough_space
    MPI.COMM_WORLD.send(hough_space, dest=0, tag=98)

def distributed(x_max,y_max,img,color,theta_dim,theta_max,r_dim,r_max):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    hough_space = np.zeros((r_dim,theta_dim))
    xrange = x_max//size
    allotedx = rank*xrange
    if rank == 0:
        hough_space = seqforx(allotedx,allotedx+xrange,y_max,img,color,theta_dim,theta_max,r_dim,r_max)
    else : seqforx(allotedx,allotedx+xrange,y_max,img,color,theta_dim,theta_max,r_dim,r_max)

    if rank == 0:
        for i in range(1, size):
            data = comm.recv(source=i,tag=98)
            hough_space = np.add(hough_space,data)
        return hough_space





#----------------------------------------------------------------------------------------#
# Step 2: Hough Space

img_shape = img.shape

x_max = img_shape[0]
y_max = img_shape[1]

theta_max = 1.0 * math.pi
theta_min = 0.0

r_min = 0.0
r_max = math.hypot(x_max, y_max)

r_dim = 200
theta_dim = 300

if method == 0:
    hough_space = sequential(x_max,y_max,img,color,theta_dim,theta_max,r_dim,r_max)
elif method == 1:
    hough_space = asyncio.run(parallel(x_max,y_max,img,color,theta_dim,theta_max,r_dim,r_max))
else:
    hough_space = distributed(x_max,y_max,img,color,theta_dim,theta_max,r_dim,r_max)

if MPI.COMM_WORLD.Get_rank() == 0:
    plt.imshow(hough_space, origin='lower')
    plt.xlim(0,theta_dim)
    plt.ylim(0,r_dim)

    tick_locs = [i for i in range(0,theta_dim,40)]
    tick_lbls = [round( (1.0 * i * theta_max) / theta_dim,1) for i in range(0,theta_dim,40)]
    plt.xticks(tick_locs, tick_lbls)

    tick_locs = [i for i in range(0,r_dim,20)]
    tick_lbls = [round( (1.0 * i * r_max ) / r_dim,1) for i in range(0,r_dim,20)]
    plt.yticks(tick_locs, tick_lbls)

    plt.xlabel(r'Theta')
    plt.ylabel(r'r')
    plt.title('Hough Space')

    plt.savefig(folderName + "hough_space_r_theta.png",bbox_inches='tight')

    plt.close()

#----------------------------------------------------------------------------------------#
# Find maximas 1
'''
Sorted_Index_HoughTransform =  np.argsort(hough_space, axis=None)

print 'Sorted_Index_HoughTransform[0]', Sorted_Index_HoughTransform[0]
#print Sorted_Index_HoughTransform.shape, r_dim * theta_dim

shape = Sorted_Index_HoughTransform.shape

k = shape[0] - 1
list_r = []
list_theta = []
for d in range(5):
    i = int( Sorted_Index_HoughTransform[k] / theta_dim )
    #print i, round( (1.0 * i * r_max ) / r_dim,1)
    list_r.append(round( (1.0 * i * r_max ) / r_dim,1))
    j = Sorted_Index_HoughTransform[k] - theta_dim * i
    print 'Maxima', d+1, 'r: ', j, 'theta', round( (1.0 * j * theta_max) / theta_dim,1)
    list_theta.append(round( (1.0 * j * theta_max) / theta_dim,1))
    print "--------------------"
    k = k - 1


#theta = list_theta[7]
#r = list_r[7]

#print " r,theta",r,theta, math.degrees(theta)
'''
#----------------------------------------------------------------------------------------#
# Step 3: Find maximas 2

import scipy.ndimage.filters as filters
import scipy.ndimage as ndimage

if MPI.COMM_WORLD.Get_rank() == 0:
    neighborhood_size = 20
    threshold = 140

    data_max = filters.maximum_filter(hough_space, neighborhood_size)
    maxima = (hough_space == data_max)


    data_min = filters.minimum_filter(hough_space, neighborhood_size)
    diff = ((data_max - data_min) > threshold)
    maxima[diff == 0] = 0

    labeled, num_objects = ndimage.label(maxima)
    slices = ndimage.find_objects(labeled)

    x, y = [], []
    for dy,dx in slices:
        x_center = (dx.start + dx.stop - 1)/2
        x.append(x_center)
        y_center = (dy.start + dy.stop - 1)/2
        y.append(y_center)

    print(x)
    print(y)

    plt.imshow(hough_space, origin='lower')
    plt.savefig(folderName + 'hough_space_i_j.png', bbox_inches = 'tight')

    plt.autoscale(False)
    plt.plot(x,y, 'ro')
    plt.savefig(folderName + 'hough_space_maximas.png', bbox_inches = 'tight')

    plt.close()

    #----------------------------------------------------------------------------------------#
    # Step 4: Plot lines

    line_index = 1

    for i,j in zip(y, x):

        r = round( (1.0 * i * r_max ) / r_dim,1)
        theta = round( (1.0 * j * theta_max) / theta_dim,1)

        fig, ax = plt.subplots()

        ax.imshow(img)

        ax.autoscale(False)

        px = []
        py = []
        for i in range(-y_max-40,y_max+40,1):
            px.append( math.cos(-theta) * i - math.sin(-theta) * r )
            py.append( math.sin(-theta) * i + math.cos(-theta) * r )

        ax.plot(px,py, linewidth=10)

        plt.savefig(folderName + "image_line_"+ "%02d" % line_index +".png",bbox_inches='tight')

        #plt.show()

        plt.close()

        line_index = line_index + 1

#----------------------------------------------------------------------------------------#
# Plot lines
'''
i = 11
j = 264

i = y[1]
j = x[1]

print i,j

r = round( (1.0 * i * r_max ) / r_dim,1)
theta = round( (1.0 * j * theta_max) / theta_dim,1)

print 'r', r
print 'theta', theta


fig, ax = plt.subplots()

ax.imshow(img)

ax.autoscale(False)

px = []
py = []
for i in range(-y_max-40,y_max+40,1):
    px.append( math.cos(-theta) * i - math.sin(-theta) * r ) 
    py.append( math.sin(-theta) * i + math.cos(-theta) * r )

print px
print py

ax.plot(px,py, linewidth=10)

plt.savefig("PlottedLine_07.png",bbox_inches='tight')

#plt.show()

'''