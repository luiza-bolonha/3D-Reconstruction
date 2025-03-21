import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from scipy import ndimage, misc
from PIL import Image

def plane_sweep_ncc(im_l,im_r,start,steps,wid):
    """ Find disparity image using normalized cross-correlation. """
    m,n = im_l.shape
    
    # arrays to hold the different sums
    mean_l = np.zeros((m,n))
    mean_r = np.zeros((m,n))
    s = np.zeros((m,n))
    s_l = np.zeros((m,n))
    s_r = np.zeros((m,n))
    # array to hold depth planes
    dmaps = np.zeros((m,n,steps))
    
    # compute mean of patch
    ndimage.uniform_filter(im_l,wid,mean_l)
    ndimage.uniform_filter(im_r,wid,mean_r)
    
    # normalized images
    norm_l = im_l - mean_l
    norm_r = im_r - mean_r
    
    # try different disparities
    for displ in range(steps):
      # move left image to the right, compute sums
      
      ndimage.uniform_filter(norm_l*np.roll(norm_r,displ+start),wid,s) # sum nominator
      ndimage.uniform_filter(norm_l*norm_l,wid,s_l)		
      ndimage.uniform_filter(np.roll(norm_r,displ+start)*np.roll(norm_r,displ+start),wid,s_r) # sum denominator
      # store ncc scores			
      dmaps[:,:,displ] = s/np.sqrt(np.absolute(s_l*s_r))
      
      
    # pick best depth for each pixel
    best_map = np.argmax(dmaps,axis=2) + start
    
    return best_map

	
#############
### MAIN ####

# Read images
IL = cv2.imread('esquerda.ppm') # left image
IR = cv2.imread('direita.ppm')  # right image
gray1 = cv2.cvtColor(IL, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(IR, cv2.COLOR_BGR2GRAY)    

print(IL.shape)


# intrinsic parameter matrix
fm = 403.657593 # Focal distantce in pixels
cx = 161.644318 # Principal point - x-coordinate (pixels) 
cy = 124.202080 # Principal point - y-coordinate (pixels) 
bl = 119.929 # baseline (mm)
# for the right camera    
right_k = np.array([[ fm, 0, cx],[0, fm, cy],[0, 0, 1.0000]])

# for the left camera
left_k = np.array([[fm, 0, cx],[0, fm, cy],[0, 0, 1.0000]])

# Extrinsic parameters
# Translation between cameras
T = np.array([-bl, 0, 0]) 
# Rotation
R = np.array([[ 1,0,0],[ 0,1,0],[0,0,1]])


print('Intrinsic Paramenters')
print('Left_K:\n', left_k)
print('Right_K:\n', right_k)

print('Extrinsic Paramenters')
print('R:\n', R)
print('T:\n', T)


# Show images 

fig, ax = plt.subplots(nrows=1, ncols=2)
plt.subplot(1, 2, 1)
plt.imshow(IL,cmap='gray')
plt.subplot(1, 2, 2)
plt.imshow(IR,cmap='gray')
plt.show(block=False)

plt.show()


# MAPA DE DISPARIDADE E PROFUNDIDADE
im_l = np.array(Image.open('esquerda.ppm').convert('L'),'f')
im_r = np.array(Image.open('direita.ppm').convert('L'),'f')
m,n = im_l.shape
steps = 50   
start = 24     
wid1 = 9
res1 = plane_sweep_ncc(im_l,im_r,start,steps,wid1)


fsx = fm*cx
# RECUPERA O Z
Z = np.zeros((m,n))
for i in range(m):
	for j in range(n):
		if (res1[i,j]== 0):
			# Consider Z = inf for points that were not defined in the depthmap and are filled with zero
			Z[i,j] = np.inf 
		else: Z[i,j]=(fsx*bl)/res1[i,j]
        
# RECUPERA X E Y DOS PONTOS DA CAIXA
X,Y = np.meshgrid(np.arange(n),np.arange(m))
X = np.reshape(X, m*n)
Y = np.reshape(Y, m*n)
Z = np.reshape(Z, m*n)


# Select just the estimated depth
estimated = (np.isinf(Z)==False) 

# SPARSE RECONSTRUCTION
xpL = X[estimated][0:-1:5]
ypL = Y[estimated][0:-1:5]

pixel_color=[]

for i in range(xpL.shape[0]):
    pixel_color.append (IL[int(ypL[i]),int(xpL[i])])

pixel_color = np.asarray(pixel_color)

# Show points in a 3D plot with the recovered depth
fig = plt.figure(figsize=[10,8])
ax = fig.add_subplot(projection='3d')
ax.scatter3D(X[estimated][0:-1:5],Y[estimated][0:-1:5],Z[estimated][0:-1:5],c=pixel_color/255.0,marker='.')

ax.view_init(elev=-75,azim=-90)
plt.show()