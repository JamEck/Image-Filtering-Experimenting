
# coding: utf-8

# In[1]:


import matplotlib.image  as im
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


def applyKernel(k, img): #img must be 1 layer (single channel)
    kright = int(k.shape[0]/2)
    kdown  = int(k.shape[1]/2)
    kleft  = kright - (k.shape[0]+1)%2
    kup    = kdown  - (k.shape[1]+1)%2
    
    ans = img.copy()
    imgBuf = np.pad(img, ((kleft, kright), (kup, kdown)), 'edge').astype(np.float32)
      
    for i in range(img.size):
        x = int(i%img.shape[0])
        y = int(i/img.shape[0])
        win = imgBuf[x:x+k.shape[0], y:y+k.shape[1]]
        ans[x,y] = (win*k).sum()
    ans = ans.astype(img.dtype)
    return ans


# In[3]:


zac = im.imread("Zac.jpg")
bnw = zac.mean(axis=2)

plt.figure()
plt.imshow(bnw, cmap="gray")

s = 5
kernel = np.ones((s,s))

blur = applyKernel(kernel, bnw)

kernel[1:s-1,1:s-1] = 0

blur2 = applyKernel(kernel, bnw)

kernel[  1:s-1] = 0
kernel[:,1:s-1] = 0

blur3 = applyKernel(kernel, bnw)

plt.figure()
plt.imshow(blur , cmap="gray")

plt.figure()
plt.imshow(blur2, cmap="gray")

plt.figure()
plt.imshow(blur3, cmap="gray")


# In[4]:


def edgeDetect(imgOrig):
    kernel = np.arange(2)+1
    kernel = np.pad(kernel, (0,1), "reflect")
    s = kernel.size
    kernel = kernel*np.ones((s,s))
    s = int(s/2)
    kernel[:s] *= -1
    kernel[s] = 0
    kernel /= np.sum(np.abs(kernel))
    kernel2 = kernel.transpose()    
    
    img = imgOrig.astype(np.float32)/imgOrig.max()

    bottom = applyKernel(kernel, img)
    top    = applyKernel(np.flipud(kernel), img)
    right  = applyKernel(kernel2, img)
    left   = applyKernel(np.fliplr(kernel2), img)
    
    av = np.abs(top)/4.0 + np.abs(bottom)/4.0 + np.abs(left)/4.0 + np.abs(right)/4.0
    av *= 255/av.max()
    av = av.astype(np.uint8)
    return av


# In[5]:


ed = edgeDetect(bnw)

plt.figure()
plt.imshow(bnw)
plt.figure()
plt.imshow(ed)


# In[6]:


coled = np.zeros(zac.shape).astype(np.uint8)

plt.figure()
plt.imshow(zac)

coled[:,:,0] = edgeDetect(zac[:,:,0])
coled[:,:,1] = edgeDetect(zac[:,:,1])
coled[:,:,2] = edgeDetect(zac[:,:,2])

plt.figure()
plt.imshow(coled)


# In[7]:


k = np.zeros((1,20))
k[0,0] = 1
k[0,-1] = 1
k /= k.sum()
seeDouble = np.zeros(coled.shape).astype(np.uint8)

seeDouble[:,:,0] = applyKernel(k, coled[:,:,0])
seeDouble[:,:,1] = applyKernel(k, coled[:,:,1])
seeDouble[:,:,2] = applyKernel(k, coled[:,:,2])

plt.figure()
plt.imshow(seeDouble)


# In[11]:


kernel = np.ones((20,20))/400
redBlur = zac.copy()
redBlur[:,:,0] = applyKernel(kernel, redBlur[:,:,0])

plt.figure()
plt.imshow(zac)
plt.figure()
plt.imshow(redBlur)

