# PCA-on-Images
Ran PCA algorithm on images . Reconstructed the image using top N (N = 10%, 25%, 50%) principal components( with their corresponding error image)
import numpy as np
from PIL import Image
import random as rand
import matplotlib.pyplot as plt

img = Image.open('C:/Users/Asus/Documents/PRML/39 (1).jpg')
# convert image to grayscale
gray_img = img.convert('LA')
# convert to numpy array
img_mat = np.array(list(gray_img.getdata(band=0)), float)
# Reshape according to orginal image dimensions
img_mat.shape = (gray_img.size[1], gray_img.size[0])
plt.imshow(img_mat, cmap='gray')
plt.show()
Frob=[]
Frob_err=[]
R_img=[]
err_img=[]
Fro=np.linalg.norm(img_mat,'fro')
U, D, V = np.linalg.svd(img_mat)
A=np.flip(np.sort(D))
k=[10,25,50]
c=0
for i in [round(0.1*(img_mat.shape[0])),round(0.25*(img_mat.shape[0])),round(0.5*(img_mat.shape[0]))]:
    rec_img = np.matrix(U[:, :i]) * np.diag(A[:i]) * np.matrix(V[:i, :])
    R_img.append(rec_img)
    plt.imshow(rec_img, cmap='gray')
    title = "n = %s%% Principal Components" % k[c]
    plt.title(title)
    plt.show()
    
    c+=1
    Frob.append([(np.linalg.norm(rec_img,'fro')/Fro)*100])
    Frob_err.append([100-((np.linalg.norm(rec_img,'fro')/Fro)*100)])
print('The quality of the restructured image for 10%,25%,50% of the top principal components is :',Frob)
print('The reconstruction error for 10%,25%,50% of the top principal components is :',Frob_err)
d1=rand.sample(list(D),round(0.1*(img_mat.shape[0])))
recimg1 = np.matrix(U[:, :round(0.1*(img_mat.shape[0]))]) * np.diag(d1[:round(0.1*(img_mat.shape[0]))]) * np.matrix(V[:round(0.1*(img_mat.shape[0])), :])
plt.imshow(recimg1, cmap='gray')
title = "n = %s" % 10 +'%'+'(random principal components)'
plt.title(title)
plt.show()
err_rec=img_mat-recimg1
plt.imshow(err_rec,cmap='gray')
plt.title("Error Image for N=10% Random Principal Components ")
plt.show()
for i in range(3):
    err_img=(img_mat-R_img[i])
    plt.imshow(err_img,cmap='gray')
    s=[10,25,50]
    plt.title("Error Image for N=%s Principal Components "%s[i]+'%')
    plt.show()
Fro_ran=np.linalg.norm(recimg1,'fro')/Fro
print('The quality of the restructured image for 10% random principal components is :',Fro_ran*100)
print('The reconstruction error for 10% random principal components is :',100-(Fro_ran*100))
plt.title("Reconstruction Error Vs N(Number of Principal Components)")
plt.xlabel("N")
plt.ylabel("Reconstruction Error")
plt.plot(k,Frob_err)
plt.grid(True)
plt.show()
