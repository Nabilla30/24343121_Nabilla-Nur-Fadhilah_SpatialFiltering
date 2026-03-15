import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import mean_squared_error
from skimage.metrics import structural_similarity


# =====================================
# LOAD IMAGE
# =====================================

img = cv2.imread("Crombol.jpeg",0)

if img is None:
    print("Gambar tidak ditemukan!")
    exit()

print("Ukuran gambar:", img.shape)


# =====================================
# TAMBAH NOISE
# =====================================

def gaussian_noise(image):

    row,col = image.shape
    mean = 0
    sigma = 25

    gauss = np.random.normal(mean,sigma,(row,col))
    noisy = image + gauss

    noisy = np.clip(noisy,0,255)

    return noisy.astype(np.uint8)


def salt_pepper_noise(image):

    out = image.copy()

    prob = 0.04

    rnd = np.random.rand(*image.shape)

    out[rnd < prob/2] = 0
    out[rnd > 1 - prob/2] = 255

    return out


def speckle_noise(image):

    row,col = image.shape

    gauss = np.random.randn(row,col)

    noisy = image + image * gauss

    noisy = np.clip(noisy,0,255)

    return noisy.astype(np.uint8)


gaussian_img = gaussian_noise(img)
sp_img = salt_pepper_noise(img)
speckle_img = speckle_noise(img)


# =====================================
# FILTER
# =====================================

def mean_filter(image,k):
    return cv2.blur(image,(k,k))


def gaussian_filter(image,sigma):
    return cv2.GaussianBlur(image,(5,5),sigma)


def median_filter(image,k):
    return cv2.medianBlur(image,k)


def max_filter(image,k):
    kernel = np.ones((k,k),np.uint8)
    return cv2.dilate(image,kernel)


# =====================================
# EVALUASI METRIK
# =====================================

def evaluate(original,filtered):

    mse = mean_squared_error(original,filtered)

    psnr = peak_signal_noise_ratio(original,filtered)

    ssim = structural_similarity(original,filtered)

    return mse,psnr,ssim


# =====================================
# DAFTAR FILTER
# =====================================

filters = {

"Mean 3x3": lambda x: mean_filter(x,3),
"Mean 5x5": lambda x: mean_filter(x,5),

"Gaussian sigma1": lambda x: gaussian_filter(x,1),
"Gaussian sigma2": lambda x: gaussian_filter(x,2),

"Median 3x3": lambda x: median_filter(x,3),
"Median 5x5": lambda x: median_filter(x,5),

"Max 3x3": lambda x: max_filter(x,3)

}


noise_images = {

"Gaussian Noise": gaussian_img,
"SaltPepper Noise": sp_img,
"Speckle Noise": speckle_img

}


# =====================================
# PROSES FILTERING DAN EVALUASI
# =====================================

results = []

for noise_name,noise_img in noise_images.items():

    for filter_name,f in filters.items():

        start = time.time()

        filtered = f(noise_img)

        end = time.time()

        mse,psnr,ssim = evaluate(img,filtered)

        comp_time = end-start

        results.append([noise_name,filter_name,mse,psnr,ssim,comp_time])


# =====================================
# CETAK TABEL HASIL
# =====================================

print("\nTABEL HASIL EVALUASI\n")

print("----------------------------------------------------------------------------")
print("| {:<15} | {:<15} | {:<10} | {:<10} | {:<10} | {:<8} |".format(
"Noise","Filter","MSE","PSNR","SSIM","Time"))
print("----------------------------------------------------------------------------")

for r in results:

    noise,filter_name,mse,psnr,ssim,time_comp = r

    print("| {:<15} | {:<15} | {:<10.2f} | {:<10.2f} | {:<10.3f} | {:<8.5f} |".format(
    noise,filter_name,mse,psnr,ssim,time_comp))

print("----------------------------------------------------------------------------")


# =====================================
# VISUALISASI HASIL
# =====================================

plt.figure(figsize=(12,8))

plt.subplot(2,2,1)
plt.title("Original")
plt.imshow(img,cmap='gray')
plt.axis("off")

plt.subplot(2,2,2)
plt.title("Gaussian Noise")
plt.imshow(gaussian_img,cmap='gray')
plt.axis("off")

plt.subplot(2,2,3)
plt.title("Salt Pepper Noise")
plt.imshow(sp_img,cmap='gray')
plt.axis("off")

plt.subplot(2,2,4)
plt.title("Speckle Noise")
plt.imshow(speckle_img,cmap='gray')
plt.axis("off")

plt.tight_layout()
plt.show()