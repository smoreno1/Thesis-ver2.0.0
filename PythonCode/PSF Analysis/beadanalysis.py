import numpy as np
from numpy import mean
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage import data
from skimage import filters
from skimage import feature
import scipy
from tifffile import imread
import seaborn as sns
import pandas as pd
from tifffile import imsave
import ipywidgets as widgets
import matplotlib.cm as cm


def deskew(im, angle):
    imcopy = np.copy(im)
    im = np.zeros((imcopy.shape[0], imcopy.shape[1], imcopy.shape[2]))
    for z in range(imcopy.shape[0]):
        for i in range(imcopy.shape[1]):
            im[z,:,i] = imcopy[z,:,round(i-z*np.cos(45*np.pi/180))]

    imsave("test.tif", im)
    return im

def gaussian(x,a,u,s,b): # gaussian function for fitting
    return a*np.exp((-(x-u)**2)/(2*s**2)) + b

def reject_outliers(data, m = 2.):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else np.zeros(len(d))
    return data[s<m]

def dist(x,y): # distance between two points
    return sum((x-y)**2)**0.5

def nearest_bead(x, beads): # distance from a point to the nearest bead
    distance = [dist(x,y) for y in beads if not (x == y).all()]
    return np.absolute(distance).min(axis=0)

def inside(shape, center, window):
    """
    Returns True if a center and its window is fully contained
    within the shape of the image on all three axes
    """
    return all([(center[i]-window[i] >= 0) & (center[i]+window[i] <= shape[i]) for i in range(0,3)])

def findBeads(im, window, thresh):
    #smoothed = filters.gaussian(im, 0.1, output=None, mode='nearest', cval=0)
    centers = feature.peak_local_max(im, min_distance=3, threshold_abs=thresh, exclude_border=True)
    return centers

def keepBeads(im, window, centres, options): # keeps beads that are a certain distance apart
    centresM = np.array([[x[0]/options['pxPerUmAx'], x[1]/options['pxPerUmLat'], x[2]/options['pxPerUmLat']] for x in centres])
    centreDists = [nearest_bead(x,centresM) for x in centresM]
    keep = np.where([x>10 for x in centreDists])
    centres = centres[keep[0],:]
    keep = np.where([inside(im.shape, x, [5,5,5]) for x in centres])
    return centres[keep[0],:]

def getprofiles(im, centres, window, options, samplename = "", save = False): # extracts pixel values along x, y, and z for fitting
    xprofile = []
    yprofile = []
    zprofile = []

    i = 0
    for x in centres:
        if x[0]-window[0] < 0 or x[1]-window[1] < 0 or x[2]-window[2] < 0 or x[0]+window[0] > np.shape(im)[0] or x[1]+window[1] > np.shape(im)[1] or x[2]+window[2] > np.shape(im)[2]:
            continue
        subvolume = im[(x[0]-window[0]):(x[0]+window[0]), (x[1]-window[1]):(x[1]+window[1]), (x[2]-window[2]):(x[2]+window[2])]
        xprofile.append(subvolume[window[0],window[2],:])
        yprofile.append(subvolume[window[0],:,window[1]])
        zprofile.append(subvolume[:,window[1],window[2]])
        
        if save ==True: # saves images of the individual beads if specified
            imsave(r"./bead_data/{}_{}.tif".format(samplename,i), subvolume)
        i = i + 1
    return xprofile, yprofile, zprofile


def fitpsfs(xprofile, yprofile, zprofile, im, centres, window, options):
    xfit, yfit, zfit = [], [], [] 
    x_r2, y_r2, z_r2 = [], [], []
    x = np.divide(np.subtract(np.arange(0,len(xprofile[0])), window[2]),options["pxPerUmLat"])
    y = np.divide(np.subtract(np.arange(0,len(yprofile[0])), window[1]),options["pxPerUmLat"])
    z = np.divide(np.subtract(np.arange(0,len(zprofile[0])), window[0]),options["pxPerUmAx"])

    i = 0
    for c in centres:
        if c[0]-window[0] < 0 or c[1]-window[1] < 0 or c[2]-window[2] < 0 or c[0]+window[0] > np.shape(im)[0] or c[1]+window[1] > np.shape(im)[1] or c[2]+window[2] > np.shape(im)[2]:
            continue
        try:
            xparam, cov = scipy.optimize.curve_fit(gaussian, x, xprofile[i])
            yparam, cov = scipy.optimize.curve_fit(gaussian, y, yprofile[i])
            zparam, cov = scipy.optimize.curve_fit(gaussian, z, zprofile[i])

            xfit.append(xparam)
            yfit.append(yparam)
            zfit.append(zparam)

            # Calculate R² for the fits
            x_fit_values = gaussian(x, *xparam)
            x_residuals = xprofile[i] - x_fit_values
            x_ss_res = np.sum(x_residuals**2)
            x_ss_tot = np.sum((xprofile[i] - np.mean(xprofile[i]))**2)
            x_r2.append(1 - (x_ss_res / x_ss_tot))

            y_fit_values = gaussian(y, *yparam)
            y_residuals = yprofile[i] - y_fit_values
            y_ss_res = np.sum(y_residuals**2)
            y_ss_tot = np.sum((yprofile[i] - np.mean(yprofile[i]))**2)
            y_r2.append(1 - (y_ss_res / y_ss_tot))

            z_fit_values = gaussian(z, *zparam)
            z_residuals = zprofile[i] - z_fit_values
            z_ss_res = np.sum(z_residuals**2)
            z_ss_tot = np.sum((zprofile[i] - np.mean(zprofile[i]))**2)
            z_r2.append(1 - (z_ss_res / z_ss_tot))

            i = i + 1
        except: # stops the code from breaking if a gaussian can't be fit
            xprofile.pop(i)
            yprofile.pop(i)
            zprofile.pop(i)
    return xfit, yfit, zfit, x, y, z, xprofile, yprofile, zprofile, x_r2, y_r2, z_r2 #return xfit, yfit, zfit, xyfit, xzfit, yzfit, x, y, z, xy, xz, yz



def plotpsfs(xfit, yfit, zfit, x, y, z, xprof, yprof, zprof, x_r2, y_r2, z_r2, index): # plots fits and profiles in x, y, and z

    # calculate fwhm of gaussian fits
    xstdev = [a[2] for a in xfit]
    ystdev = [a[2] for a in yfit]
    zstdev = [a[2] for a in zfit]

    xfwhm = [abs(2.355*a) for a in xstdev]
    yfwhm = [abs(2.355*a) for a in ystdev]
    zfwhm = [abs(2.355*a) for a in zstdev]

    xfwhm = [v for v in xfwhm if v > 0]
    yfwhm = [v for v in yfwhm if v > 0]
    zfwhm = [v for v in zfwhm if v > 0]

    fig, axs = plt.subplots(3, 1, figsize=(10,10))

    # Plot xfit
    axs[0].scatter(x - xfit[index][1], xprof[index], marker='x')
    axs[0].plot(np.linspace(x[0], x[len(x)-1], 1000), gaussian(np.linspace(x[0], x[len(x)-1], 1000), xfit[index][0], 0, xfit[index][2], xfit[index][3]))
    axs[0].set_title('X Fit, FWHM={:.2f}, R^2={:.2f}'.format(xfwhm[index], x_r2[index]))

    # Plot yfit
    axs[1].scatter(y - yfit[index][1], yprof[index], marker='x')
    axs[1].plot(np.linspace(y[0], y[len(y)-1], 1000), gaussian(np.linspace(y[0], y[len(y)-1], 1000), yfit[index][0], 0, yfit[index][2], yfit[index][3]))
    axs[1].set_title('Y Fit, FWHM={:.2f}, R^2={:.2f}'.format(yfwhm[index], y_r2[index]))

    # Plot zfit
    axs[2].scatter(z - zfit[index][1], zprof[index], marker='x')
    axs[2].plot(np.linspace(z[0], z[len(z)-1], 1000), gaussian(np.linspace(z[0], z[len(z)-1], 1000), zfit[index][0], 0, zfit[index][2], zfit[index][3]))
    axs[2].set_title('Z Fit, FWHM={:.2f}, R^2={:.2f}'.format(zfwhm[index], z_r2[index]))

    plt.tight_layout()
    plt.show()

    del xfwhm[len(xfwhm)-round(percent*len(xfwhm)/100):]
    del yfwhm[len(yfwhm)-round(percent*len(yfwhm)/100):]
    del zfwhm[len(zfwhm)-round(percent*len(zfwhm)/100):]

    


def everything(im, window, options, samplename="", save=False): # a function that does the whole sampling and fitting by collating the previous functions
    centres = findBeads(im, window, options["thresh"])
    xprof, yprof, zprof = getprofiles(im, centres, window, options, samplename, save)
    xfit, yfit, zfit, x, y, z, xprof, yprof, zprof, x_r2, y_r2, z_r2 = fitpsfs(xprof, yprof, zprof, im, centres, window, options)
    return centres, x, y, z, xfit, yfit, zfit, xprof, yprof, zprof, x_r2, y_r2, z_r2

def singleviolins(xfit, yfit, zfit, x_r2, y_r2, z_r2, rmin=0.85):
    #calculate fwhm
    xstdev = [a[2] for a in xfit]
    ystdev = [a[2] for a in yfit]
    zstdev = [a[2] for a in zfit]

    xfwhm = [abs(2.355*a) for a in xstdev]
    yfwhm = [abs(2.355*a) for a in ystdev]
    zfwhm = [abs(2.355*a) for a in zstdev]

    #R² filtering
    original = len(xfwhm)
    removed = 0
    for i in range(0, len(xfwhm)):
        if x_r2[i] < rmin:
            xfwhm.pop(i-removed)
            yfwhm.pop(i-removed)
            zfwhm.pop(i-removed)
            removed = removed + 1
        elif y_r2[i] < rmin:
            xfwhm.pop(i-removed)
            yfwhm.pop(i-removed)
            zfwhm.pop(i-removed)
            removed = removed + 1
        elif z_r2[i] < rmin:
            xfwhm.pop(i-removed)
            yfwhm.pop(i-removed)
            zfwhm.pop(i-removed)
            removed = removed + 1

    # rejects outliers
    xfwhm = np.array(xfwhm)
    yfwhm = np.array(yfwhm)
    zfwhm = np.array(zfwhm)

    xfwhm = reject_outliers(xfwhm)
    yfwhm = reject_outliers(yfwhm)
    zfwhm = reject_outliers(zfwhm)

    xfwhm = xfwhm.tolist()
    yfwhm = yfwhm.tolist()
    zfwhm = zfwhm.tolist()

    data = {
    'Value': xfwhm + yfwhm + zfwhm,
    'Direction': ['X'] * len(xfwhm) + ['Y'] * len(yfwhm) + ['Z'] * len(zfwhm)
    }

    df = pd.DataFrame(data)
    #violin plot
    plt.figure(figsize=(7, 7))
    sns.violinplot(x='Direction', y='Value', data=df, color="lightslategray", inner='box', inner_kws={'color': 'black'})
    plt.xlabel('Direction', fontsize = 18)
    plt.ylabel('FWHM (μm)', fontsize = 18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print("Totals over whole Image")
    print("-------------------------------------")
    print("X FWHM:{:.2f}+-{:.2f}".format(np.mean(xfwhm), np.std(xfwhm)))
    print("Y FWHM:{:.2f}+-{:.2f}".format(np.mean(yfwhm), np.std(yfwhm)))
    print("Z FWHM:{:.2f}+-{:.2f}".format(np.mean(zfwhm), np.std(zfwhm)))
    print("")
    print("Got rid of {:.2f}% of the beads, {}".format((removed/original)*100, len(xfwhm)))
    return xfwhm, yfwhm, zfwhm



def doubleviolins(xfit, yfit, zfit, nxfit, nyfit, nzfit, x_r2, y_r2, z_r2, nx_r2, ny_r2, nz_r2, rmin=0.85): # same thing but plots a double sided violin but to compare two different samples
    xstdev = [a[2] for a in xfit]
    ystdev = [a[2] for a in yfit]
    zstdev = [a[2] for a in zfit]
    nxstdev = [a[2] for a in nxfit]
    nystdev = [a[2] for a in nyfit]
    nzstdev = [a[2] for a in nzfit]

    xfwhm = [abs(2.355*a) for a in xstdev]
    yfwhm = [abs(2.355*a) for a in ystdev]
    zfwhm = [abs(2.355*a) for a in zstdev]
    nxfwhm = [abs(2.355*a) for a in nxstdev]
    nyfwhm = [abs(2.355*a) for a in nystdev]
    nzfwhm = [abs(2.355*a) for a in nzstdev]


    original = len(xfwhm)
    removed = 0
    for i in range(0, len(xfwhm)):
        if x_r2[i] < rmin:
            xfwhm.pop(i-removed)
            yfwhm.pop(i-removed)
            zfwhm.pop(i-removed)
            removed = removed + 1
        elif y_r2[i] < rmin:
            xfwhm.pop(i-removed)
            yfwhm.pop(i-removed)
            zfwhm.pop(i-removed)
            removed = removed + 1
        elif z_r2[i] < rmin:
            xfwhm.pop(i-removed)
            yfwhm.pop(i-removed)
            zfwhm.pop(i-removed)
            removed = removed + 1

    original = len(nxfwhm)
    removed = 0
    for i in range(0, len(nxfwhm)):
        if nx_r2[i] < rmin:
            nxfwhm.pop(i-removed)
            nyfwhm.pop(i-removed)
            nzfwhm.pop(i-removed)
            removed = removed + 1
        elif ny_r2[i] < rmin:
            nxfwhm.pop(i-removed)
            nyfwhm.pop(i-removed)
            nzfwhm.pop(i-removed)
            removed = removed + 1
        elif nz_r2[i] < rmin:
            nxfwhm.pop(i-removed)
            nyfwhm.pop(i-removed)
            nzfwhm.pop(i-removed)
            removed = removed + 1


    xfwhm = np.array(xfwhm)
    yfwhm = np.array(yfwhm)
    zfwhm = np.array(zfwhm)
    nxfwhm = np.array(nxfwhm)
    nyfwhm = np.array(nyfwhm)
    nzfwhm = np.array(nzfwhm)

    xfwhm = reject_outliers(xfwhm)
    yfwhm = reject_outliers(yfwhm)
    zfwhm = reject_outliers(zfwhm)
    nxfwhm = reject_outliers(nxfwhm)
    nyfwhm = reject_outliers(nyfwhm)
    nzfwhm = reject_outliers(nzfwhm)

    xfwhm = xfwhm.tolist()
    yfwhm = yfwhm.tolist()
    zfwhm = zfwhm.tolist()
    nxfwhm = nxfwhm.tolist()
    nyfwhm = nyfwhm.tolist()
    nzfwhm = nzfwhm.tolist()

    data = {
    'Value': xfwhm + nxfwhm + yfwhm + nyfwhm + zfwhm + nzfwhm,
    'Method': ['Shear'] * len(xfwhm) + ['No Shear'] * len(nxfwhm) +
            ['Shear'] * len(yfwhm) + ['No Shear'] * len(nyfwhm) +
            ['Shear'] * len(zfwhm) + ['No Shear'] * len(nzfwhm),
    'Direction': ['X'] * len(xfwhm) + ['X'] * len(nxfwhm) +
                ['Y'] * len(yfwhm) + ['Y'] * len(nyfwhm) +
                ['Z'] * len(zfwhm) + ['Z'] * len(nzfwhm)
    }

    df = pd.DataFrame(data)

    plt.figure(figsize=(7, 7))
    sns.violinplot(x='Direction', y='Value', hue='Method', data=df, split=True, palette=["royalblue", "gold"], inner='box', inner_kws={'color': 'black'}, density_norm='area', common_norm=True)
    plt.xlabel('Direction', fontsize = 18)
    plt.ylabel('FWHM (μm)', fontsize = 18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    #plt.ylim(0, 18)
    plt.legend(fontsize = 18)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print("Shear avgs")
    print("-------------------------------------")
    print("X FWHM:{:.2f} +- {:.2f}".format(np.mean(xfwhm), np.std(xfwhm)))
    print("Y FWHM:{:.2f} +- {:.2f}".format(np.mean(yfwhm), np.std(yfwhm)))
    print("Z FWHM:{:.2f} +- {:.2f}".format(np.mean(zfwhm), np.std(zfwhm)))
    print("")
    print("N = {}".format(len(xfwhm)))

    print("No Shear avgs")
    print("-------------------------------------")
    print("X FWHM:{:.2f} +- {:.2f}".format(np.mean(nxfwhm), np.std(nxfwhm)))
    print("Y FWHM:{:.2f} +- {:.2f}".format(np.mean(nyfwhm), np.std(nyfwhm)))
    print("Z FWHM:{:.2f} +- {:.2f}".format(np.mean(nzfwhm), np.std(nzfwhm)))
    print("")
    print("N = {}".format(len(nxfwhm)))


    


