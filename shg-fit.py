#!/usr/bin/env python3
from numpy.random import uniform, seed
from os import listdir
from os.path import isfile, join
from scipy import signal
from scipy.interpolate import griddata
from scipy.interpolate import interp1d
from tqdm import tqdm
import cmath
import math
import matplotlib
import matplotlib.colors as mc
import matplotlib.pyplot as plt
import numpy as np
import os
import peakutils
import scipy
import sys
from symfit import Parameter, Variable, re, im, I, Fit, Model, Ge, Eq,latex
from symfit.contrib.interactive_guess import InteractiveGuess2D

cyl_R = 15.7
space = 0.15 # mm, additional space between ceramic disks in stack
#isS1P=True
isS1P=False

# isAllSpecs = True
isAllSpecs = False

isFilterQ = True
#isFilterQ = False

isMakeFit = True
# isMakeFit = False

spectra_dir = "data";
from_n = 0; to_n = 1

output_dir = spectra_dir+"_output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
print(output_dir)
speed = 299792458

Gfac = 2

def get_estimated_peaks_and_widths(x, y, plt_ax):
    # span = int(x.size/20) # Moving average size (to substract as background)
    span = int(x.size/10) # Moving average size (to substract as background)
    # make span to be even number for correct padding and convolution
    span = span+1 if span%2 == 0 else span 
    #peak_thres = 0.24
    peak_thres = 0.05
    #peak_thres = 0.253

    # span = 181 
    # peak_thres = 0.25
    #scipy filter
    b, a = signal.butter(1, 0.06)
    y_butter = signal.filtfilt(b, a, y)
        # Analog of Matalb smooth - moving average filter.
    y_pad = np.pad(y, int(span/2), 'edge')
    y_aver = np.convolve(y_pad, np.ones((span,))/span, mode='valid')
    # print(len(y), len(y_aver))
    y_filtered=y_butter-y_aver # filter some noise and background

    idxs = peakutils.indexes(y_filtered,
                             thres=peak_thres,
                             #min_dist=51
    )
    max_try = 10000
    test_thres = peak_thres
    for i in range(max_try):
        if len(idxs) < 2: break
        test_thres *= 1.002
        idxs = peakutils.indexes(y_filtered,
                             thres=test_thres,
                             #min_dist=51
                             )
        # print(i, test_thres, len(idxs))

    #get estimates of peak widths
    sign_switch = np.pad((np.diff(np.sign(y_filtered)) != 0)*1, (0,1), 'constant')
    peak_width_mask = np.zeros(sign_switch.shape)
    peak_widths = []
    idx_widths = []
    for i in idxs:
        j=0
        width = 0.
        idx_width = 0
        factor = 3
        while (j*factor+i+1 < len(x)):
            j+=1
            if sign_switch[i+j] == 0: continue
            if i-j*factor < 0: continue
            idx_width = int(j*factor*1)
            peak_width_mask[i-idx_width] = i
            peak_width_mask[i+idx_width] = i
            width = x[i+idx_width]-x[i-idx_width]
            break
        peak_widths.append(width)
        idx_widths.append(idx_width)
    # print(peak_widths)
    x_switch = x[peak_width_mask>0]
    y_switch = y_filtered[peak_width_mask>0]

    plt_ax.plot(x, y_filtered, lw=0.3, color = "black")
    plt_ax.plot(x, y, lw=0.3, marker='o', markersize=0.8, alpha=0.5, color = "blue")
    # plt_ax.plot(x, y_aver, lw=0.5, color = "cyan")
    plt_ax.scatter(x[idxs],y_filtered[idxs], s=30, alpha=0.8, c='r')
    plt_ax.scatter(x[idxs],y[idxs], s=30, alpha=0.5, c='r')
    plt_ax.scatter(x_switch,y_switch, s=30, alpha=0.5, c='b')
    #plt.show()
    return idxs[0], peak_widths, idx_widths

def multi_savetxt(output_dir, in_var):
    for pair in in_var:
        np.savetxt(output_dir+"/"+pair[1], pair[0])

def multi_loadtxt(dir, filelist):
    output = ()
    for fname in filelist:
        out = np.loadtxt(dir+"/"+fname)
        output += (out,)
    return output

def set_pbar(total, desc):
    return tqdm(total=total, 
                desc=desc, 
                file=sys.stdout,
                bar_format='{l_bar}{bar}| elapsed: {elapsed} remaining: {remaining}')
def read_data_from_txt(dir_name, fname):
    freq = np.loadtxt(dir_name+'/'+'frequency.txt')
    ratio = np.loadtxt(dir_name+'/'+'ratio.txt')
    ratio = ratio[rlcrop1:rlcrop2]
    data = np.loadtxt(dir_name+'/'+fname)
    print('data',data.shape)
    data_list = []
    data_h = []
    imax = data.shape[-1]
    for i in range(imax):
        if i < int(imax*1/7) or i > int(imax*2.5/7): continue
        d = data[:,i]
        dd = np.vstack((freq,d,d,d*0)).T
        data_list.append(dd)
        data_h.append(cyl_R/ratio[i])
    return np.array(data_list),np.array([]),np.array(data_h)

def read_cst_data(dir_name, cst_data_file):
    fout =''
    from fs import open_fs
    from fs.walk import Walker
    home_fs = open_fs('./'+dir_name)
    if isS1P: walker = Walker(filter=['*.s1p'])
    else: walker = Walker(filter=['*.s2p'])
    for path in walker.files(home_fs):
        return
    with open(dir_name+'/'+cst_data_file, 'r') as f:
        for line in f:
            if len(line) < 2: continue
            if line.startswith('Parameters'):
                split = line.split()
                for s in split:
                    if s.startswith('ratio'):
                        fout = str(cyl_R/float(s[6:-1]))+'.s1p'
                        print(fout)
            if '=' in line: continue
            if len(fout) == 0: continue
            with open(dir_name+'/'+fout, 'a') as f:
                f.write(line)
        # print(line)

def read_data_from_files(dir_name, isAllSpecs):
    data_names = []
    data_h=[]
    import os
    for filename in os.listdir(dir_name):
        path = dir_name+"/"+filename
        if path.count('/') > 1: continue
        try:
            tt=float(filename[6:-4])
            # print(tt)
        except:
            continue
        data_names.append(filename)
        data_h.append(tt)
    # data_names.sort()
    data_val_names = np.array([[x, y] for x,y in sorted(zip(data_h,data_names))]).T
    data_h = [float(x) for x in data_val_names[0,:]]
    data_names = data_val_names[1,:]
    # print(data_val_names)
    # data_h = sorted(data_h)

    # print(data_names)
    # print(data_h)
    if not isAllSpecs:
        data_names = data_names[from_n:to_n]
        data_h = data_h[from_n:to_n]
    print(data_names, data_h)
    data_list = []
    pbar = set_pbar(total=len(data_names), desc='Reading data        ')
    for name in data_names:
        print(name)
        # data = np.loadtxt(dir_name+"/"+name, delimiter='\t', comments=['#','!'], usecols=[0,1]) #for matlab
        data = np.loadtxt(dir_name+"/"+name, delimiter='\t', comments=['#','!'], skiprows=3)
        # data = np.loadtxt(dir_name+"/"+name, comments=['#','!'], delimiter='\t')
        # data = np.loadtxt(dir_name+"/"+name, comments=['#','!'], delimiter=',')
        # data[:,2] = 1. - data[:,2]
        data_list.append(data)
        pbar.update()
    pbar.close()

    return np.array(data_list), np.array(data_names), np.array(data_h)



def uniform_h(output_dir, d, h):
    """
    interpolate data to uniform h steps
    d - data array, h - used input heights (ordered min to max)
    """
    from_h = np.min(h)
    to_h = np.max(h)    
    hi = np.linspace(from_h, to_h, 300)
    sign = +1
    if not np.isclose(from_h,h[0]):
        hi = hi[::-1]
        sign = -1 
    di = np.empty((hi.size,d[0].shape[0],d[0].shape[1] ))

    j, jj = 0, 0
    for i in range(hi.size):
        if np.isclose(hi[i], h[j]): # initial case
            di[i] = d[j]            
        if hi[i] < h[jj] and not np.isclose(hi[i], h[jj]):
            j , jj = jj, jj+1
        mean_h = (h[j]+h[jj])/2.
        if hi[i] > mean_h:
            di[i] = d[j]
        else:
            di[i] = d[jj]
    test = di[:,:,1]
    f = di[0,:,0]/1e9
    # print(test.shape)
    from matplotlib.colors import LogNorm
    plt.imshow(test, aspect='auto',
               # interpolation='quadric',
               interpolation='none',
               # norm=LogNorm(vmin=np.min(test), vmax=np.max(test)
               # ),
               # norm=LogNorm(vmin=np.min(test)*7e3, vmax=np.max(test)/3
               # ),
               cmap=plt.cm.plasma,
               # extent =(np.min(f), np.max(f), from_h, to_h)
    )
    plt.colorbar() # draw colorbar
    plt.xlabel(r"$f$,GHz")
    plt.ylabel(r"${}^R{/}_L$", fontsize=14)
    plt.tight_layout(pad=0.5, w_pad=0.0, h_pad=0.0)
    # for vh in h:
    #     plt.axhline(vh)
    plt.savefig((output_dir+"/colormap_free_"+free_space_file[-8:-4]+txt_file+".pdf"), dpi=150)
    plt.clf(); plt.close()
    plt.show()
    return di,  hi


def filter_data(data_list_sub):
    data_filtered = []
    for i in range(len(data_list_sub)):
        xx = data_list_sub[i][:,0]
        b, a = signal.butter(1, 0.06)
        y = data_list_sub[i][:,1]
        y_butter = signal.filtfilt(b, a, y)
        data = np.array(np.vstack((xx,y_butter))).T
        data_filtered.append(data)
    return np.array(data_filtered)


def get_freq_estimate(data_list_sub, data_h):
    plt.figure('estimate')
    ax = plt.gca()
    x1 = data_list_sub[0][:,0];        y1 = data_list_sub[0][:,1];     h1 = data_h[0]
    x2 = data_list_sub[-1][:,0];       y2 = data_list_sub[-1][:,1];    h2 = data_h[-1]
    idx1, peak_widths, idx_widths = get_estimated_peaks_and_widths(x1, y1, ax)
    idx2, peak_widths, idx_widths = get_estimated_peaks_and_widths(x2, y2, ax)    
    # plt.show()
    plt.clf(); plt.close()
    return idx1, idx2

def get_data_crops(idx1, idx2, RL1, RL2, data_list_sub, data_h):
    plt.figure('shifted')
    data_crop = []
    xspan =16
    for i in range(len(data_list_sub)):
        # RLi = cyl_R/data_h[i]
        # xi = int(idx2 + (idx1-idx2)/(RL1-RL2)*(RLi-RL2))
        xx = data_list_sub[i][:,0]
        y = data_list_sub[i][:,1]
        # xx = data_list_sub[i][xi-xspan:xi+xspan,1]
        # y = data_list_sub[i][xi-xspan:xi+xspan,2]
        data = np.array(np.vstack((xx,y))).T
        data_crop.append(data)
        b, a = signal.butter(1, 0.06)
        y = data_list_sub[i][:,1]
        y_butter = signal.filtfilt(b, a, y)
        step = -0.001
        plt.plot(data[:,0], data[:,1]+step*i, lw=1)
    # plt.show(); sys.exit()
    plt.clf(); plt.close()
    return np.array(data_crop)


def make_fano_fit(data):
    x_data = data[:,0]-np.abs(data[0,0]+data[-1,0])/2
    y_range = np.max(data[:,1]-np.min(data[:,1]))
    y_min = np.min(data[:,1])
    y_data = (data[:,1]-y_min)/y_range
    # x_data = data[:,0]
    # y_data = data[:,1]    
    x = Variable('x')
    y = Variable('y')
    fl = Parameter('fl', 0)
    max_width = np.abs(x_data[0]-x_data[-1])
    G = Parameter('G'  , max_width/8)
    q = Parameter('q',5)
    I = Parameter('I',np.max(y_data)-np.min(y_data))
    bg_level = Parameter('bg_level',0)
#     constraints = [
#     # Eq(x**3 - y, 0),    # Eq: ==
#     # Ge(max_width, G),       # Ge: >=
# ]

    model = { y: I# /(1+q**2)
              *( (2*(x-fl)/G) + q )**2/( (2*(x-fl)/G)**2 + 1 )+bg_level }
    model = Model(model)


    fit = Fit(model, x=x_data, y=y_data, absolute_sigma=False)#, constraints=constraints)
    fit_result = fit.execute()

    q = Parameter('q', fit_result.value(q))
    fl = Parameter('fl', fit_result.value(fl))
    G = Parameter('G', fit_result.value(G))
    I = Parameter('I', fit_result.value(I))
    bg_level = Parameter('bg_level', fit_result.value(bg_level))

    # guess = InteractiveGuess2D(model, x=x_data, y=y_data, n_points=250)
    # guess.execute()

    xfit = np.linspace(x_data[0], x_data[-1],300)
    solutions = model(x=xfit, **fit_result.params)

    fit = Fit(model, x=x_data, y=y_data, absolute_sigma=False)#, constraints=constraints)
    fit_result = fit.execute()

    yfit = solutions.y
    q_stdev = fit_result.stdev(q)
    q = fit_result.value(q)
    G = fit_result.value(G)

    xfit = xfit + np.abs(data[0,0]+data[-1,0])/2
    yfit = yfit * y_range + y_min

    # y_data = (data[:,1]-y_min)/y_range

    # y_data = (data[:,1]-np.min(data[:,1]))/np.max(data[:,1]-np.min(data[:,1]))

    return xfit, yfit, q, q_stdev, G


def gaussian(x, fl, G):
    return np.exp(-(x-fl)**2/(2*G**2))

def fano(x, I,q,G,fl, bg_level,slope, slope2):
    G = np.abs(G)
    return ((I**2 +0.0)**0.5 /(1+q**2)
                 *(q*G/2. + x -fl  )**2
              /( (G/2.)**2 + (x-fl)**2 )
              +bg_level+x*slope# + slope2*x**2
            )

def fano_k(x, I,q,G,fl, bg_level):
    G = np.abs(G)
    return ((I**2+0.1)**0.5 /(1+q**2)
            *(q*G/2. + x -fl  )**2
            /( (G/2.)**2 + (x-fl)**2 )
            +bg_level#+x*slope# + slope2*x**2
            )

# def fano_k(xdata, I,q,G,fl, bg_level):
#     return (bg_level+
#             (I**2)**0.5*(q+2*(xdata-fl)/G)**2/(1+4*(xdata-fl)**2/(G**2))
#     )

def fano_curve_fit(x, y, fano, I,q,G,fl, bg_level, slope, slope2, sigma_data = None):
    params = (I,q,G,fl, bg_level, slope, slope2)
    if fano.__name__=="fano_k":
        params = (I,q,G,fl, bg_level)

    popt, pcov = scipy.optimize.curve_fit(fano, x, y, p0 = params, sigma=sigma_data)
    perr = np.sqrt(np.diag(pcov))
    print('perr = ', perr)
    x_plot = np.linspace(x[0], x[-1], 300)
    fig = plt.figure(num='scalar fit')
    ax = fig.gca()
    y_fit = fano(x_plot, *popt)
    ax.plot(x_plot, y_fit, lw=1.4, alpha=0.7, color = "black", ls='--')
    ax.plot(x, y)
    # plt.show()
    plt.clf(); plt.close()
    print("Estimate:  ",tuple(popt))
    print(perr)
    return popt, perr


def make_fano_fit2(data,q, fl, G, I, bg_level):
    x_data = data[:,0]-np.abs(data[0,0]+data[-1,0])/2
    y_range = np.max(data[:,1]-np.min(data[:,1]))
    y_min = np.min(data[:,1])
    y_data = (data[:,1]-y_min)/y_range
    # x_data = data[:,0]
    # y_data = data[:,1]
    I,q,G,fl, bg_level = fano_curve_estimate(x_data, y_data)

    x = Variable('x')
    y = Variable('y')

    if np.isclose(I, 0.):
        fl = Parameter('fl', 0)
        max_width = np.abs(x_data[0]-x_data[-1])
        G = Parameter('G'  , max_width/8)
        q = Parameter('q',-1)
        I = Parameter('I',1)
        bg_level = Parameter('bg_level',0)
    else:
        fl = Parameter('fl', fl)
        G = Parameter('G'  , G)
        q = Parameter('q',q)
        I = Parameter('I',I)
        bg_level = Parameter('bg_level',bg_level)
        

    constraints = [
#     # Eq(x**3 - y, 0),    # Eq: ==
#     # Ge(max_width, G),       # Ge: >=
#                    Ge(I,0.1)
                   # Eq(I,1),
            # Eq(bg_level,0)
      ]

    model = { y: (I**2+0.2)**0.5 /(1+q**2)
              *(q*G/2. + x -fl  )**2
              /( (G/2.)**2 + (x-fl)**2 )
              +bg_level    }
    # model = { y: (I**2+0.2)**0.5 /(1+q**2)
    #           *( (2*(x-fl)/G) + q )**2/( (2*(x-fl)/G)**2 + 1 )+bg_level }
    model = Model(model)

    fit = Fit(model, x=x_data, y=y_data, constraints=constraints, absolute_sigma=False)
    fit_result = fit.execute()

    q = Parameter('q', fit_result.value(q))
    fl = Parameter('fl', fit_result.value(fl))
    G = Parameter('G', fit_result.value(G))
    I = Parameter('I', fit_result.value(I))
    bg_level = Parameter('bg_level', fit_result.value(bg_level))

    # guess = InteractiveGuess2D(model, x=x_data, y=y_data, n_points=250)
    # guess.execute()

    xfit = np.linspace(x_data[0], x_data[-1],300)
    solutions = model(x=xfit, **fit_result.params)
    
    # sigma_data = 1/np.array([1. if gaussian(x_data[i],fit_result.value(fl),fit_result.value(G))> 0.01 else 0.00001 for i in range(len(x_data))])
    Gsigma = fit_result.value(G)
    sigma_data = 1/gaussian(x_data,fit_result.value(fl),(Gsigma if Gsigma > 1 else 3)*Gfac)
    # sigma_data = 1/(gaussian(x_data,fit_result.value(fl),fit_result.value(G)*Gfac)*0+1.)

    fit = Fit(model, x=x_data[sigma_data<10], y=y_data[sigma_data<10], constraints=constraints,
              absolute_sigma=False, sigma_y = sigma_data[sigma_data<10])
    fit_result = fit.execute()

    yfit = solutions.y
    q_stdev = fit_result.stdev(q)
    bg_level = fit_result.value(bg_level)
    fl = fit_result.value(fl)
    q = fit_result.value(q)
    I = fit_result.value(I)
    G = fit_result.value(G)
    xfit = xfit + np.abs(data[0,0]+data[-1,0])/2
    yfit = yfit * y_range + y_min
    # y_data = (data[:,1]-y_min)/y_range

    # y_data = (data[:,1]-np.min(data[:,1]))/np.max(data[:,1]-np.min(data[:,1]))
    isCF = False
    return x_data, y_data, xfit, yfit, q, q_stdev, fl, G, I, bg_level, model[y], sigma_data, isCF

def fano_curve_estimate(x, y):
    plt.figure('fano')
    # plt.plot(x, y)
    b, a = signal.butter(1, 0.06)
    y = signal.filtfilt(b, a, y)
    plt.plot(x, y)
    idx = [np.argmin(y),np.argmax(y)]
    ymin, ymax = y[idx[0]], y[idx[1]]
    xmin, xmax = x[idx[0]], x[idx[1]]
    bg_level = ymin
    I = ymax-ymin
    fl = (xmin+xmax)/2.
    G = np.abs(xmax-xmin)
    # G = np.sqrt( -4 * (xmin*xmax - fl*(xmin+xmax) - fl**2))
    q = 2* (xmax -fl)/G

    # for _ in range(10):
    #     fl = (xmin/q+xmax*q)/2.
    #     G = np.sqrt( np.abs(-4 * (xmin*xmax - fl*(xmin+xmax) - fl**2)))
    #     # G = np.sqrt( np.abs(xmin*xmax))
    #     # q = 2* (xmax -fl)/G
    #     q = -2* (xmin -fl)/G

    
    print("Estimate:  ",I,q,G,fl, bg_level)
    yfano = fano(x, I,q,G,fl, bg_level, 0., 0.)
    plt.plot(x, y)
    plt.plot(x, yfano)
    plt.scatter(x[idx],y[idx], c='r', zorder=10)
    # plt.show()
    plt.clf(); plt.close()
    return I,q,G,fl, bg_level

# min
# -q Gamma = 2 (f[min] -fl)
# max
#  Gamma = 2 (f[max] -fl) q
#
# -q**2 (f[max] -fl)  = (f[min] -fl)
#  Gamma/(2 (f[max] -fl) ) = q
# Gamma**2 = -4 (f[min] -fl) (f[max] -fl)
# Gamma**2 = -4 (f[min]*f[max] - fl(f[min]+f[max]) - fl**2 )

def make_fano_fit3(data,q, fl, G, I, bg_level, slope, slope2):
    x_data = data[:,0]-np.abs(data[0,0]+data[-1,0])/2
    # x_data = data[:,0]
    y_range = np.max(data[:,1]-np.min(data[:,1]))
    y_min = np.min(data[:,1])
    y_data = (data[:,1]-y_min)/y_range
    xfit = np.linspace(x_data[0], x_data[-1],300)
    xfit = xfit + np.abs(data[0,0]+data[-1,0])/2

    yfit = 0.0
    # yfit = yfit * y_range + y_min
    # I,q,G,fl, bg_level = fano_curve_estimate(x_data, y_data)
    Gsigma = 1
    sigma_data = 1/gaussian(x_data,fl,(Gsigma if Gsigma > 1 else 3)*Gfac)
    try:
    # if True:
        for i in range(3):
            I,q,G,fl, bg_level = fano_curve_estimate(x_data, y_data)
            popt, perr = fano_curve_fit(x_data, y_data, fano, I,q,G,fl, bg_level, slope, slope2)
            I,q,G,fl, bg_level,slope,slope2 = tuple(popt)

            y_data = y_data-slope*x_data#-slope2*x_data**2

        # I,q,G,fl, bg_level = fano_curve_estimate(x_data, y_data)
        # popt, perr = fano_curve_fit(x_data, y_data, fano, I,q,G,fl, bg_level, slope, slope2)
        # I,q,G,fl, bg_level,slope,slope2 = tuple(popt)
        #
        # y_data = y_data-slope*x_data#-slope2*x_data**2

        Gsigma = G
        # sigma_data = 1/gaussian(x_data,fl,(Gsigma if Gsigma > 1 else 3)*Gfac)
        sigma_data = 1/gaussian(x_data,fl,(Gsigma)*Gfac)
        for i in range(5):
            popt, perr = fano_curve_fit(x_data, y_data, fano_k, I,q,G,fl, bg_level, slope, slope2, sigma_data = sigma_data)
            I,q,G,fl, bg_level = tuple(popt)
            if q > 40: q = 40
            if q < -40: q = -40
        popt, perr = fano_curve_fit(x_data, y_data, fano_k, I,q,G,fl, bg_level, slope, slope2, sigma_data = sigma_data)
        I,q,G,fl, bg_level = tuple(popt)
        isCF = True
        print("slope:",slope, "slope2:",slope2, " q:", q)
    except:
        q, fl, G, I, bg_level = np.nan,0.,0.,0.,0.
        isCF = False
    yfit = fano(xfit, I,q,G,fl, bg_level,slope, slope2)

    #params = (I,q,G,fl, bg_level, slope, slope2)
    try:
        q_stdev = perr[1]
        G_stdev = perr[2]
    except:
        q = np.nan
        q_stdev = np.inf
        G_stdev = np.inf
    print("Fit     :  ",I,q,G,fl, bg_level)
    return x_data, y_data, xfit, yfit, q, q_stdev, fl, G, G_stdev, I, bg_level, 'Fano fit', sigma_data, isCF


def spaced(RL):
    cyl_h = cyl_R/RL
    return cyl_R/(cyl_h+space)


def main(spectra_dir):
    global output_dir;
    output_dir = spectra_dir+"_output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    data_list, data_names, data_h = read_data_from_files(spectra_dir, isAllSpecs)
    data_list_sub = data_list
    print(data_list.shape)
    # # uniform_h(output_dir, data_list_sub, cyl_R/data_h)
    # # sys.exit(1)
    i=0
    idx1, idx2 = get_freq_estimate(data_list_sub, data_h)
    RL1, RL2 = data_h[0], data_h[-1]
    data_list = get_data_crops(idx1, idx2, RL1, RL2, data_list_sub, data_h)
    RLs, QQs, QQs2stdev, Q0s, RLns, f0 = [],[],[], [],[], []
    print(len(data_list))
    q, fl, G, I, bg_level, slope, slope2 = np.nan,0.,0.,0.,0., 0., 0.,
    for i in range(len(data_list_sub)):
        print('-----------------------------------------------')
        print(i, 'of', len(data_list_sub))
        print('-----------------------------------------------')

        plt.figure('fit',figsize=(6,1*(len(data_list_sub))))
        data = data_list[i]
        if len(data) == 0: continue
        xdd, ydd, xfit, yfit, q, q_stdev, fl, G, G_stdev, I, bg_level, model, sigma_data, isCF = (
            make_fano_fit3(data,q, fl, G, I, bg_level, slope, slope2) )
            # make_fano_fit3(data,q, fl, G, I, bg_level) )
        x_mid = np.abs(data[0,0]+data[-1,0])/2
        # xfit, yfit, q , q_stdev, G= make_fano_fit(data)

        y_range = np.max(data[:,1]-np.min(data[:,1]))
        y_min = np.min(data[:,1])
        xfit = np.linspace(data[0,0], data[-1,0],300)
        x = xfit-np.abs(data[0,0]+data[-1,0])/2
        yfit = ((I**2+0.1)**0.5/(1+q**2)
              *(q*G/2. + x -fl  )**2
              /( (G/2.)**2 + (x-fl)**2 )
                +bg_level)
        if isCF:
            yfit = fano_k(x, I,q,G,fl, bg_level)#, 0, 0)
            # print(yfit)
        # yfit = yfit * y_range + y_min
        xfit = x
        step = -0.001
        step = -1.7
        # plt.plot(data[:,0], data[:,1]+step*i, lw=0.2, alpha = 0.8)
        plt.plot(xfit, yfit+step*i, lw=1.5, alpha = 0.7)
        # plt.plot(xfit, gaussian(xfit,fl,G*Gfac)+step*i, lw=0.5, alpha = 0.4, color='black', ls='--')
        plt.plot(xdd, 1/sigma_data+step*i, lw=0.5, alpha = 0.4, color='black', ls='--')
        plt.plot(xdd, ydd+step*i, lw=0.5, alpha = 0.5, marker=',', markersize=0.1)
        if np.abs(I)<0.05 or np.abs(I)>2: continue
        # if np.abs(q) > 50: continue
        # try:
        plt.annotate(r"q=%5.3g" "±%g"
                     "        G=%5.3g±%g        fl=%5.4g       f0=%5.4g"%(q ,np.abs(q_stdev)
                                                                                       ,G, G_stdev, fl, x_mid),
                                                    ((np.min(xfit)+np.max(xfit))/2, np.min(ydd)+step*(i+0.2)), fontsize=4)
        plt.annotate(r"n=%6.5g     I=%3.2g     bg=%3.2g  (f0+fl)/G=%5.4g"%(data_h[i], I, bg_level, (x_mid+fl)/G), (np.min(xfit), np.min(ydd)+step*(i+0.2)), fontsize=4)
        # except: pass
        # plt.show()
        print("RL=",cyl_R/data_h[i],"(f0+fl)/G=",(x_mid+fl)/G)
        print(q)
        print('Q_calc', np.abs((x_mid+fl)/G))

        if np.isinf(q_stdev): q = np.nan
        RLs.append(#cyl_R/
                   (data_h[i]#+0.15
        )); QQs.append(q); QQs2stdev.append(q_stdev); Q0s.append(np.abs((x_mid+fl)/G))
        RLns.append(#cyl_R/
                    (data_h[i]));
        f0.append(x_mid+fl)
    multi_savetxt(output_dir,((RLns, "RLns-fano.txt"),(RLs, "RLs-fano.txt"),(Q0s, "Q0s-fano.txt"),
                              (QQs, "q-fano.txt"),(QQs2stdev,"QQs2stdev.txt"), (f0, "f0.txt")))

    plt.figure('fit')
    # plt.title(r"$\frac{I_0}{1+q^2} \frac{(qG/2 + x -f_l)^2 }{ (G/2)^2 + (x-f_l)^2 } +I_{bg}$", fontsize=10)
    # plt.title(model, mode="plain", fontsize=8)
    # plt.ylim(-22,2)
    plt.savefig(output_dir+'/fano_fit.pdf')
    plt.clf(); plt.close()


    np.savetxt('x_data', xdd)
    np.savetxt('y_data', ydd)
    np.savetxt('x_fit', xfit)
    np.savetxt('y_fit', yfit)
    # RLs3,QQs3 = multi_loadtxt(output_dir, ("RLs.txt", "QQs.txt"))
    # RLs_fano_b, Q0s_fano_b, q_fano_b = multi_loadtxt('../bi-scattering-4b_output',("RLs-fano.txt", "Q0s-fano.txt", "q-fano.txt"))
    # RLs_fano_c, Q0s_fano_c, q_fano_c = multi_loadtxt('../bi-scattering-4c_output',("RLs-fano.txt", "Q0s-fano.txt", "q-fano.txt"))
    # RLs_fano_f, Q0s_fano_f, q_fano_f = multi_loadtxt('../bi-scattering-4f_output',("RLs-fano.txt", "Q0s-fano.txt", "q-fano.txt"))
    # # RLns_fano2, Q0s_fano2, q_fano2 = multi_loadtxt('../bi-scatteing-4_output',("RLns-fano.txt", "Q0s-fano.txt", "q-fano.txt"))
    # RLns_fano2, Q0s_fano2, q_fano2 = multi_loadtxt('../TE2_output',("RLns-fano.txt", "Q0s-fano.txt", "q-fano.txt"))
    # RLsG2, QQsG2 = multi_loadtxt('../Glad-tuned_output',("RLsG2.txt", "QQsG2.txt"))
    #
    # qKK = np.loadtxt('qKK.txt')
    # RLs2 = qKK[:,0]
    # QQs2 = qKK[:,1]
    # RLs = np.array(RLs)
    # QQs = np.array(QQs)
    # QQs2stdev = np.array(QQs2stdev)
    # selector = np.all([RLs>0.40, QQs<100, QQs>-100], axis=0)
    # RLs = RLs[selector];  QQs = QQs[selector];
    # QQs2stdev = QQs2stdev[selector]
    #
    # plt.figure('rl q')
    # plt.axvline(0, color='black', lw =0.5, zorder = 1)
    # plt.xlabel('q')
    # plt.ylabel('R/L')
    # plt.scatter(QQs, spaced(RLs), color='magenta',  zorder = 4#, label='far'
    # )
    # # plt.scatter(q_fano2[0], RLns_fano2[0], color='green',  zorder = 3)
    # # plt.scatter(q_fano_c, RLs_fano_c, color='green',  zorder = 4, alpha = 0.8, label='close')
    # # plt.scatter(q_fano_b, RLs_fano_b, color='cyan',  zorder = 3, alpha =0.8, label='mid')
    # plt.legend()
    # print(QQs)
    # # plt.scatter(QQs3, RLs3, color='black', alpha=0.5, zorder = 3, s=2)
    # plt.scatter(QQsG2, RLsG2, color='green', alpha=0.5, zorder = 3, s=2)
    # plt.plot(QQs2, RLs2, color='blue', alpha=0.8, lw=1, zorder = 2)
    # plt.errorbar(QQs, spaced(RLs), xerr=QQs2stdev, yerr=0.005,linestyle='None', lw=0.3, ms =0.5, marker='o', capsize=2, capthick=0.3, zorder=12, color='red')
    #
    # # plt.xlim(np.min(QQs)*1.1, np.max(QQs)*1.1)
    # plt.xlim(-10, 10)
    # # plt.ylim(np.min(RLs)*1.1, np.max(RLs)*1.1)
    # # plt.ylim(0.685, 0.75)
    # plt.ylim(0.68, 0.72)
    # plt.locator_params(axis='x', nbins=5)
    # plt.locator_params(axis='y', nbins=5)
    # plt.tight_layout()
    # plt.savefig(output_dir+'/fano_fit_q_'+free_space_file[25:-4]+'.pdf')
    # plt.clf(); plt.close()

# dirs = ["ED", "EQ", "MD", "MQ"]
# dirs = ["ED"]
# dirs = ["theta0_discrete_samples"]
# dirs = ['mul_results']
dirs = ['cst_300_npts_adaptive']
# dirs = ['cst_speed_sec']
# dirs = ['M103spectra_new']
# dirs = ['suppl_verification']
if isMakeFit:
    for dir in dirs:
        main(dir)

# plt.figure('nq')
# for dir in dirs:
#     output_dir = dir+"_output"
#     x,y = multi_loadtxt(output_dir, ("RLs-fano.txt", "Q0s-fano.txt"))
#     log_x, log_y = np.log(x), np.log(y)
#     curve_fit = np.polyfit(log_x, log_y, 1)
#     q_fitted = curve_fit[0]*log_x + curve_fit[1]
#     q_fitted = np.exp(q_fitted)
#     plt.title(curve_fit[0])
#     plt.plot(x, q_fitted, color='red', lw=1, alpha=0.5)
#     plt.plot(x, y, marker='o', linestyle='None', mfc='None', mec='black', mew=2, ms=8)
#     plt.legend()
# plt.xlabel('n')
# plt.ylabel('q')
# plt.yscale('log')
# plt.savefig('nq.pdf')
# plt.clf(); plt.close()
