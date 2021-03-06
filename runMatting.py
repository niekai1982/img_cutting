#!/usr/bin/env python
'''
===============================================================================
Interactive Image Cutting

src_path : src file path
save_path: result file save path
speed_flag: speed of process, default value is 2
save_flag : saveflag
show_flag : result showflag
===============================================================================
'''

import os
import sys
import scipy
import argparse
import numpy as np
from scipy import sparse
from skimage import morphology, transform
import matplotlib.pyplot as plt
from skimage import filters

DICT = ['jpg','bmp','JPG','BMP']

def parseArgument():
    if (len(sys.argv) < 2):
        raise Exception,u"arguments needed"

    #init
    argus = {}

    argus["srcPath"] = r''
    argus["savePath"] = ''
    argus["speed"] = 1

    #set
    argus["srcPath"] = sys.argv[1]
    argus["savePath"] = sys.argv[2]

    return argus


def getLaplacian1(I, consts, epsilon=0.0000001, win_size=1):
    neb_size = (win_size * 2 + 1) ** 2
    h, w, c = I.shape
    img_size = w * h
    consts = morphology.erosion(consts, np.ones((win_size * 2 + 1, win_size * 2 + 1)))
    indsM = np.arange(img_size).reshape((h, w), order='F')
    tlen = np.sum(np.sum(1 - consts[win_size:-win_size, win_size:-win_size])) * (neb_size ** 2)
    row_inds = np.zeros((tlen, 1))
    col_inds = np.zeros((tlen, 1))
    vals = np.zeros((tlen, 1))
    len = 0
    for j in range(win_size, w - win_size):
        for i in range(win_size, h - win_size):
            if consts[i, j]:
                continue
            win_inds = indsM[i - win_size:i + win_size + 1, j - win_size:j + win_size + 1]
            win_inds = win_inds.flatten(order='F').reshape(-1, 1)
            winI = I[i - win_size:i + win_size + 1, j - win_size:j + win_size + 1, :]
            winI = winI.reshape((neb_size, c), order='F')
            win_mu = np.mean(winI, axis=0).T
            win_mu.shape = -1, 1
            win_var = np.linalg.inv(
                np.dot(winI.T, winI) / neb_size - np.dot(win_mu, win_mu.T) + epsilon / neb_size * np.eye(c))
            winI = winI - np.repeat(win_mu.T, neb_size, 0)
            tvals = (1 + np.dot(np.dot(winI, win_var), winI.T)) / neb_size
            row_inds[len:neb_size ** 2 + len] = np.repeat(win_inds.T, neb_size, 0).reshape(neb_size ** 2, 1)
            col_inds[len:neb_size ** 2 + len] = np.repeat(win_inds, neb_size, 1).reshape(neb_size ** 2, 1)
            vals[len:neb_size ** 2 + len] = tvals.flatten().reshape(-1, 1)
            len = len + neb_size ** 2
    vals = vals[0:len]
    row_inds = row_inds[0:len]
    col_inds = col_inds[0:len]
    A = sparse.csc_matrix((vals.flatten(), (row_inds.flatten(), col_inds.flatten())), shape=(img_size, img_size))
    sumA = A.sum(axis=1)
    A = sparse.spdiags(sumA.flatten(), 0, img_size, img_size) - A
    return A


def solveAlpha(I, consts_map, consts_vals, thr_alpha=0.02):
    h, w, c = I.shape
    img_size = w * h
    A = getLaplacian1(I, consts_map)
    D = sparse.spdiags(consts_map.flatten(order='F'), 0, img_size, img_size)
    lbda = 100
    temp = (lbda * consts_map * consts_vals).flatten('F')
    x = scipy.sparse.linalg.spsolve(A + lbda * D, temp)
    alpha = np.maximum(np.minimum(x, 1), 0)
    alpha.shape = w, h
    alpha = alpha.T
    return alpha


def get_mI(I):
    h, w = I.shape[:2]
    mI = I.copy()
    mI[:2 * h / 3, :w / 20] = 0
    mI[:2 * h / 3, -w / 20:] = 0
    mI[-h / 10:, (w / 2 - 15):(w / 2 + 15)] = 1
    mI[h / 10:, (w / 2 - 5):(w / 2 + 5)] = 1
    mI[-h / 30:, :] = 1
    return mI
    

def get_mI_test(I):
    h, w = I.shape[:2]
    mI = I.copy()
    threshold = np.sum(I[:30,:30]) / (30 * 30)
    binI = np.sum(I, axis=2) < threshold - 0.1
    for i in range(5):
        binI = morphology.dilation(binI, np.ones((3,3)))
    for i in range(10):
        binI = morphology.erosion(binI, np.ones((3,3)))

    binI_r = 1 - binI
    for i in range(10):
        binI_r = morphology.erosion(binI_r, np.ones((3,3)))
    binI_r = binI_r > 0
    for i in range(5):
        binI = morphology.erosion(binI, np.ones((3,3)))


    mI[binI_r] = 0
    print binI.shape
    # mI[-h / 10:, (w / 2 - 15):(w / 2 + 15)] = 1
    # mI[h / 10:, (w / 2 - 5):(w / 2 + 5)] = 1
    # mI[-h / 30:, :] = 1
    mI[binI] = 1
    plt.imshow(mI)
    plt.show()
    return mI


def get_mI_test1(I):
    h, w = I.shape[:2]
    mI = I.copy()

    gradI_r = filters.sobel(mI[:,:,0])
    gradI_g = filters.sobel(mI[:,:,1])
    gradI_b = filters.sobel(mI[:,:,2])

    gradI = gradI_r * gradI_r + gradI_b * gradI_b + gradI_g * gradI_g

    mask   = gradI > 0.0001
    mask_g = gradI < 0.0001

    for i in range(2):
        mask_g = morphology.erosion(mask_g,np.ones((3,3)))
    for i in range(2):
        mask = morphology.erosion(mask,np.ones((3,3)))

    mask = mask * (gradI < gradI.mean()/4) * (np.sum(I, axis=2) > 2.6)

    mask[:h * 4 / 5,:] = False
    mask_g[h * 2 / 3:,:] = False

    mI[mask] = 1
    mI[mask_g] = 0

    mI[:2 * h / 3, :w / 20] = 0
    mI[:2 * h / 3, -w / 20:] = 0
    mI[h / 7:, (w / 2 - w / 8):(w / 2 + w / 8)] = 1
    mI[-h / 30:, :] = 1

    return mI


def get_mI_test_temp(I):
    h, w = I.shape[:2]
    mI = I.copy()

    gradI_r = filters.sobel(mI[:,:,0])
    gradI_g = filters.sobel(mI[:,:,1])
    gradI_b = filters.sobel(mI[:,:,2])

    gradI = gradI_r * gradI_r + gradI_b * gradI_b + gradI_g * gradI_g

    mask   = gradI > 0.0001
    mask_g = gradI < 0.0001

    for i in range(2):
        mask_g = morphology.erosion(mask_g,np.ones((3,3)))
    for i in range(2):
        mask = morphology.erosion(mask,np.ones((3,3)))

    mask = mask * (gradI < gradI.mean()/4) * (np.sum(I, axis=2) > 2.6)

    mask[:h * 4 / 5,:] = False
    mask_g[h * 2 / 3:,:] = False

    mI[mask] = 1
    mI[mask_g] = 0

    mI[:2 * h / 3, :w / 20] = 0
    mI[:2 * h / 3, -w / 20:] = 0
    mI[h / 7:, (w / 2 - w / 8):(w / 2 + w / 8)] = 1
    mI[-h / 30:, :] = 1

    return mI


def change_backgroud(src_path, save_path, speed_flag = 2, save_flag=1, show_flag=0):
    '''
    src_path : src file path
    save_path: result file save path
    speed_flag: speed of process, default value is 2
    save_flag : saveflag
    show_flag : result showflag
    '''
    # print 'process...'
    if not os.path.exists(src_path):
        print "src path not exits"
        return
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    files_res = os.listdir(save_path)
    files_src = os.listdir(src_path)

    for file_src in files_src:

        if file_src.split('.')[-1] not in DICT or file_src in files_res:
            continue
        print file_src
        I_src = plt.imread(os.path.join(src_path, file_src))
        I_src = I_src.astype(np.double)

        I = I_src[::speed_flag, ::speed_flag]
        I = I.astype(np.double) / 255

        mI = get_mI_test1(I)

        consts_map = np.sum(np.abs(I - mI), axis=2) > 0.001

        if len(I.shape) == 3:
            consts_vals = mI[:, :, 0] * consts_map
        if len(I.shape) == 2:
            consts_vals = mI * consts_map

        alpha = solveAlpha(I, consts_map, consts_vals)

        alpha = transform.resize(alpha, (I_src.shape[0], I_src.shape[1]))

        mask = np.zeros_like(I_src)

        mask[:, :, 0] = 1
        mask[:, :, 1] = 0
        mask[:, :, 2] = 0

        ff = I_src / 255 * np.dstack((alpha, alpha, alpha))

        bb = mask * np.dstack((1 - alpha, 1 - alpha, 1 - alpha))

        if save_flag:
            file_save_name = os.path.join(save_path, file_src.split('.')[0] + '.' + file_src.split('.')[-1])
            plt.imsave(file_save_name, ff + bb)

        if show_flag:
            plt.subplot(1, 3, 1)
            plt.imshow(mI)
            plt.subplot(1, 3, 2)
            plt.imshow(I_src / 255)
            plt.subplot(1, 3, 3)
            plt.imshow(ff + bb)
            plt.show()

    return "process succeed"

if __name__ == '__main__':
    # print __doc__
    parser = argparse.ArgumentParser()
    parser.add_argument("-s",  help='src path of file',required=True)
    parser.add_argument("-r",  help='res path of file',required=True)
    parser.add_argument("-vf", help='result visual flag',type=int,default=1)
    parser.add_argument("-sf", help='result save  flag',type=int, default=1)
    parser.add_argument("-sp", help='speed control param',type=int, default=4)
    arg = parser.parse_args()
    # parser = argparse.ArgumentParser()
    res = change_backgroud(src_path=arg.s,save_path=arg.r,show_flag=arg.vf,save_flag=arg.sf,speed_flag=arg.sp)
    print res
    # pass

