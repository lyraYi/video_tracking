import cv2
import numpy as np
import struct
import time
import matplotlib.pyplot as plt
import os

##############################################################################
#                          CINE File Reading Functions                       #
##############################################################################

def read_L(f):
    return int(struct.unpack('<l', f.read(4))[0])

def read_Q(f):
    return struct.unpack('Q', f.read(8))[0]

def read_Q_array(f, n):
    a = np.zeros(n, dtype='Q')
    for i in range(n):
        a[i] = read_Q(f)
    return a

def read_B_2Darray(f, ypix, xpix):
    n = xpix * ypix
    a = np.array(struct.unpack(f'{n}B', f.read(n * 1)), dtype='B')
    return a.reshape(ypix, xpix)

def read_H_2Darray(f, ypix, xpix):
    n = xpix * ypix
    a = np.array(struct.unpack(f'{n}H', f.read(n * 2)), dtype='H')
    return a.reshape(ypix, xpix)

def read_cine(ifn):
    with open(ifn, 'rb') as cf:
        t_read = time.time()
        print("Reading .cine file...")

        cf.read(16)
        baseline_image = read_L(cf)
        image_count = read_L(cf)

        pointers = np.zeros(3, dtype='L')
        pointers[0] = read_L(cf)
        pointers[1] = read_L(cf)
        pointers[2] = read_L(cf)

        cf.seek(58)
        nbit = read_L(cf)

        cf.seek(int(pointers[0]) + 4)
        xpix = read_L(cf)
        ypix = read_L(cf)

        cf.seek(int(pointers[1]) + 768)
        pps = read_L(cf)
        exposure = read_L(cf)

        cf.seek(int(pointers[2]))
        pimage = read_Q_array(cf, image_count)

        dtype = 'B' if nbit == 8 else 'H'
        frame_arr = np.zeros((image_count, ypix, xpix), dtype=dtype)

        for i in range(image_count):
            p = struct.unpack('<l', struct.pack('<L', pimage[i] & 0xffffffffffffffff))[0]
            cf.seek(p)
            ofs = read_L(cf)
            cf.seek(p + ofs)
            frame_arr[i] = read_B_2Darray(cf, ypix, xpix) if nbit == 8 else read_H_2Darray(cf, ypix, xpix)

        time_arr = np.linspace(
            baseline_image / pps, 
            (baseline_image + image_count) / pps, 
            image_count, 
            endpoint=False
        )

        print("Done reading .cine file (%.1f s)" % (time.time() - t_read))
        return time_arr, frame_arr
