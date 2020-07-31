#!/usr/bin/python

from astropy.io import ascii
from astropy.table import Table
import numpy as np
import pyfits
import glob as glob



folder_files = "HD217891/original/"

file_list = glob.glob(folder_files+"*")


for ifile in range(0,len(file_list)):
    filename = file_list[ifile]
    print filename

    hdulist = pyfits.open(filename)
    header = hdulist[0].header #read header
    flux = hdulist[0].data #flux data in the primary
    flux = np.array(flux, dtype=np.float64)

    start_wave = header['CRVAL1'] #initial wavelenght
    step = header['CDELT1'] #increment per pixel

    w0, dw, n = start_wave, step, len(flux)
    w = start_wave + step * n
    wave = np.linspace(w0, w, n, endpoint=False)

    print filename
    print 'step ', step
    print 'starting wavelength ', start_wave
    print 'last wavelength ', wave[-1]

    data = Table([wave, flux], names=(str(header['CRVAL1']), str(step)))
    ascii.write(data, filename[:-5]+".dat", delimiter='\t')  #new file created
    print '----------------------------------'
    print 'fits to ascii: DONE'
