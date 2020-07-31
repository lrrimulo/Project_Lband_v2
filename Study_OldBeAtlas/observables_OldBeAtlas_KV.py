"""

"""

import glob as glob
import numpy as np
import pyhdust as hdt
import pyhdust.phc as phc
import pyhdust.lrr as lrr
import pyhdust.lrr.roche_singlestar as rss
import pyhdust.spectools as spt
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt






###########################
import read_everything

files_fullsed_new, files_source_new, files_temps_new, fullsed_contents, \
        fullsed_path, source_path, temps_path, dist_std = \
        read_everything.read_everything()

outputfile = "observables_BeAtlas_KV.txt"











#############################
### Obtaining magnitudes


filters=[   'bess-u','bess-b','bess-v','bess-r','bess-i',\
            'bess-j','bess-h','bess-k',\
            'filt_Ha',\
            'wise1','wise2','wise3','wise4',\
            'irac_ch1','irac_ch2','irac_ch3','irac_ch4'
        ]

npts_interp=50 ### this number must be >= 3. A good choice is 50.

    
print("Obtaining zero point constants...")
zp=[]
for j in xrange(0,len(filters)):
    zp.append(lrr.obtain_pogson_zp('spct1',filters[j],\
                npts=npts_interp))
print("")


all_photflux=[]
all_Mag=[]
for ifile in xrange(0,len(files_fullsed_new)):
        
    print("Obtaining magnitudes for file")
    print(files_fullsed_new[ifile][0])
        
    fullsedtest=files_fullsed_new[ifile][1]
    sourcetest=files_source_new[ifile]


    photflux_vec=[]
    for j in xrange(0,len(filters)):
        print("Obtaining photon fluxes for filter "+str(filters[j]))
        mu,lamb,flambda,photflux=lrr.fullsed2photonflux(fullsedtest,\
            sourcetest,filters[j],npts=npts_interp,dist=10.)
        photflux_vec.append(photflux)
    all_photflux.append(photflux_vec)

    Mag_vec=[]
    for j in xrange(0,len(filters)):
        Mag_vec.append(lrr.pogson(photflux_vec[j],zp[j]))
    all_Mag.append(Mag_vec)

    print("")






#############################
### Obtaining vsini
            
all_vsini=[]
for ifile in xrange(0,len(files_fullsed_new)):
    contents=fullsed_contents[ifile][1]
    stelpars=[elem for elem in contents[4]]
    rpolenow=stelpars[1]
    massnow=stelpars[0]
    Wnow=stelpars[2]
    lixo,omeganow,lixo,Wnow=rss.rocheparams(Wnow,"W")
    veqfile=rss.cte_veq(rpolenow,massnow,omeganow,1.0)
        
    auxiifile=[]
    for iobs in xrange(0,len(mu)):
        auxiifile.append(veqfile*(1.-mu[iobs]**2.)**0.5)
    all_vsini.append(np.array(auxiifile))


#############################
### Writing in the external file
print("Writing in the external file")


f0=open(outputfile,"w")

for ifile in xrange(0,len(files_fullsed_new)):
    f0.write("MODEL "+\
            str(files_fullsed_new[ifile][0][0])+" "+\
            str(files_fullsed_new[ifile][0][1])+" "+\
            str(files_fullsed_new[ifile][0][2])+" "+\
            str(files_fullsed_new[ifile][0][3])+"\n")
            
    contents=fullsed_contents[ifile][1]
    
    f0.write("    SOURCE "+\
            str(contents[4][0])+" "+\
            str(contents[4][1])+" "+\
            str(contents[4][2])+" "+\
            str(contents[4][3])+" "+\
            str(contents[4][4])+"\n")
            
    f0.write("    TEMP_R ")
    for ii in range(0,len(contents[6][0,:])):
        f0.write(str(contents[6][0,ii])+" ")
    f0.write("\n")
    f0.write("    TEMP_T ")
    for ii in range(0,len(contents[6][1,:])):
        f0.write(str(contents[6][1,ii])+" ")
    f0.write("\n")
    
    for incs in range(0,len(contents[1])):
        f0.write("    COSI "+\
            str(contents[1][incs])+"\n")
        
        f0.write("        UBVRI "+\
            str(all_Mag[ifile][0][incs])+" "+\
            str(all_Mag[ifile][1][incs])+" "+\
            str(all_Mag[ifile][2][incs])+" "+\
            str(all_Mag[ifile][3][incs])+" "+\
            str(all_Mag[ifile][4][incs])+"\n")
        f0.write("        JHK "+\
            str(all_Mag[ifile][5][incs])+" "+\
            str(all_Mag[ifile][6][incs])+" "+\
            str(all_Mag[ifile][7][incs])+"\n")
        f0.write("        HALPHA_SOAR "+\
            str(all_Mag[ifile][8][incs])+"\n")
        f0.write("        WISE_filters "+\
            str(all_Mag[ifile][9][incs])+" "+\
            str(all_Mag[ifile][10][incs])+" "+\
            str(all_Mag[ifile][11][incs])+" "+\
            str(all_Mag[ifile][12][incs])+"\n")
        f0.write("        IRAC_filters "+\
            str(all_Mag[ifile][13][incs])+" "+\
            str(all_Mag[ifile][14][incs])+" "+\
            str(all_Mag[ifile][15][incs])+" "+\
            str(all_Mag[ifile][16][incs])+"\n")
                
        f0.write("    END_COSI \n")
    f0.write("END_MODEL \n")

f0.close()





        
        
        
        
