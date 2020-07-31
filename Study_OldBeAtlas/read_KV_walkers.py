import numpy as np
import matplotlib.pyplot as plt


walkersfile = "KV_walkers.txt"








f0 = open(walkersfile,"r")
lines = f0.readlines()
f0.close()

walkers = np.zeros((len(lines),len(lines[0].split()))); walkers[:,:] = np.nan

for iline in range(0,len(lines)):
    for icol in range (0,len(lines[0].split())):
        walkers[iline,icol] = float(lines[iline].split()[icol])
    








   
theta_lims =    [   [np.nan,np.nan],
                    [np.nan,np.nan],
                    [np.nan,np.nan],
                    [np.nan,np.nan],
                    [np.nan,np.nan]
                ]
X_lims =    [   [np.nan,np.nan],
                [np.nan,np.nan],
                [np.nan,np.nan],
                [np.nan,np.nan],
                [np.nan,np.nan],
                [np.nan,np.nan],
                [np.nan,np.nan]
            ]





if 1==1:

    xmax = np.nanmax(walkers[:,6]-walkers[:,7])
    xmin = np.nanmin(walkers[:,6]-walkers[:,7])
    ymax = np.nanmax(walkers[:,7])
    ymin = np.nanmin(walkers[:,7])
    plt.scatter(walkers[:,6]-walkers[:,7],walkers[:,7],alpha = 0.03)
    plt.xlabel("B-V")
    plt.ylabel("V")
    plt.xlim([xmin-0.1*(xmax-xmin),\
                xmax+0.1*(xmax-xmin)])
    plt.ylim([ymax+0.1*(ymax-ymin),\
                ymin-0.1*(ymax-ymin)])
    plt.show()

    
    xmax = np.nanmax(walkers[:,7]-walkers[:,9])
    xmin = np.nanmin(walkers[:,7]-walkers[:,9])
    ymax = np.nanmax(walkers[:,7])
    ymin = np.nanmin(walkers[:,7])
    plt.scatter(walkers[:,7]-walkers[:,9],walkers[:,7],alpha = 0.03)
    plt.xlabel("V-I")
    plt.ylabel("V")
    plt.xlim([xmin-0.1*(xmax-xmin),\
                xmax+0.1*(xmax-xmin)])
    plt.ylim([ymax+0.1*(ymax-ymin),\
                ymin-0.1*(ymax-ymin)])
    plt.show()
    
    
    xmax = np.nanmax(walkers[:,11]-walkers[:,12])
    xmin = np.nanmin(walkers[:,11]-walkers[:,12])
    ymax = np.nanmax(walkers[:,10]-walkers[:,11])
    ymin = np.nanmin(walkers[:,10]-walkers[:,11])
    plt.scatter(walkers[:,11]-walkers[:,12],walkers[:,10]-walkers[:,11],alpha = 0.03)
    plt.xlabel("H-K")
    plt.ylabel("J-H")
    plt.xlim([xmin-0.1*(xmax-xmin),\
                xmax+0.1*(xmax-xmin)])
    plt.ylim([ymin-0.1*(ymax-ymin),\
                ymax+0.1*(ymax-ymin)])
    plt.show()


    xmax0 = np.nanmax(walkers[:,10]-walkers[:,13])
    xmin0 = np.nanmin(walkers[:,10]-walkers[:,13])
    xmax1 = np.nanmax(walkers[:,10]-walkers[:,14])
    xmin1 = np.nanmin(walkers[:,10]-walkers[:,14])
    xmax2 = np.nanmax(walkers[:,10]-walkers[:,15])
    xmin2 = np.nanmin(walkers[:,10]-walkers[:,15])
    xmax3 = np.nanmax(walkers[:,10]-walkers[:,16])
    xmin3 = np.nanmin(walkers[:,10]-walkers[:,16])
    xmax = np.nanmax(np.array([xmax0,xmax1,xmax2,xmax3]))
    xmin = np.nanmin(np.array([xmin0,xmin1,xmin2,xmin3]))
    ymax = np.nanmax(walkers[:,10])
    ymin = np.nanmin(walkers[:,10])
    plt.scatter(walkers[:,10]-walkers[:,13],walkers[:,10],alpha = 0.03)
    plt.xlabel("J-[1]")
    plt.ylabel("J")
    plt.xlim([xmin-0.1*(xmax-xmin),\
                xmax+0.1*(xmax-xmin)])
    plt.ylim([ymax+0.1*(ymax-ymin),\
                ymin-0.1*(ymax-ymin)])
    plt.show()
    plt.scatter(walkers[:,10]-walkers[:,14],walkers[:,10],alpha = 0.03)
    plt.xlabel("J-[2]")
    plt.ylabel("J")
    plt.xlim([xmin-0.1*(xmax-xmin),\
                xmax+0.1*(xmax-xmin)])
    plt.ylim([ymax+0.1*(ymax-ymin),\
                ymin-0.1*(ymax-ymin)])
    plt.show()
    plt.scatter(walkers[:,10]-walkers[:,15],walkers[:,10],alpha = 0.03)
    plt.xlabel("J-[3]")
    plt.ylabel("J")
    plt.xlim([xmin-0.1*(xmax-xmin),\
                xmax+0.1*(xmax-xmin)])
    plt.ylim([ymax+0.1*(ymax-ymin),\
                ymin-0.1*(ymax-ymin)])
    plt.show()
    plt.scatter(walkers[:,10]-walkers[:,16],walkers[:,10],alpha = 0.03)
    plt.xlabel("J-[4]")
    plt.ylabel("J")
    plt.xlim([xmin-0.1*(xmax-xmin),\
                xmax+0.1*(xmax-xmin)])
    plt.ylim([ymax+0.1*(ymax-ymin),\
                ymin-0.1*(ymax-ymin)])
    plt.show()





