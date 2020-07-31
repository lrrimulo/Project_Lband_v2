"""
This program identifies the lines of Cesar's files and also obtain the 
errors. Then, it prints everything in another folder. 
This is a necessary step for the main program of the L-band project.
"""

import numpy as np
import glob as glob
import matplotlib.pyplot as plt
import pyhdust.spectools as spt
import emcee


folder_data = "./DatosBeStarsL/"
output_folder = "./DatosBeStarsL_lrrimulo/"

##############################
### Collecting the correct locations of the hydrogen lines in the L-band.

### Maximum level for the hydrogen atom (100 is more than sufficient)
N=100
### Left and right boundaries of the spectral region where the transitions
### are to be found [microns]
lamb_left = 3.3
lamb_right = 4.1

### 
selectedlines=[]
lambs=[]
ji=[]
### transitions, from j to i (emission)
for i in range(1,N):
    for j in range(i+1,N+1):
        lamb = spt.hydrogenlinewl(j, i)*1e6 ### lambda [microns]
        ### If this transition is in the desired lambda domain, save it:
        if lamb_left <= lamb <= lamb_right:
            selectedlines.append([j,i,lamb])
            lambs.append(lamb)
            ji.append([j,i])

### Now, since it is known that the Humphrey 17 line "merges" with
### Pfund gamma, we will create lists that exclude this line.

transitions_exclude=[[17,6]]

selectedlines_v2 = []
lambs_v2 = []
ji_v2 = []
for i in range(0,len(ji)):
    if not (ji[i] in transitions_exclude): 
        selectedlines_v2.append(selectedlines[i])
        lambs_v2.append(lambs[i])
        ji_v2.append(ji[i])




##############################
### 

### Reading all files and putting their contents in 'file_contents'.
file_list = sorted(glob.glob(folder_data+"*"))
file_contents = []
for ifile in range(0,len(file_list)):
    f0 = open(file_list[ifile],"r")
    lines = f0.readlines()
    f0.close()
    lines_receive = []
    for iline in range(0,len(lines)):
        linesplitted = lines[iline].split()
        if len(linesplitted) > 0:
            lines_receive.append(lines[iline])
    file_contents.append([file_list[ifile],lines_receive])

### Since, 

def means_and_stds(index_cesar,file_contents,numb_repetitions = 3):
    """
    Since Cesar's data repeats 'numb_repetitions' times, in this function the 
    mean and standard deviation of a column of his files are calculated.
    """
    result = []
    ### Loop over files found in Cesar's folder:
    for ifile in range(0,len(file_contents)):
        means=[]
        stds=[]
        ### Loop over the lines found for that specific star
        leng=len(file_contents[ifile][1])/numb_repetitions
        for iline in range(0,leng):
            ### Loop over the repetitions, in order to calculate means
            ### and standard deviations:
            means_sum = 0.
            means_sum2 = 0.
            for j in range(0,numb_repetitions):
                means_sum += float(file_contents[ifile][1][iline+j*leng].\
                                split()[index_cesar])
                means_sum2 += float(file_contents[ifile][1][iline+j*leng].\
                                split()[index_cesar])**2.
            ### Means and standard deviations:
            means.append(means_sum/float(numb_repetitions))
            stds.append((means_sum2/float(numb_repetitions)-\
                (means_sum/float(numb_repetitions))**2.)**0.5)
        ### Attribution of the results to 'result':
        result.append([file_contents[ifile][0],means,stds])
    
    return result

### 0th column: apparent lambdas (measured by Cesar) [Angstroms]
apparent_lambdas = means_and_stds(0,file_contents,3)
### 1st column: ??? Cesar???
continuums = means_and_stds(1,file_contents,3)
### 2nd column: line fluxes [erg/s cm^2]
fluxes = means_and_stds(2,file_contents,3)
### 3rd column: equivalent widths [Angstroms]
EWs = means_and_stds(3,file_contents,3)
### 4th column: ??? Cesar???
cores = means_and_stds(4,file_contents,3)
### 5th column: gaussian FWHM [Angstroms]
gaussianFWHMs = means_and_stds(5,file_contents,3)
### 6th column: lorentzian FWHM [Angstroms]
lorentzianFWHMs = means_and_stds(6,file_contents,3)




######################################
######################################
### (This is by far the most time consuming part of this program.)
### Finding the best correction for the apparent lambdas, allowing 
### a better identification of the lines from Cesar's files.
### 
### The idea is to apply a transformation to the apparent lambdas, 
### in order to make them closer to the theoretical lambdas. 
### This should more or less mimic the visual work of a person moving and
### stretching a transparent paper with a barcode in order to match it 
### with a theoretical barcode.

def polynom_lambda(lamb_app,lamb0,beta,a1,a2,a3,a4,a5,a6):
    """
    Polynomial transformation to be applied to the "apparent lambda". 
    The parameters of this polynomial will be fitted in the MCMC process.
    """
    lamb_real = lamb_app*np.exp(\
        beta+\
        a1*np.log(lamb_app/lamb0)+\
        a2*np.log(lamb_app/lamb0)**2.+\
        a3*np.log(lamb_app/lamb0)**3.+\
        a4*np.log(lamb_app/lamb0)**4.+\
        a5*np.log(lamb_app/lamb0)**5.+\
        a6*np.log(lamb_app/lamb0)**6.\
        )
    return lamb_real


ndim = 8             ### Number of dimensions of the distribution. 
nwalkers = 10*ndim   ### We'll sample with >=2*ndim walkers.
Nchain = 2000        ### Number of elements of the chains.


limits = [  [40000.,41000.],    ### lambda0 [Angstroms]
            [-0.005,0.005],     ### beta
            [-0.05,0.05],       ### a1
            [-0.1,0.1],         ### a2
            [-0.3,0.3],         ### a3
            [-1.0,1.0],         ### a4
            [-3.0,3.0],         ### a5
            [-1e1,1e1]          ### a6
            ]                



def lnprob(theta, apparents, sig2_apparents, lhydro, limits):
    """
    The distribution I invented for the fitting process.
    The idea is to reduce sum of the quadratic distances between
    the corrected lambdas and their nearest theoretical lambda.
    """
    
    ### Allowed domain for the fitted parameters
    if not(limits[0][0] <= theta[0] <= limits[0][1]) or \
                not(limits[1][0] <= theta[1] <= limits[1][1]) or \
                not(limits[2][0] <= theta[2] <= limits[2][1]) or \
                not(limits[3][0] <= theta[3] <= limits[3][1]) or \
                not(limits[4][0] <= theta[4] <= limits[4][1]) or \
                not(limits[5][0] <= theta[5] <= limits[5][1]) or \
                not(limits[6][0] <= theta[6] <= limits[6][1]) or \
                not(limits[7][0] <= theta[7] <= limits[7][1]):
        return -np.inf
    
    else:
        ### It is assumed that the lambdas are in descending order. 
        ### (Cesar's data is organized in this way.)
        chi2 = 0.
        ### Fit only the N first apparent lambdas. (Because it is noticed 
        ### that there is an acumulation of theoretical lines for smaller 
        ### lambdas, which makes the fitting worse, if they are included.)
        N = np.nanmin([5,len(apparents)]) 
        ### The idea is to reduce sum of the quadratic distances between
        ### the corrected lambdas and their nearest theoretical lambda.
        for i in range(0,N):
            lamb_real = polynom_lambda(apparents[i],theta[0],theta[1],\
                            theta[2],theta[3],theta[4],theta[5],theta[6],theta[7])
            ### the quadratic distance between one corrected lambda and its 
            ### nearest theoretical lambda
            dist2 = np.nanmin([(lamb_real - lhydro[k]*1e4)**2. \
                    for k in range(0,len(lhydro))])
            chi2 += dist2/sig2_apparents[i]
        return -0.5*chi2


### Here comes the MCMC part:

### 'real_lambdas' will receive the corrected lambdas for all stars
real_lambdas = []
### Loop over all stars:
for ifile in range(0,len(file_contents)):
    ### Choose an initial set of positions for the walkers.
    ### (It is a list of ndim-dimensional arrays.)
    p0 = [[np.random.uniform(limits[0][0]*1e0,limits[0][1]*1e0),\
        np.random.uniform(limits[1][0]*1e0,limits[1][1]*1e0),\
        np.random.uniform(limits[2][0]*1e0,limits[2][1]*1e0),\
        np.random.uniform(limits[3][0]*1e0,limits[3][1]*1e0),\
        np.random.uniform(limits[4][0]*1e0,limits[4][1]*1e0),\
        np.random.uniform(limits[5][0]*1e0,limits[5][1]*1e0),\
        np.random.uniform(limits[6][0]*1e0,limits[6][1]*1e0),\
        np.random.uniform(limits[7][0]*1e0,limits[7][1]*1e0)] \
        for i in xrange(nwalkers)]

    ### Lists of apparent lambdas and their variances for this specific star
    apparents = apparent_lambdas[ifile][1]
    sig2_apparents = [elem**2. for elem in apparent_lambdas[ifile][2]]

    ### Defining the sampler
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, \
            args=[apparents, sig2_apparents, lambs_v2, limits], a=2, threads=1)
    ### Running MCMC:
    print("SAMPLING FOR "+file_contents[ifile][0]+": file "+str(ifile+1)+\
        " of "+str(len(file_contents)))
    pos, prob, state = sampler.run_mcmc(p0, Nchain)
    print("SAMPLING DONE.")
    print("")
    

    ### Activate this to see the convergence of the emcee fitting 
    ### for this specific star:
    if 1==2:
        for iprob in range(0,len(sampler.lnprobability)):
            plt.plot([np.arcsinh(sampler.lnprobability[iprob][ii]) \
                for ii in range(0,len(sampler.lnprobability[iprob]))], \
                linewidth=0.3, color="black")
        plt.show()

    ### Selecting the most probable parameters of the fitting function
    probmax = np.nanmax(sampler.flatlnprobability)
    idx = [sampler.flatlnprobability[i] for i in \
        range(0,len(sampler.flatlnprobability))].index(probmax)
    ### Most likely vector of model parameters of the fitting function
    theta_hat = [   sampler.flatchain[idx][0],sampler.flatchain[idx][1],\
                    sampler.flatchain[idx][2],sampler.flatchain[idx][3],\
                    sampler.flatchain[idx][4],sampler.flatchain[idx][5],\
                    sampler.flatchain[idx][6],sampler.flatchain[idx][7]
                ]
    print("lambda0 [Angstroms] = "+str(theta_hat[0])) 
    print("beta                = "+str(theta_hat[1]))
    print("a1                  = "+str(theta_hat[2]))
    print("a2                  = "+str(theta_hat[3]))
    print("a3                  = "+str(theta_hat[4]))
    print("a4                  = "+str(theta_hat[5]))
    print("a5                  = "+str(theta_hat[6]))
    print("a6                  = "+str(theta_hat[7]))
    print("")


    ### The corrected lambdas for this specific star: 
    real_lambdas.append([polynom_lambda(apparent_lambdas[ifile][1][ii],\
        theta_hat[0],theta_hat[1],theta_hat[2],theta_hat[3],\
        theta_hat[4],theta_hat[5],theta_hat[6],theta_hat[7]) \
        for ii in range(0,len(apparent_lambdas[ifile][1]))])


    ### Activate this to see a graphical result of the emcee fitting
    ### for this specific star:
    if 1==2:
        for i in range(0,len(lambs)):
            plt.plot([lambs[i]*1e4,lambs[i]*1e4],[-1.,2],color="black",linewidth=0.3)
        for iline in range(0,len(apparent_lambdas[ifile][1])):
            plt.scatter([apparent_lambdas[ifile][1][iline]],[0.],color="black")
            plt.scatter([real_lambdas[-1][iline]],[0.+1.0],color="blue")
        plt.show()








############################################
############################################
### Now, comes the attribution of labels to the lines and the writting
### to the external file.

def attribution_lines(lambdas,selectedlines):
    """
    This function finds the closest theoretical wavelengths to a each 
    element of a list of wavelengths. 
    It returns the list of associated transitions 
    and a measure of the "goodness" of the attribution.
    """
    attributed = []
    goodness = []
    ### Loop over the detected lambdas for this specific star
    for ilamb in range(0,len(lambdas)):
        square_dist = []
        ### Loop over the theoretical lines
        for isel in range(0,len(selectedlines)):
            square_dist.append((lambdas[ilamb]-selectedlines[isel][2]*1e4)**2.)
        square_dist_sorted = sorted(square_dist)
        min_sqdist = square_dist_sorted[0]
        second_min_sqdist = square_dist_sorted[1]
        idx = square_dist.index(min_sqdist)
        attributed.append([
                            selectedlines[idx][0],      ### j of transition
                            selectedlines[idx][1],      ### i of transition
                            selectedlines[idx][2]*1e4   ### the theoretical
                                                        ### lambda [Angstroms]
                        ])
        ### The goodness of the identification is a number greater than zero.
        ### The greater the better. The formula follows below.
        ### (Probably, when goodness > 3, the identification was good.)
        goodness.append((second_min_sqdist-min_sqdist)/min_sqdist)
    return attributed, goodness

### 
identification = []
for ifile in range(0,len(real_lambdas)):
    attributed, goodness = attribution_lines(real_lambdas[ifile],selectedlines_v2)
    identification.append([attributed, goodness])










### Now, preparing the list of parameters to be printed in external files.
lines_print = []
for ifile in range(0,len(file_contents)):
    lines_print.append([file_contents[ifile][0].replace(folder_data,output_folder),[]])
    lines_print[-1][1].append([\
        "# This is a file containing several observables from lines "])
    lines_print[-1][1].append([\
        "# identified by L. Rimulo. "])
    lines_print[-1][1].append([\
        "# Each line contains: "])
    lines_print[-1][1].append(["# "])
    lines_print[-1][1].append([\
        "# 0: j: initial hydrogen level "])
    lines_print[-1][1].append([\
        "# 1: i: final hydrogen level "])
    lines_print[-1][1].append([\
        "# 2: theoretical wavelength of the transition [Angstroms] "])
    lines_print[-1][1].append([\
        "# 3: goodness of the identification "])
    lines_print[-1][1].append(["# "])
    lines_print[-1][1].append([\
        "# 4: apparent wavelength (before identification) [Angstroms] "])
    lines_print[-1][1].append([\
        "# 5: corrected wavelength (used for identification) [Angstroms] "])
    lines_print[-1][1].append([\
        "# 6: error of apparent wavelength [Angstroms] "])
    lines_print[-1][1].append(["# "])
    lines_print[-1][1].append([\
        "# 7: \"continuum\" "])
    lines_print[-1][1].append([\
        "# 8: uncertainty of \"continuum\" "])
    lines_print[-1][1].append(["# "])
    lines_print[-1][1].append([\
        "# 9: line flux [erg/s cm2] "])
    lines_print[-1][1].append([\
        "# 10: uncertainty of line flux [erg/s cm2] "])
    lines_print[-1][1].append(["# "])
    lines_print[-1][1].append([\
        "# 11: equivalent width [Angstroms] "])
    lines_print[-1][1].append([\
        "# 12: uncertainty of equivalent width [Angstroms] "])
    lines_print[-1][1].append(["# "])
    lines_print[-1][1].append([\
        "# 13: \"core\" "])
    lines_print[-1][1].append([\
        "# 14: uncertainty of \"core\" "])
    lines_print[-1][1].append(["# "])
    lines_print[-1][1].append([\
        "# 15: gaussian FWHM [Angstroms] "])
    lines_print[-1][1].append([\
        "# 16: uncertainty of gaussian FWHM [Angstroms] "])
    lines_print[-1][1].append(["# "])
    lines_print[-1][1].append([\
        "# 17: lorentzian FWHM [Angstroms] "])
    lines_print[-1][1].append([\
        "# 18: uncertainty of lorentzian FWHM [Angstroms] "])
    lines_print[-1][1].append(["# "])
    ### 
    for iline in range(0,len(apparent_lambdas[ifile][1])):
        lines_print[-1][1].append([\
            ### j, i, theoretical_lambda, goodness
            identification[ifile][0][iline][0],\
            identification[ifile][0][iline][1],\
            identification[ifile][0][iline][2],\
            identification[ifile][1][iline],\
            ### Center_old, Center_new, err_center
            apparent_lambdas[ifile][1][iline],\
            real_lambdas[ifile][iline],\
            apparent_lambdas[ifile][2][iline],\
            ### continuum, err_continuum
            continuums[ifile][1][iline],\
            continuums[ifile][2][iline],\
            ### flux, err_flux
            fluxes[ifile][1][iline],\
            fluxes[ifile][2][iline],\
            ### EW, err_EW
            EWs[ifile][1][iline],\
            EWs[ifile][2][iline],\
            ### core, err_core
            cores[ifile][1][iline],\
            cores[ifile][2][iline],\
            ### gaussianFWHM, err
            gaussianFWHMs[ifile][1][iline],\
            gaussianFWHMs[ifile][2][iline],\
            ### lorentzianFWHM, err
            lorentzianFWHMs[ifile][1][iline],\
            lorentzianFWHMs[ifile][2][iline] \
            ])


for ifile in range(0,len(lines_print)):
    f1 = open(lines_print[ifile][0],"w")
    for iline in range(0,len(lines_print[ifile][1])):
        for elem in lines_print[ifile][1][iline]:
            f1.write(str(elem)+" ")
        f1.write("\n")
    f1.close()


    





###
if 1==1:
    for i in range(0,len(lambs)):
        plt.plot([lambs[i]*1e4,lambs[i]*1e4],[-1.,len(file_list)],\
                color="black",linewidth=0.3)

    for ifile in range(0,len(apparent_lambdas)):
        for iline in range(0,len(apparent_lambdas[ifile][1])):
            plt.scatter([apparent_lambdas[ifile][1][iline]],[ifile],color="black")
            plt.scatter([real_lambdas[ifile][iline]],[ifile+0.5],color="blue")
    plt.show()
















