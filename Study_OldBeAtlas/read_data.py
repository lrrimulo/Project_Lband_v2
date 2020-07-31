"""
This file contains several functions/procedures to read the data on our
Be stars.
"""
 
from __future__ import division
import numpy as np
import glob as glob
import pyhdust.lrr as lrr
import pyhdust.spectools as spt






### 
def newlog10abs(x,B):
    """
    This function is defined for all x real.
    It is a good approximation to log10(|x|), if |B*x| >> 1.
    
    Obs: This function explores the fact that 
    arcsinh(B*x) ~ sign(B*x)*A*ln[2*|B*x|], when |B*x| >> 1.
    """
    ### A*np.arcsinh(B*x)-np.sign(B*x)*A*np.log(2.*abs(B*x)) = 0
    ### A*np.arcsinh(B*x)-np.sign(B*x)*A*(np.log(2.*abs(B))+np.log(abs(x))) = 0
    ### np.sign(B*x)*np.arcsinh(B*x)-np.log(2.*abs(B))=np.log(abs(x))
    return np.arcsinh(abs(B*x))/np.log(10.)-np.log10(2.*abs(B))


### 
def err_frac(A,B,errA,errB):
    """
    Suppose you have a function y = A/B.
    Then, its variance is given by 
    sigy^2 = y^2(errA^2/A^2+errB^2/B^2)
    
    This function returns the standard deviation of the function y = A/B.
    """
    
    return abs(A/B)*np.sqrt(errA*errA/A/A+errB*errB/B/B)
    


### 
def alphaL(B__L,lamb1B,lamb2B,R__L,lamb1R,lamb2R):
    """
    Definition of the parameter "alphaL", as a function 
    of the red and blue fluxes, 'R__L' and 'B__L', and their domains in 
    wavelength.
    """
    return 1.-np.log(B__L/(lamb2B-lamb1B)*(lamb2R-lamb1R)/R__L)/\
                np.log((lamb2R+lamb1R)/(lamb2B+lamb1B))


def err_alphaL(B__L,lamb1B,lamb2B,R__L,lamb1R,lamb2R,errB__L,errR__L):
    """
    This function returns the standard deviation of the function "alphaL".
    """
    return 1./np.log((lamb2R+lamb1R)/(lamb2B+lamb1B))*\
                np.sqrt(errB__L**2./B__L**2.+errR__L**2./R__L**2.)


### 
def Vegaflux(lamb1,lamb2,Nnpts = 50):
    """
    Returns the flux of Vega [erg s^-1 cm^-2] 
    between 'lamb1' [microns] and 'lamb2' [microns].
    """
    
    ### Returns the SED of VEGA (9 nm - 160 microns):
    ### * array of lambdas [Angstroms]
    ### * array of Flambda [erg cm^-2 s^-1 A^-1]
    xlp,ylp = lrr.VEGA_spct("spct1")
    
    ### vectors of lambda and dlambda in [Angstroms]:
    llamb = np.array([lamb1+(lamb2-lamb1)/\
        float(Nnpts-1)*float(i) for i in range(0,Nnpts)])*1e4
    dllamb = np.array([llamb[i+1]-llamb[i] for i in range(0,Nnpts-1)])
    

    ylpf = np.array([lrr.interpLinND([llamb[i]],[xlp],ylp) \
        for i in range(0,Nnpts)])
    
    ### Flux of Vega between 'lamb1' and 'lamb2'
    return lrr.integrate_trapezia(ylpf,dllamb)  ### [erg s^-1 cm^-2]


### 
def ap_mag_Menn(flux,FVega):
    return -2.5*np.log10(flux/FVega)

def err_ap_mag_Menn(flux,errflux):
    """
    Uncertainty associated with the function given by "ap_mag_Menn"
    """
    return abs(2.5/np.log(10.)/flux)*errflux

                        




##########################################################
### Reading data on Be stars: 

def List_Stars(to_return):
    """
    * 'to_return' = "stars", If to return basic data on the Be stars
    * 'to_return' = "Cesar", If to return ...
    * 'to_return' = "Cesar_BR", If to return ...
    * 'to_return' = "dist", If to return ...
    * 'to_return' = "WISE", If to return ...
    """
    
    ### If to return basic data on the Be stars:
    if to_return == "stars":
        ### Directory of the location of the data on the Be stars
        data_folder = "./../MedidasLogs/"
        ### List of HD names in ascending order
        HDnames =   [
                    "144",   
                    "4180",
                    "5394",  
                    "6811",  
                    "11606", 
                    "20336", 
                    "23302", 
                    "23480", 
                    "23630", 
                    "23862", 
                    "187811",
                    "191610",
                    "193009",
                    "194335",
                    "194883",
                    "195907",
                    "197419",
                    "200310",
                    "202904",
                    "204722",
                    "208057",
                    "210129",
                    "217675",
                    "217891"
                    ]
        ### List of correspondent popular names of the stars
        Starnames =     [
                        "10 Cas",
                        "$o$ Cas",
                        "$\\gamma$ Cas",
                        "$\\phi$ And",
                        "V777 Cas",
                        "Bk Cam",
                        "17 Tau",
                        "23 Tau",
                        "25 Tau",
                        "28 Tau",
                        "12 Vul",
                        "28 Cyg",
                        "V2113 Cyg",
                        "V2119 Cyg",
                        "V2120 Cyg",
                        "V2123 Cyg",
                        "V568 Cyg",
                        "60 Cyg",
                        "$\\upsilon$ Cyg",
                        "V2162 Cyg",
                        "16 Peg",
                        "25 Peg",
                        "$o$ And",
                        "$\\beta$ Psc"
                        ]
        ### List of spectral types found in the SIMBAD web base
        SIMBAD_Spt =    [
                        "B9IIIe",
                        "B5IIIe",
                        "B0.5IVpe",
                        "B5IIIe",
                        "B2Vne",
                        "B2.5Vn(e)",
                        "B6IIIe",
                        "B6IV(e)",
                        "B7III",
                        "B8Vne",
                        "B2.5Ve",
                        "B2.5Ve",
                        "B1V:nnpe",
                        "B2IIIe",
                        "B2Ve",
                        "B1.5V",
                        "B2IV-Ve",
                        "B1Ve",
                        "B2Vne",
                        "B1.5IV:np",
                        "B3Ve",
                        "B7Vne",
                        "B6IV/V\_sh",
                        "B6Ve"
                        ]
        ### Date of measurement of L-band of the stars
        YYYYMMDD_LBAND =    [
                            "2017 10 01",
                            "2017 10 01",
                            "2017 10 01",
                            "2017 10 01",
                            "2017 10 01",
                            "2017 10 01",
                            "2017 10 01",
                            "2017 10 01",
                            "2017 10 01",
                            "2017 10 01",
                            "2017 10 01",
                            "2017 10 01",
                            "2017 10 01",
                            "2017 10 01",
                            "2017 10 01",
                            "2017 10 01",
                            "2017 10 01",
                            "2017 10 01",
                            "2017 10 01",
                            "2017 10 01",
                            "2017 10 01",
                            "2017 10 01",
                            "2017 10 01",
                            "2017 10 01"
                            ]
        ### 
        return data_folder, HDnames, Starnames, SIMBAD_Spt, YYYYMMDD_LBAND



        
        
        
    ### If to return the Gaia DR2 or Hipparcos distances
    if to_return == "dist":
        ### File containing DR2 distances
        distDR2_file = "distGDR2.dat"
        ### File containing HIP distances
        distHIP_file = "distHIP.dat"
        ### List of star names corresponding to 'HDnames'
        dist_correspond =   [
                            "10Cas",
                            "omiCas",                        
                            "gammaCas",                        
                            "phiAnd",                     
                            "V777Cas",                         
                            "BkCam",                        
                            "17Tau",                        
                            "23Tau",                        
                            "25Tau",                        
                            "28Tau",                        
                            "12Vul",                        
                            "28Cyg",                        
                            "V2113Cyg",                        
                            "V2119Cyg",                        
                            "V2120Cyg",                        
                            "V2123Cyg",                        
                            "V568Cyg",                         
                            "60Cyg",                        
                            "upsCyg",                        
                            "V2162Cyg",                        
                            "16Peg",                        
                            "25Peg",                        
                            "omiAnd",                       
                            "betPsc"                        
                            ]

        return distDR2_file, distHIP_file, dist_correspond
        
        

    ### If to return ...
    if to_return == "Cesar":
        Cesar = "Cesar/DatosBeStarsL_lrrimulo/"
        Cesar_correspond =  [
                            "Cas10.dat",
                            "OmiCas.dat",
                            "gCas.dat",
                            None,
                            "V777.dat",
                            "BkCam.dat",
                            None,
                            "Tau23.dat",
                            "Tau25.dat",
                            "28Tau.dat",
                            None,
                            "Cyg28.dat",
                            "V2113.dat",
                            "V2119.dat",
                            "V2120.dat",
                            "V2123.dat",
                            "V568.dat",
                            "60Cyg.dat",
                            "UpsCyg.dat",
                            "V2162.dat",
                            None,
                            "Peg25.dat",
                            "oAnd.dat",
                            "betPsc.dat"
                            ]
                        
        return Cesar, Cesar_correspond
       
       
       
                        
    ### If to return ...
    if to_return == "Cesar_BR":
        Cesar_BR = "Cesar/"
        Cesar_BR_file = "FluxRatio_Be_Stars.txt"
        Cesar_BR_correspond =   [
                                "10_Cas",
                                "omi_Cas",   
                                "gamma_Cas",
                                "phi_And",
                                "V777_Cas",
                                "Bk_Cam",
                                "17_Tau",
                                "23_Tau",
                                "25_Tau",
                                "28_Tau",
                                "12_Vul",
                                "28_Cyg",
                                "V2113_Cyg",
                                "V2119_Cyg",
                                "V2120_Cyg",
                                "V2123_Cyg",
                                "V568_Cyg",
                                "60_Cyg",
                                "ups_Cyg",
                                "V2162_Cyg",
                                "16_Peg",
                                "25_Peg",
                                "omi_And",
                                "betPsc"
                                ]
                            
        return Cesar_BR, Cesar_BR_file, Cesar_BR_correspond
        
        
        
        
        
    ### If to return ...
    if to_return == "WISE":
        WISE_file = "AllWISE.dat"
        WISE_correspond =   [
                            "10Cas",
                            "omiCas",                        
                            "gammaCas",                        
                            "phiAnd",                     
                            "V777Cas",                         
                            "BkCam",                        
                            "17Tau",                        
                            "23Tau",                        
                            "25Tau",                        
                            "28Tau",                        
                            "12Vul",                        
                            "28Cyg",                        
                            "V2113Cyg",                        
                            "V2119Cyg",                        
                            "V2120Cyg",                        
                            "V2123Cyg",                        
                            "V568Cyg",                         
                            "60Cyg",                        
                            "upsCyg",                        
                            "V2162Cyg",                        
                            "16Peg",                        
                            "25Peg",                        
                            "omiAnd",                       
                            "betPsc"                        
                            ]

        return WISE_file, WISE_correspond

    
    ### If to return...
    if to_return == "Vieira2017_results":
        Vieira =    [
                    None,   
                    None,
                    "5394",  
                    "6811",  
                    "11606", 
                    "20336", 
                    None, 
                    "23480", 
                    "23630", 
                    None, 
                    "187811",
                    "191610",
                    None,
                    "194335",
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    "217891"
                    ]
        ### Results from Vieira 2017 et al. fittings:
        ### Mass [Msun]
        ### W
        ### n
        ### log10(rho0 [g/cm3])
        Vieira_fitted =     [
                            None,
                            None,
                            [(25.,9.,9.),(0.9,0.1,0.1),\
                                    (np.nan,np.nan,np.nan),\
                                    (np.nan,np.nan,np.nan)],  
                            [(5.6,1.,1.),(0.6,0.2,0.2),\
                                    (np.nan,np.nan,np.nan),\
                                    (np.nan,np.nan,np.nan)],  
                            [(9.,2.,2.),(0.7,0.1,0.1),\
                                    (2.3,0.1,0.2),(-12.2,0.1,0.1)], 
                            [(7.,1.,1.),(0.7,0.1,0.1),\
                                    (2.6,0.1,0.2),(-11.2,0.1,0.1)], 
                            None, 
                            [(3.8,0.6,0.6),(0.8,0.1,0.1),\
                                    (2.0,0.1,0.2),(-12.2,0.1,0.1)], 
                            [(5.,1.,1.),(0.8,0.1,0.1),\
                                    (3.2,0.2,0.3),(-12.3,0.3,0.2)], 
                            None, 
                            [(6.5,0.7,0.7),(0.7,0.1,0.1),\
                                    (5.,1.,1.),(-12.1,0.2,0.2)],
                            [(5.,1.,1.),(0.8,0.1,0.1),\
                                    (3.2,0.2,0.2),(-11.2,0.3,0.3)],
                            None,
                            [(11.,2.,2.),(0.7,0.1,0.1),\
                                    (3.3,0.2,0.1),(-11.,0.2,0.2)],
                            None,
                            None,
                            None,
                            None,
                            None,
                            None,
                            None,
                            None,
                            None,
                            [(5.3,0.9,0.9),(0.32,0.06,0.06),\
                                    (2.4,0.1,0.1),(-12.1,0.1,0.2)]
                            ]

    
    
    
    ### If to return the vsini values according to the BeSS database
    if to_return == "vsini":

        vsini =     [
                    125.,
                    195.,
                    432.,
                    85.,
                    280.,
                    328.,
                    170.,
                    240.,
                    140.,
                    290.,
                    245.,
                    300.,
                    None, 
                    315.,
                    None,
                    None,
                    105.,
                    320.,
                    185.,
                    None,
                    100.,
                    130.,
                    260.,
                    95.
                    ]
        
        return vsini

    if to_return == "BeSS":
        BeSSfolder = "BeSS/"
        ### List of HD names in ascending order
        HDnames_BeSS =  [
                        "HD144",   
                        "HD4180",
                        "HD5394",  
                        "HD6811",  
                        "HD11606", 
                        "HD20336", 
                        "HD23302", 
                        "HD23480", 
                        "HD23630", 
                        "HD23862", 
                        "HD187811",
                        "HD191610",
                        "HD193009",
                        "HD194335",
                        "HD194883",
                        "HD195907",
                        "HD197419",
                        "HD200310",
                        "HD202904",
                        "HD204722",
                        "HD208057",
                        "HD210129",
                        "HD217675",
                        "HD217891"
                        ]        
        
        return BeSSfolder, HDnames_BeSS



############
### 

def returnDATA_LBAND():
    """
    Returns 'DATA_LBAND', a list...
    """

    ### 
    data_folder, HDnames, Starnames, SIMBAD_Spt, YYYYMMDD_LBAND = \
            List_Stars("stars")
    DATA_LBAND = []
    ### Loop over all stars
    for i in range(0,len(HDnames)):
        DATA_LBAND.append(  [
                HDnames[i],     ### [0] HD name
                Starnames[i],   ### [1] star's popular name
                [], ### [2] dist, err_dist, source
                YYYYMMDD_LBAND[i],
                [
                [], ### [4][0] indexes of transitions
                [], ### [4][1] theoretical_lambda [Angstroms]
                [], ### [4][2] goodness
                [], ### [4][3] Center_old, Center_new, err_center [Angstroms]
                [], ### [4][4] continuum, err_continuum
                [], ### [4][5] flux, err_flux [erg s^-1 cm^-2]
                [], ### [4][6] EW, err_EW [Angstroms]
                [], ### [4][7] core, err_core [?]
                [], ### [4][8] gaussianFWHM, err [Angstroms]
                []  ### [4][9] lorentzianFWHM, err [Angstroms]
                ],
                [
                [], ### [5][0] F(B), errF(B), lamb1, lamb2 [erg s^-1 cm^-2],[A]
                [], ### [5][1] F(R), errF(R), lamb1, lamb2 [erg s^-1 cm^-2],[A]
                [], ### [5][2] B, errB, R, errR     [mag]
                [], ### [5][3] MB, errMB, MR, errMR [mag]
                []  ### [5][4] alphaL, erralphaL
                ],
                [
                "", ### [6][0] date of WISE measurements
                [], ### [6][1] W1, errW1, W2, errW2, W3, errW3, W4, errW4
                [], ### [6][2] alphaW1W2, erralphaW1W2 
                [], ### [6][3] alphaW2W3, erralphaW2W3 
                [], ### [6][4] alphaW3W4, erralphaW3W4
                []  ### [6][5] MW1, errMW1, MW2, errMW2, MW3, errMW3, MW4, errMW4 
                ],
                "", ### [7] Variability status
                SIMBAD_Spt[i],  ### [8] SIMBAD Spectral type
                "", ### [9] vsini [km/s]
                "", ### [10] (Halpha)[WISE time] EW [A], err_EW [A], PS [km/s], err_PS [km/s], FWHM [km/s], err_FWHM [km/s]  
                ""  ### [11] (Halpha)[Lband time] EW [A], err_EW [A], PS [km/s], err_PS [km/s], FWHM [km/s], err_FWHM [km/s]  
                            ]   )


    ############
    ### Obtaining distances with errors, to be inserted in 'DATA_LBAND[:][2]'

    distDR2_file, distHIP_file, dist_correspond = List_Stars("dist")
    ### Reading file containing Gaia DR2 parallaxes
    fDR2 = open(data_folder+distDR2_file,"r")
    linesDR2 = fDR2.readlines()
    fDR2.close()
    ### Reading file containing Hipparcos parallaxes
    fHIP = open(data_folder+distHIP_file,"r")
    linesHIP = fHIP.readlines()
    fHIP.close()

    ### Inserting parallaxes in the lists 'plx_DR2' and 'plx_HIP'
    plx_DR2 = []
    plx_HIP = []
    for j in range(0,len(dist_correspond)):
        for i in range(0,len(linesDR2)):
            if linesDR2[i].split()[0] == dist_correspond[j]:
                if linesDR2[i].split()[1] != "None":
                    plx_DR2.append( [float(linesDR2[i].split()[6]),\
                                    float(linesDR2[i].split()[7])]
                                    )
                else:
                    plx_DR2.append( [np.nan,\
                                    np.nan]
                                    )
        for i in range(0,len(linesHIP)):
            if linesHIP[i].split()[0] == dist_correspond[j]:
                if linesHIP[i].split()[1] != "Ignored":
                    plx_HIP.append( [float(linesHIP[i].split()[1]),\
                                    float(linesHIP[i].split()[2])]
                                    )
                else:
                    plx_HIP.append( [np.nan,\
                                    np.nan]
                                    )

    ### Inserting distances with errors in 'DATA_LBAND[:][2]'
    for ifile in range(0,len(dist_correspond)):
        if not np.isnan(plx_DR2[ifile][0]): ### preference for DR2
            DATA_LBAND[ifile][2] = [1e3/plx_DR2[ifile][0],\
                                    1e3*plx_DR2[ifile][1]/plx_DR2[ifile][0]**2.,\
                                    "DR2"]
        elif not np.isnan(plx_HIP[ifile][0]):   ### if DR2 is empty, use HIP
            DATA_LBAND[ifile][2] = [1e3/plx_HIP[ifile][0],\
                                    1e3*plx_HIP[ifile][1]/plx_HIP[ifile][0]**2.,\
                                    "HIP"]
        else:   ### if DR2 and HIP are empty, attribute NaN
            DATA_LBAND[ifile][2] = [np.nan,np.nan,None]        


    ############
    ### Obtaining data on the Lband lines, to be inserted in 
    ### 'DATA_LBAND[:][4][:]'

    ### 
    Cesar, Cesar_correspond = List_Stars("Cesar")
    files_cesar = glob.glob(data_folder+Cesar+"*")

    ### sorting 'files_cesar' in the order of 'Cesar_correspond'
    fc_auxi = []
    for j in range(0,len(Cesar_correspond)):
        if Cesar_correspond[j] != None:
            for i in range(0,len(files_cesar)):
                if Cesar_correspond[j] in files_cesar[i]:
                    fc_auxi.append(files_cesar[i])
        else:
            fc_auxi.append(None)
    files_cesar = [elem for elem in fc_auxi]


    ### Storing the data on 'DATA_LBAND[:][4][:]' 
    for ifile in range(0,len(files_cesar)):
        ### 
        if files_cesar[ifile] != None:
            f0 = open(files_cesar[ifile],"r")
            lines = f0.readlines()
            f0.close()

            for iline in range(31,len(lines)):
                line_now = lines[iline].split()
                
                DATA_LBAND[ifile][4][0].append(int(line_now[0]))
                DATA_LBAND[ifile][4][0].append(int(line_now[1]))
                
                DATA_LBAND[ifile][4][1].append(float(line_now[2]))
                
                DATA_LBAND[ifile][4][2].append(float(line_now[3]))
                
                DATA_LBAND[ifile][4][3].append(float(line_now[4]))
                DATA_LBAND[ifile][4][3].append(float(line_now[5]))
                DATA_LBAND[ifile][4][3].append(float(line_now[6]))
                
                DATA_LBAND[ifile][4][4].append(float(line_now[7]))
                DATA_LBAND[ifile][4][4].append(float(line_now[8]))
                
                DATA_LBAND[ifile][4][5].append(float(line_now[9]))
                DATA_LBAND[ifile][4][5].append(float(line_now[10]))
                
                DATA_LBAND[ifile][4][6].append(float(line_now[11]))
                DATA_LBAND[ifile][4][6].append(float(line_now[12]))
            
                DATA_LBAND[ifile][4][7].append(float(line_now[13]))
                DATA_LBAND[ifile][4][7].append(float(line_now[14]))
            
                DATA_LBAND[ifile][4][8].append(float(line_now[15]))
                DATA_LBAND[ifile][4][8].append(float(line_now[16]))
            
                DATA_LBAND[ifile][4][9].append(float(line_now[17]))
                DATA_LBAND[ifile][4][9].append(float(line_now[18]))
        
        else:
            
            DATA_LBAND[ifile][4][0].append(np.nan)
            DATA_LBAND[ifile][4][0].append(np.nan)
                                        
            DATA_LBAND[ifile][4][1].append(np.nan)
        
            DATA_LBAND[ifile][4][2].append(np.nan)
        
            DATA_LBAND[ifile][4][3].append(np.nan)
            DATA_LBAND[ifile][4][3].append(np.nan)
            DATA_LBAND[ifile][4][3].append(np.nan)
        
            DATA_LBAND[ifile][4][4].append(np.nan)
            DATA_LBAND[ifile][4][4].append(np.nan)
        
            DATA_LBAND[ifile][4][5].append(np.nan)
            DATA_LBAND[ifile][4][5].append(np.nan)
        
            DATA_LBAND[ifile][4][6].append(np.nan)
            DATA_LBAND[ifile][4][6].append(np.nan)
        
            DATA_LBAND[ifile][4][7].append(np.nan)
            DATA_LBAND[ifile][4][7].append(np.nan)
        
            DATA_LBAND[ifile][4][8].append(np.nan)
            DATA_LBAND[ifile][4][8].append(np.nan)
        
            DATA_LBAND[ifile][4][9].append(np.nan)
            DATA_LBAND[ifile][4][9].append(np.nan)        


    ############
    ### Obtaining continuum Lband data, to be inserted in 'DATA_LBAND[:][5][:]'

    Cesar_BR, Cesar_BR_file, Cesar_BR_correspond = List_Stars("Cesar_BR")
    fBR = open(data_folder+Cesar_BR+Cesar_BR_file,"r")
    lines = fBR.readlines()
    fBR.close()


    lamb1_BL=3.41 ; lamb2_BL=3.47   ### limits of BL [microns]
    lamb1_RL=3.93 ; lamb2_RL=4.00   ### limits of RL [microns]
    Nnpts=50
    
    ### Returns the SED of VEGA (9 nm - 160 microns):
    ### * array of lambdas [Angstroms]
    ### * array of Flambda [erg cm^-2 s^-1 A^-1]
    xlp,ylp = lrr.VEGA_spct("spct1")
    
    ### vectors of lambda and dlambda in [Angstroms]:
    llambB = np.array([lamb1_BL+(lamb2_BL-lamb1_BL)/\
        float(Nnpts-1)*float(i) for i in range(0,Nnpts)])*1e4
    llambR = np.array([lamb1_RL+(lamb2_RL-lamb1_RL)/\
        float(Nnpts-1)*float(i) for i in range(0,Nnpts)])*1e4
    dllambB = np.array([llambB[i+1]-llambB[i] for i in range(0,Nnpts-1)])
    dllambR = np.array([llambR[i+1]-llambR[i] for i in range(0,Nnpts-1)])
    

    ylpfB = np.array([lrr.interpLinND([llambB[i]],[xlp],ylp) \
        for i in range(0,Nnpts)])
    ylpfR = np.array([lrr.interpLinND([llambR[i]],[xlp],ylp) \
        for i in range(0,Nnpts)])
    
    ### Fluxes FB and FR of Vega
    B_Vega = lrr.integrate_trapezia(ylpfB,dllambB)  ### [erg s^-1 cm^-2]
    R_Vega = lrr.integrate_trapezia(ylpfR,dllambR)  ### [erg s^-1 cm^-2]
    
    ### Flux densities in B and R of Vega:
    FB_Vega = B_Vega/(lamb2_BL-lamb1_BL)*1e-4   ### [erg s^-1 cm^-2 A^-1]
    FR_Vega = R_Vega/(lamb2_RL-lamb1_RL)*1e-4   ### [erg s^-1 cm^-2 A^-1]


    for icor in range(0,len(Cesar_BR_correspond)):
        for iline in range(0,len(lines)):
            if lines[iline].split()[0] == "Obj.":
                for iobj in range(1,len(lines[iline].split())):
                    if lines[iline].split()[iobj] == Cesar_BR_correspond[icor]:
                        
                        lamb2B = float(lines[iline+1].split()[iobj]) ### [Angs]
                        lamb1B = float(lines[iline+2].split()[iobj]) ### [Angs]
                        B__L = float(lines[iline+3].split()[iobj])
                        ### TODO: Cesar should send me the true errors
                        errB__L = 0.1*abs(B__L)
                        
                        lamb2R = float(lines[iline+4].split()[iobj]) ### [Angs]
                        lamb1R = float(lines[iline+5].split()[iobj]) ### [Angs]
                        R__L = float(lines[iline+6].split()[iobj])
                        ### TODO: Cesar should send me the true errors
                        errR__L = 0.1*abs(R__L)
                        
                        ### 'alpha_L' and error:
                        alpha__L = 1.-np.log(B__L/\
                            (lamb2B-lamb1B)*(lamb2R-lamb1R)/R__L)/\
                            np.log((lamb2R+lamb1R)/(lamb2B+lamb1B))
                        erralpha__L = 1./np.log((lamb2R+lamb1R)/\
                            (lamb2B+lamb1B))*\
                            np.sqrt(errB__L**2./B__L**2.+errR__L**2./R__L**2.)
                        
                        ### Apparent magnitudes B and R and error [mag]
                        MagB = -2.5*np.log10(B__L/(lamb2B-lamb1B)/FB_Vega)
                        errMagB = 2.5/np.log(10.)*errB__L/B__L
                        
                        MagR = -2.5*np.log10(R__L/(lamb2R-lamb1R)/FR_Vega)
                        errMagR = 2.5/np.log(10.)*errR__L/R__L
                        
                        ### Absolute magnitudes B and R and error [mag]
                        absMagB = MagB-5.*np.log10(DATA_LBAND[icor][2][0])+5.
                        errabsMagB = np.sqrt(errMagB**2.+\
                            (5./np.log(10.))**2.*DATA_LBAND[icor][2][1]**2./\
                            DATA_LBAND[icor][2][0]**2.)
                        
                        absMagR = MagR-5.*np.log10(DATA_LBAND[icor][2][0])+5.
                        errabsMagR = np.sqrt(errMagR**2.+\
                            (5./np.log(10.))**2.*DATA_LBAND[icor][2][1]**2./\
                            DATA_LBAND[icor][2][0]**2.)                    
                    
                        ### Storing the data:
                        DATA_LBAND[icor][5][0].append(B__L)
                        DATA_LBAND[icor][5][0].append(errB__L)
                        DATA_LBAND[icor][5][0].append(lamb1B)
                        DATA_LBAND[icor][5][0].append(lamb2B)
                    
                        DATA_LBAND[icor][5][1].append(R__L)
                        DATA_LBAND[icor][5][1].append(errR__L)
                        DATA_LBAND[icor][5][1].append(lamb1R)
                        DATA_LBAND[icor][5][1].append(lamb2R)
                    
                        DATA_LBAND[icor][5][2].append(MagB)
                        DATA_LBAND[icor][5][2].append(errMagB)
                        DATA_LBAND[icor][5][2].append(MagR)
                        DATA_LBAND[icor][5][2].append(errMagR)
                    
                        DATA_LBAND[icor][5][3].append(absMagB)
                        DATA_LBAND[icor][5][3].append(errabsMagB)
                        DATA_LBAND[icor][5][3].append(absMagR)
                        DATA_LBAND[icor][5][3].append(errabsMagR)
                    
                        DATA_LBAND[icor][5][4].append(alpha__L)
                        DATA_LBAND[icor][5][4].append(erralpha__L)







    ############
    ### 

    ### defining the vector of alphas
    alphamin = -6.
    alphamax = 2.
    N_pts = 80
    alphavec = np.array([alphamin+(alphamax-alphamin)*float(i)/float(N_pts)\
        for i in range(0,N_pts+1)])
    
    ### obtaining the 'zp's for the evaluation of colors as a function 
    ### of alphas
    zp1 = lrr.obtain_pogson_zp('spct1',"wise1")
    zp2 = lrr.obtain_pogson_zp('spct1',"wise2")
    zp3 = lrr.obtain_pogson_zp('spct1',"wise3")
    zp4 = lrr.obtain_pogson_zp('spct1',"wise4")
    
    ### obtaining colors as functions of alphas
    colorW1W2, err_facW1W2 = lrr.color_from_alpha(alphavec,"wise1","wise2",zp1,zp2)
    colorW2W3, err_facW2W3 = lrr.color_from_alpha(alphavec,"wise2","wise3",zp2,zp3)
    colorW3W4, err_facW3W4 = lrr.color_from_alpha(alphavec,"wise3","wise4",zp3,zp4)






    WISE_file, WISE_correspond = List_Stars("WISE")

    fWISE = open(data_folder+WISE_file,"r")
    linesWISE = fWISE.readlines()
    fWISE.close()

    for ifile in range(0,len(WISE_correspond)):
        for iline in range(0,len(linesWISE)):
            linenow = linesWISE[iline].split()
            if linenow[0] == WISE_correspond[ifile]:
                if linenow[1] != "None":
                    
                    if linenow[3] != "null" and linenow[4] != "null":
                        W1now = float(linenow[3])
                        errW1now = float(linenow[4])
                    else:
                        W1now = np.nan
                        errW1now = np.nan                    
                    
                    if linenow[5] != "null" and linenow[6] != "null":
                        W2now = float(linenow[5])
                        errW2now = float(linenow[6])
                    else:
                        W2now = np.nan
                        errW2now = np.nan
                    
                    if linenow[7] != "null" and linenow[8] != "null":
                        W3now = float(linenow[7])
                        errW3now = float(linenow[8])
                    else:
                        W3now = np.nan
                        errW3now = np.nan
                    
                    if linenow[9] != "null" and linenow[10] != "null":
                        W4now = float(linenow[9])
                        errW4now = float(linenow[10])
                    else:
                        W4now = np.nan
                        errW4now = np.nan
                
                    ### Date of measurements
                    DATA_LBAND[ifile][6][0] = "2012 03 14"
                
                    ### Storing the data:
                    DATA_LBAND[ifile][6][1].append(W1now)
                    DATA_LBAND[ifile][6][1].append(errW1now)
                    DATA_LBAND[ifile][6][1].append(W2now)
                    DATA_LBAND[ifile][6][1].append(errW2now)
                    DATA_LBAND[ifile][6][1].append(W3now)
                    DATA_LBAND[ifile][6][1].append(errW3now)
                    DATA_LBAND[ifile][6][1].append(W4now)
                    DATA_LBAND[ifile][6][1].append(errW4now)

                    ### Obtaining alphas and errors associated with each color:
                    if not np.isnan(W1now-W2now):
                        alphaW1W2now = lrr.interpLinND([W1now-W2now],\
                            [colorW1W2],alphavec,allow_extrapolation="no")
                        erralphaW1W2now = lrr.interpLinND([W1now-W2now],\
                            [colorW1W2],err_facW1W2,allow_extrapolation="no")*\
                            np.sqrt(errW1now**2.+errW2now**2.)
                    else:
                        alphaW1W2now = np.nan
                        erralphaW1W2now = np.nan
                    
                    if not np.isnan(W2now-W3now):
                        alphaW2W3now = lrr.interpLinND([W2now-W3now],\
                            [colorW2W3],alphavec,allow_extrapolation="no")
                        erralphaW2W3now = lrr.interpLinND([W2now-W3now],\
                            [colorW2W3],err_facW2W3,allow_extrapolation="no")*\
                            np.sqrt(errW2now**2.+errW3now**2.)
                    else:
                        alphaW2W3now = np.nan
                        erralphaW2W3now = np.nan
                    
                    if not np.isnan(W3now-W4now):
                        alphaW3W4now = lrr.interpLinND([W3now-W4now],\
                            [colorW3W4],alphavec,allow_extrapolation="no")
                        erralphaW3W4now = lrr.interpLinND([W3now-W4now],\
                            [colorW3W4],err_facW3W4,allow_extrapolation="no")*\
                            np.sqrt(errW3now**2.+errW4now**2.)
                    else:
                        alphaW3W4now = np.nan
                        erralphaW3W4now = np.nan
                
                    ### Storing the alphas with errors:
                    DATA_LBAND[ifile][6][2].append(alphaW1W2now)
                    DATA_LBAND[ifile][6][2].append(erralphaW1W2now)
                
                    DATA_LBAND[ifile][6][3].append(alphaW2W3now)
                    DATA_LBAND[ifile][6][3].append(erralphaW2W3now)
                
                    DATA_LBAND[ifile][6][4].append(alphaW3W4now)
                    DATA_LBAND[ifile][6][4].append(erralphaW3W4now)
                                                 
                    ### Distance modulus and error [mag]
                    munow = 5.*np.log10(DATA_LBAND[ifile][2][0])-5.
                    errmunow = 5./np.log(10.)*DATA_LBAND[ifile][2][1]/\
                            DATA_LBAND[ifile][2][0]
                
                    ### Storing absolute WISE magnitudes and errors [mag]
                    DATA_LBAND[ifile][6][5].append(W1now-munow)
                    DATA_LBAND[ifile][6][5].append(np.sqrt(errW1now**2.+\
                            errmunow**2.))
                    DATA_LBAND[ifile][6][5].append(W2now-munow)
                    DATA_LBAND[ifile][6][5].append(np.sqrt(errW2now**2.+\
                            errmunow**2.))
                    DATA_LBAND[ifile][6][5].append(W3now-munow)
                    DATA_LBAND[ifile][6][5].append(np.sqrt(errW3now**2.+\
                            errmunow**2.))
                    DATA_LBAND[ifile][6][5].append(W4now-munow)
                    DATA_LBAND[ifile][6][5].append(np.sqrt(errW4now**2.+\
                            errmunow**2.))



                else:
            
                    DATA_LBAND[ifile][6][0] = "2012 03 14"
                
                    DATA_LBAND[ifile][6][1].append(np.nan)
                    DATA_LBAND[ifile][6][1].append(np.nan)
                    DATA_LBAND[ifile][6][1].append(np.nan)
                    DATA_LBAND[ifile][6][1].append(np.nan)
                    DATA_LBAND[ifile][6][1].append(np.nan)
                    DATA_LBAND[ifile][6][1].append(np.nan)
                    DATA_LBAND[ifile][6][1].append(np.nan)
                    DATA_LBAND[ifile][6][1].append(np.nan)
                                                 
                    DATA_LBAND[ifile][6][2].append(np.nan)
                    DATA_LBAND[ifile][6][2].append(np.nan)
                    
                    DATA_LBAND[ifile][6][3].append(np.nan)
                    DATA_LBAND[ifile][6][3].append(np.nan)
                
                    DATA_LBAND[ifile][6][4].append(np.nan)
                    DATA_LBAND[ifile][6][4].append(np.nan)                                                 
                                                 
                    DATA_LBAND[ifile][6][5].append(np.nan)
                    DATA_LBAND[ifile][6][5].append(np.nan)
                    DATA_LBAND[ifile][6][5].append(np.nan)
                    DATA_LBAND[ifile][6][5].append(np.nan)
                    DATA_LBAND[ifile][6][5].append(np.nan)
                    DATA_LBAND[ifile][6][5].append(np.nan)
                    DATA_LBAND[ifile][6][5].append(np.nan)
                    DATA_LBAND[ifile][6][5].append(np.nan)                                                 
                                                 

    ############
    ### TODO: Variability status
    
    ### 
    BeSSfolder, HDnames_BeSS = List_Stars("BeSS")
    
    for ihd in range(0,len(HDnames_BeSS)):
    
        ### 
        f0 = open(data_folder+BeSSfolder+HDnames_BeSS[ihd]+".dat","r")
        f0lines = f0.readlines()
        f0.close()

        ### 
        tau0 = (np.nan,"???")
        tau1 = (np.nan,"???")
        for iline in range(0,len(f0lines)):
            if f0lines[iline].split()[0] == "#######":
                tau0 = (float(f0lines[iline+1].split()[0]),\
                        f0lines[iline+1].split()[1])
                tau1 = (float(f0lines[iline+2].split()[0]),\
                        f0lines[iline+2].split()[1])

        ### 
        DATA_LBAND[ihd][7] = [tau0,tau1]


    ############
    ### vsini
    
    vsini = List_Stars("vsini")
    for ifile in range(0,len(vsini)):
        if vsini[ifile] is not None:
            DATA_LBAND[ifile][9] = vsini[ifile]
        else:
            DATA_LBAND[ifile][9] = np.nan



    ############
    ### 
    
    ### 
    BeSSfolder, HDnames_BeSS = List_Stars("BeSS")
    
    for ihd in range(0,len(HDnames_BeSS)):
    
        ### 
        f0 = open(data_folder+BeSSfolder+HDnames_BeSS[ihd]+".dat","r")
        f0lines = f0.readlines()
        f0.close()

        ### 
        elem0 = np.nan
        elem1 = np.nan
        elem2 = np.nan
        elem3 = np.nan
        elem4 = np.nan
        elem5 = np.nan
        for iline in range(0,len(f0lines)):
            if f0lines[iline].split()[0] == "#####":
                elem0 = float(f0lines[iline+1].split()[1])
                elem1 = float(f0lines[iline+1].split()[2])
                elem2 = float(f0lines[iline+1].split()[3])
                elem3 = float(f0lines[iline+1].split()[4])
                elem4 = float(f0lines[iline+1].split()[5])
                elem5 = float(f0lines[iline+1].split()[6])
                
        DATA_LBAND[ihd][10] = [elem0,elem1,elem2,elem3,elem4,elem5]

        ### 
        elem0 = np.nan
        elem1 = np.nan
        elem2 = np.nan
        elem3 = np.nan
        elem4 = np.nan
        elem5 = np.nan
        for iline in range(0,len(f0lines)):
            if f0lines[iline].split()[0] == "#####":
                elem0 = float(f0lines[iline+2].split()[1])
                elem1 = float(f0lines[iline+2].split()[2])
                elem2 = float(f0lines[iline+2].split()[3])
                elem3 = float(f0lines[iline+2].split()[4])
                elem4 = float(f0lines[iline+2].split()[5])
                elem5 = float(f0lines[iline+2].split()[6])
                
        DATA_LBAND[ihd][11] = [elem0,elem1,elem2,elem3,elem4,elem5]
    

    return DATA_LBAND









############
### 

def LBAND_lines_extract(DATA_LBAND):
    """
    
    """

    max_goodness = 3.

    Nmax = 100
    fluxhumphreys = np.zeros((len(DATA_LBAND),Nmax+1,2))
    fluxhumphreys[:,:,:] = np.nan
    EWhumphreys = np.zeros((len(DATA_LBAND),Nmax+1,2))
    EWhumphreys[:,:,:] = np.nan
    GFWHMhumphreys = np.zeros((len(DATA_LBAND),Nmax+1,2))
    GFWHMhumphreys[:,:,:] = np.nan

    for ifile in range(0,len(DATA_LBAND)):
        goodness_key = 0
        for itrans in range(0,int(len(DATA_LBAND[ifile][4][0])/2)):
            for ii in range(0,Nmax+1):
                #if DATA_LBAND[ifile][4][2][itrans] < max_goodness:
                #    goodness_key = 1
                if (DATA_LBAND[ifile][4][0][2*itrans] == ii and \
                        DATA_LBAND[ifile][4][0][2*itrans+1] == 6)\
                        and DATA_LBAND[ifile][4][2][itrans] >= max_goodness\
                        and goodness_key == 0:
                    fluxhumphreys[ifile,ii,0] = \
                            DATA_LBAND[ifile][4][5][2*itrans]
                    fluxhumphreys[ifile,ii,1] = \
                            DATA_LBAND[ifile][4][5][2*itrans+1]
                    EWhumphreys[ifile,ii,0] = \
                            DATA_LBAND[ifile][4][6][2*itrans]
                    EWhumphreys[ifile,ii,1] = \
                            DATA_LBAND[ifile][4][6][2*itrans+1]
                    GFWHMhumphreys[ifile,ii,0] = \
                            DATA_LBAND[ifile][4][8][2*itrans]
                    GFWHMhumphreys[ifile,ii,1] = \
                            DATA_LBAND[ifile][4][8][2*itrans+1]
                            
        #print(fluxhumphreys[ifile,:,0])
                            

    fluxBra = np.zeros((len(DATA_LBAND),2))
    fluxBra[:,:] = np.nan
    EWBra = np.zeros((len(DATA_LBAND),2))
    EWBra[:,:] = np.nan
    GFWHMBra = np.zeros((len(DATA_LBAND),2))
    GFWHMBra[:,:] = np.nan

    for ifile in range(0,len(DATA_LBAND)):
        goodness_key = 0
        for itrans in range(0,int(len(DATA_LBAND[ifile][4][0])/2)):
            #if DATA_LBAND[ifile][4][2][itrans] < max_goodness:
            #    goodness_key = 1
            if (DATA_LBAND[ifile][4][0][2*itrans] == 5 and \
                    DATA_LBAND[ifile][4][0][2*itrans+1] == 4)\
                    and DATA_LBAND[ifile][4][2][itrans] >= max_goodness\
                    and goodness_key == 0:
                fluxBra[ifile,0] = \
                        DATA_LBAND[ifile][4][5][2*itrans]
                fluxBra[ifile,1] = \
                        DATA_LBAND[ifile][4][5][2*itrans+1]
                EWBra[ifile,0] = \
                        DATA_LBAND[ifile][4][6][2*itrans]
                EWBra[ifile,1] = \
                        DATA_LBAND[ifile][4][6][2*itrans+1]
                GFWHMBra[ifile,0] = \
                        DATA_LBAND[ifile][4][8][2*itrans]
                GFWHMBra[ifile,1] = \
                        DATA_LBAND[ifile][4][8][2*itrans+1]


    fluxPfg = np.zeros((len(DATA_LBAND),2))
    fluxPfg[:,:] = np.nan
    EWPfg = np.zeros((len(DATA_LBAND),2))
    EWPfg[:,:] = np.nan
    GFWHMPfg = np.zeros((len(DATA_LBAND),2))
    GFWHMPfg[:,:] = np.nan

    for ifile in range(0,len(DATA_LBAND)):
        goodness_key = 0
        for itrans in range(0,int(len(DATA_LBAND[ifile][4][0])/2)):
            #if DATA_LBAND[ifile][4][2][itrans] < max_goodness:
            #    goodness_key = 1
            if (DATA_LBAND[ifile][4][0][2*itrans] == 8 and \
                    DATA_LBAND[ifile][4][0][2*itrans+1] == 5)\
                    and DATA_LBAND[ifile][4][2][itrans] >= max_goodness\
                    and goodness_key == 0:
                fluxPfg[ifile,0] = \
                        DATA_LBAND[ifile][4][5][2*itrans]
                fluxPfg[ifile,1] = \
                        DATA_LBAND[ifile][4][5][2*itrans+1]
                EWPfg[ifile,0] = \
                        DATA_LBAND[ifile][4][6][2*itrans]
                EWPfg[ifile,1] = \
                        DATA_LBAND[ifile][4][6][2*itrans+1]
                GFWHMPfg[ifile,0] = \
                        DATA_LBAND[ifile][4][8][2*itrans]
                GFWHMPfg[ifile,1] = \
                        DATA_LBAND[ifile][4][8][2*itrans+1]



    return fluxhumphreys, EWhumphreys, GFWHMhumphreys, \
            fluxBra, EWBra, GFWHMBra, \
            fluxPfg, EWPfg, GFWHMPfg



















def make_table_obs1(DATA_LBAND,fileNAME):
    
    import pyhdust.lrr.jdutil as jdutil
    
    fluxhumphreys, EWhumphreys, GFWHMhumphreys, \
    fluxBra, EWBra, GFWHMBra, \
    fluxPfg, EWPfg, GFWHMPfg = LBAND_lines_extract(DATA_LBAND)
    
    
    printline = []
    
    printline.append("\\begin{table*}"+"\n")
    printline.append("\caption{Main L-band observables of the 24 Be stars \label{table_obs1}}"+"\n")
    printline.append("\\begin{tabular}{lllllllll}"+"\n")
    printline.append("\hline"+"\n")
    printline.append("HD name & Object & Spectral & $B_L\,[\mathrm{mag}]$ & $\\alpha_L$ & $F(Hu14)/$ & $F(Hu14)/$ & $F(Hu14)/$ & $D_\mathrm{pc}$ \\\ "+"\n" )
    printline.append("&  & type &  &  & $F(\lambda_B)\, [\mathrm{A}]$ & $F(Br\\alpha)$ & $F(Pf\gamma)$ &  \\\ "+"\n")
    printline.append("\hline"+"\n")
    

    for ifile in range(0,len(DATA_LBAND)):
        
        year = int(DATA_LBAND[ifile][3].split()[0])
        month = int(DATA_LBAND[ifile][3].split()[1])
        day = int(DATA_LBAND[ifile][3].split()[2])
        julian_now = jdutil.date_to_jd(year,month,day)
        
        FlambdaB = DATA_LBAND[ifile][5][0][0]/(DATA_LBAND[ifile][5][0][3]-\
                        DATA_LBAND[ifile][5][0][2])
        errFlambdaB = DATA_LBAND[ifile][5][0][1]/(DATA_LBAND[ifile][5][0][3]-\
                        DATA_LBAND[ifile][5][0][2])
    
        
        BL = DATA_LBAND[ifile][5][2][0]
        errBL = DATA_LBAND[ifile][5][2][1]
        if np.isnan(BL) or np.isnan(errBL):
            BL_write = "-"
        else:
            BL_write = " $ "+str(lrr.round_sig(BL,3))+" \pm "+\
                    str(lrr.round_sig(errBL,2))+" $ "
                    
        alphaL = DATA_LBAND[ifile][5][2][0]
        erralphaL = DATA_LBAND[ifile][5][2][1]
        if np.isnan(alphaL) or np.isnan(erralphaL):
            alphaL_write = "-"
        else:
            alphaL_write = " $ "+str(lrr.round_sig(alphaL,3))+" \pm "+\
                    str(lrr.round_sig(erralphaL,2))+" $ "
                    
        
        F14_FlambB = fluxhumphreys[ifile][14][0]/FlambdaB
        errF14_FlambB = abs(F14_FlambB)*\
            np.sqrt(\
            fluxhumphreys[ifile][14][1]**2./fluxhumphreys[ifile][14][0]**2.+\
            errFlambdaB**2./FlambdaB**2.\
            )
        if np.isnan(F14_FlambB) or np.isnan(errF14_FlambB):
            F14_FlambB_write = "-"
        else:
            F14_FlambB_write = " $ "+str(lrr.round_sig(F14_FlambB,3))+" \pm "+\
                    str(lrr.round_sig(errF14_FlambB,2))+" $ "

        F14_FBra = fluxhumphreys[ifile][14][0]/fluxBra[ifile][0]
        errF14_FBra = abs(F14_FBra)*\
            np.sqrt(\
            fluxhumphreys[ifile][14][1]**2./fluxhumphreys[ifile][14][0]**2.+\
            fluxBra[ifile][1]**2./fluxBra[ifile][0]**2.\
            )
        if np.isnan(F14_FlambB) or np.isnan(errF14_FlambB):
            F14_FBra_write = "-"
        else:
            F14_FBra_write = " $ "+str(lrr.round_sig(F14_FBra,3))+" \pm "+\
                    str(lrr.round_sig(errF14_FBra,2))+" $ "

        F14_FPfg = fluxhumphreys[ifile][14][0]/fluxPfg[ifile][0]
        errF14_FPfg = abs(F14_FPfg)*\
            np.sqrt(\
            fluxhumphreys[ifile][14][1]**2./fluxhumphreys[ifile][14][0]**2.+\
            fluxPfg[ifile][1]**2./fluxPfg[ifile][0]**2.\
            )
        if np.isnan(F14_FPfg) or np.isnan(errF14_FPfg):
            F14_FPfg_write = "-"
        else:
            F14_FPfg_write = " $ "+str(lrr.round_sig(F14_FPfg,3))+" \pm "+\
                    str(lrr.round_sig(errF14_FPfg,2))+" $ "    


        dist = DATA_LBAND[ifile][2][0]
        errdist = DATA_LBAND[ifile][2][1]
        sourcedist = DATA_LBAND[ifile][2][2]
        if np.isnan(dist) or np.isnan(errdist):
            dist_write = "-"
        else:
            if sourcedist != "DR2":
                add = "$^*$"
            else:
                add = ""
            dist_write = " $ "+str(lrr.round_sig(dist,3))+" \pm "+\
                    str(lrr.round_sig(errdist,2))+" $ "+add


        
        printline.append("HD "+DATA_LBAND[ifile][0]+" & "+DATA_LBAND[ifile][1]+" & "+\
                DATA_LBAND[ifile][8]+" & "+\
                BL_write+" & "+\
                alphaL_write+" & "+\
                F14_FlambB_write+" & "+\
                F14_FBra_write+" & "+\
                F14_FPfg_write+" & "+\
                dist_write+" \\\ "+"\n")
                

    printline.append("\end{tabular}"+"\n")
    printline.append("\end{table*}"+"\n")


    f0 = open(fileNAME,"w")
    for elem in printline:
        f0.write(elem)
    f0.close()

    return 



def make_table_obs2(DATA_LBAND,fileNAME):
    
    
    printline = []
    
    printline.append("\\begin{table*}"+"\n")
    printline.append("\caption{Complementary observables of the 24 Be stars \label{table_obs2}}"+"\n")
    printline.append("\\begin{tabular}{lcccccccc}"+"\n")
    printline.append("\hline"+"\n")
    printline.append("HD name & $v\sin i$ & $EW,(H\\alpha)$ & $\\tau_0^{-1}$ & $\\tau_1^{-1}$ & $W3$ & $\\alpha_{W1-W2}$ & $\\alpha_{W2-W3}$ & $\\alpha_{W3-W4}$ \\\ "+"\n")
    printline.append(" & [$\mathrm{km\,s^{-1}}$] & $[\\mathrm{A}]$ & [$10^5\\times\mathrm{day}^{-1}$] & [$10^5\\times\mathrm{day}^{-1}$] & [mag] & &  &  \\\ "+"\n")
    printline.append("\hline"+"\n")
    


    for ifile in range(0,len(DATA_LBAND)):
                
        EWlambHa = DATA_LBAND[ifile][11][0]
        errEWlambHa = DATA_LBAND[ifile][11][1]
        if np.isnan(EWlambHa) or np.isnan(errEWlambHa):
            EWlambHa_write = "-"
        else:
            EWlambHa_write = " $ "+str(lrr.round_sig(EWlambHa,3))+" \pm "+\
                    str(lrr.round_sig(errEWlambHa,2))+" $ "        
        
        
        if not np.isnan(DATA_LBAND[ifile][9]):
            vsini_write = " $ "+str(int(DATA_LBAND[ifile][9]))+" $ "                    
        else:
            vsini_write = "-"
            
        tau0_var = DATA_LBAND[ifile][7][0][0]
        tau1_var = DATA_LBAND[ifile][7][1][0]
        if np.isnan(tau0_var):
            tau0_var_write = "-"
        else:
            tau0_var_write = " $ "+str(lrr.round_sig(1e5/tau0_var,3))+" $ "    
        if np.isnan(tau1_var):
            tau1_var_write = "-"
        else:
            tau1_var_write = " $ "+str(lrr.round_sig(1e5/tau1_var,3))+" $ "


        W3 = DATA_LBAND[ifile][6][1][4]
        errW3 = DATA_LBAND[ifile][6][1][5]
        if np.isnan(W3) or np.isnan(errW3):
            W3_write = "-"
        else:
            W3_write = " $ "+str(lrr.round_sig(W3,3))+" \pm "+\
                    str(lrr.round_sig(errW3,2))+" $ "
                    
        alpha12 = DATA_LBAND[ifile][6][2][0]
        erralpha12 = DATA_LBAND[ifile][6][2][1]
        if np.isnan(alpha12) or np.isnan(erralpha12):
            alpha12_write = "-"
        else:
            alpha12_write = " $ "+str(lrr.round_sig(alpha12,3))+" \pm "+\
                    str(lrr.round_sig(erralpha12,2))+" $ "

        alpha23 = DATA_LBAND[ifile][6][3][0]
        erralpha23 = DATA_LBAND[ifile][6][3][1]
        if np.isnan(alpha23) or np.isnan(erralpha23):
            alpha23_write = "-"
        else:
            alpha23_write = " $ "+str(lrr.round_sig(alpha23,3))+" \pm "+\
                    str(lrr.round_sig(erralpha23,2))+" $ "

        alpha34 = DATA_LBAND[ifile][6][4][0]
        erralpha34 = DATA_LBAND[ifile][6][4][1]
        if np.isnan(alpha34) or np.isnan(erralpha34):
            alpha34_write = "-"
        else:
            alpha34_write = " $ "+str(lrr.round_sig(alpha34,3))+" \pm "+\
                    str(lrr.round_sig(erralpha34,2))+" $ "

                            
        
        printline.append("HD "+DATA_LBAND[ifile][0]+" & "+\
                vsini_write+" & "+\
                EWlambHa_write+" & "+\
                tau0_var_write+" & "+\
                tau1_var_write+" & "+\
                W3_write+" & "+\
                alpha12_write+" & "+\
                alpha23_write+" & "+\
                alpha34_write+" \\\ "+"\n")
                

    printline.append("\end{tabular}"+"\n")
    printline.append("\end{table*}"+"\n")


    f0 = open(fileNAME,"w")
    for elem in printline:
        f0.write(elem)
    f0.close()

    return 




def make_bigtables_obs(DATA_LBAND,fileNAME):
    
    import pyhdust.lrr.jdutil as jdutil
    
    fluxhumphreys, EWhumphreys, GFWHMhumphreys, \
    fluxBra, EWBra, GFWHMBra, \
    fluxPfg, EWPfg, GFWHMPfg = LBAND_lines_extract(DATA_LBAND)
    
    
    first_row = [DATA_LBAND[ifile][0] for ifile in range(0,12)]
    second_row = [DATA_LBAND[ifile][0] for ifile in range(12,24)]

    printline = []
    rows = [first_row,second_row]
    for irow in range(0,len(rows)):
        
        printline.append("\\begin{landscape}"+"\n")
        printline.append("\\begin{table}"+"\n")
        printline.append("\caption{List of Be stars and their respective bumps selected for this study \label{my_selection}}"+"\n")
        printline.append("\\begin{tabular}{rrrrrrrrrrrrr}"+"\n")
        printline.append("\hline"+"\n")
        auxi_printline = ""
        for i in range(0,len(rows[irow])):
            auxi_printline += " & HD "+rows[irow][i]
        auxi_printline += " \\\ "+"\n"
        printline.append(auxi_printline)
        printline.append("\hline"+"\n")
    
        columns = [[] for elem in range(0,len(rows[irow])+1)]
        
        for i in range(14,35+1):
            if i != 17 and i != 19:
                columns[0].append("$F_{14}/F_{19}$ & ".replace("14",str(i)))
                
                for icol in range(1,len(columns)):
                    idx = [DATA_LBAND[ifile][0] for ifile in \
                            range(0,len(DATA_LBAND))].index(rows[irow][icol-1])
                    if ~np.isnan(fluxhumphreys[idx,i,0]) and \
                            ~np.isnan(fluxhumphreys[idx,i,1]) and \
                            ~np.isnan(fluxhumphreys[idx,19,0]) and\
                            ~np.isnan(fluxhumphreys[idx,19,1]):
                        val = fluxhumphreys[idx,i,0]/fluxhumphreys[idx,19,0]
                        err = abs(val)*np.sqrt(\
                            fluxhumphreys[idx,i,1]**2./fluxhumphreys[idx,i,0]**2.+\
                            fluxhumphreys[idx,19,1]**2./fluxhumphreys[idx,19,0]**2.\
                                                )
                        columns[icol].append("$ "+str(lrr.round_sig(val,2))\
                                        +" \pm "+str(lrr.round_sig(err,1))+" $")
                    else:
                        columns[icol].append("-")
                
                    if icol != len(columns)-1:
                        columns[icol][-1] += " & "
                    else:
                        columns[icol][-1] += " \\\ \n"

                summ = ""
                for ii in range(0,len(columns)):
                    summ += columns[ii][-1]
                printline.append(summ)
        
        printline.append("\hline"+"\n")
        
        for i in range(14,35+1):
            if i != 17:
                columns[0].append("$-EW_{14}/\lambda$ & ".replace("14",str(i)))
                        
                for icol in range(1,len(columns)):
                    idx = [DATA_LBAND[ifile][0] for ifile in \
                            range(0,len(DATA_LBAND))].index(rows[irow][icol-1])
                    if ~np.isnan(EWhumphreys[idx,i,0]) and \
                            ~np.isnan(EWhumphreys[idx,i,1]):
                        val = -EWhumphreys[idx,i,0]/spt.hydrogenlinewl(i, 6)*1e-6
                        err = EWhumphreys[idx,i,1]/spt.hydrogenlinewl(i, 6)*1e-6
                        columns[icol].append("$ "+str(lrr.round_sig(val,2))+" \pm "+\
                                            str(lrr.round_sig(err,1))+" $")
                    else:
                        columns[icol].append("-")
                                        
                    if icol != len(columns)-1:
                        columns[icol][-1] += " & "
                    else:
                        columns[icol][-1] += " \\\ \n"        

                summ = ""
                for ii in range(0,len(columns)):
                    summ += columns[ii][-1]
                printline.append(summ)
                
        printline.append("\hline"+"\n")
        
        
        columns[0].append("$-EW_{Br\\alpha}/\lambda$ & ")
        for icol in range(1,len(columns)):
            idx = [DATA_LBAND[ifile][0] for ifile in \
                    range(0,len(DATA_LBAND))].index(rows[irow][icol-1])
            if ~np.isnan(EWBra[idx,0]) and \
                    ~np.isnan(EWBra[idx,1]):
                val = -EWBra[idx,0]/spt.hydrogenlinewl(5, 4)*1e-6
                err = EWBra[idx,1]/spt.hydrogenlinewl(5, 4)*1e-6
                columns[icol].append("$ "+str(lrr.round_sig(val,2))+" \pm "+\
                                    str(lrr.round_sig(err,1))+" $")
            else:
                columns[icol].append("-")
                                        
            if icol != len(columns)-1:
                columns[icol][-1] += " & "
            else:
                columns[icol][-1] += " \\\ \n"        

        summ = ""
        for ii in range(0,len(columns)):
            summ += columns[ii][-1]
        printline.append(summ)

        columns[0].append("$-EW_{Pf\gamma}/\lambda$ & ")
        for icol in range(1,len(columns)):
            idx = [DATA_LBAND[ifile][0] for ifile in \
                    range(0,len(DATA_LBAND))].index(rows[irow][icol-1])
            if ~np.isnan(EWPfg[idx,0]) and \
                    ~np.isnan(EWPfg[idx,1]):
                val = -EWPfg[idx,0]/spt.hydrogenlinewl(8, 5)*1e-6
                err = EWPfg[idx,1]/spt.hydrogenlinewl(8, 5)*1e-6
                columns[icol].append("$ "+str(lrr.round_sig(val,2))+" \pm "+\
                                    str(lrr.round_sig(err,1))+" $")
            else:
                columns[icol].append("-")
                                        
            if icol != len(columns)-1:
                columns[icol][-1] += " & "
            else:
                columns[icol][-1] += " \\\ \n"        

        summ = ""
        for ii in range(0,len(columns)):
            summ += columns[ii][-1]
        printline.append(summ)
        
        
        printline.append("\end{tabular}"+"\n")
        printline.append("\end{table}"+"\n")
        printline.append("\end{landscape}"+"\n")
        
        printline.append("\n")
        printline.append("\n")
        printline.append("\n")
        
        

    f0 = open(fileNAME,"w")
    for elem in printline:
        f0.write(elem)
    f0.close()


    return 











def hpd_grid(sample, alpha=0.05, roundto=2):
    """
    This function was found in: 
    https://github.com/PacktPublishing/Bayesian-Analysis-with-Python/blob/master/Chapter%201/hpd%20(1).py
    
    ............
    ............
    
    Calculate highest posterior density (HPD) of array for given alpha. 
    The HPD is the minimum width Bayesian credible interval (BCI). 
    The function works for multimodal distributions, returning more than one mode
    
    Parameters
    ----------
    
    sample : Numpy array or python list
        An array containing MCMC samples
    alpha : float
        Desired probability of type I error (defaults to 0.05)
    roundto: integer
        Number of digits after the decimal point for the results
    Returns
    ----------
    hpd: array with the lower 
          
    """

    import numpy as np
    import scipy.stats.kde as kde

    sample = np.asarray(sample)
    sample = sample[~np.isnan(sample)]
    ### get upper and lower bounds
    l = np.min(sample)
    u = np.max(sample)
    density = kde.gaussian_kde(sample)
    x = np.linspace(l, u, 2000)
    y = density.evaluate(x)
    ### y = density.evaluate(x, l, u) waitting for PR to be accepted
    xy_zipped = zip(x, y/np.sum(y))
    xy = sorted(xy_zipped, key=lambda x: x[1], reverse=True)
    xy_cum_sum = 0
    hdv = []
    for val in xy:
        xy_cum_sum += val[1]
        hdv.append(val[0])
        if xy_cum_sum >= (1-alpha):
            break
    hdv.sort()
    diff = (u-l)/20  ### differences of 5%
    hpd = []
    hpd.append(round(min(hdv), roundto))
    for i in range(1, len(hdv)):
        if hdv[i]-hdv[i-1] >= diff:
            hpd.append(round(hdv[i-1], roundto))
            hpd.append(round(hdv[i], roundto))
    hpd.append(round(max(hdv), roundto))
    ite = iter(hpd)
    hpd = list(zip(ite, ite))
    modes = []
    for value in hpd:
         x_hpd = x[(x > value[0]) & (x < value[1])]
         y_hpd = y[(x > value[0]) & (x < value[1])]
         modes.append(round(x_hpd[np.argmax(y_hpd)], roundto))
    return hpd, x, y, modes




def get_otherpapers():
    """
    This is a procedure under construction...
    """
    
    
    ### Directory of the location of the data on the Be stars
    data_folder = "./../MedidasLogs/otherpapers/"
    
    
    f0 = open(data_folder+"Granada_etal_2010_table1.txt","r")
    GranadaT1lines = f0.readlines()
    f0.close()
    GranadaT1lines = [GranadaT1lines[i].split() \
            for i in range(11,len(GranadaT1lines))]
    #print(GranadaT1lines)
    #print("")

    f0 = open(data_folder+"Granada_etal_2010_table2.txt","r")
    GranadaT2lines = f0.readlines()
    f0.close()    
    GranadaT2lines = [GranadaT2lines[i].split() \
            for i in range(13,len(GranadaT2lines))]
    #print(GranadaT2lines)
    #print("")
    
    f0 = open(data_folder+"Granada_etal_2010_table3.txt","r")
    GranadaT3lines = f0.readlines()
    f0.close()        
    GranadaT3lines = [GranadaT3lines[i].split() \
            for i in range(13,len(GranadaT3lines))]
    #print(GranadaT3lines)
    #print("")

    f0 = open(data_folder+"Granada_etal_2010_table4.txt","r")
    GranadaT4lines = f0.readlines()
    f0.close()            
    GranadaT4lines = [GranadaT4lines[i].split() \
            for i in range(13,len(GranadaT4lines))]
    #print(GranadaT4lines)
    #print("")
    
    f0 = open(data_folder+"Mennickent_etal_2009_table3.txt","r")
    MennickentT3lines = f0.readlines()
    f0.close()                
    MennickentT3lines = [MennickentT3lines[i].split() \
            for i in range(14,len(MennickentT3lines))]    
    #print(MennickentT3lines)
    #print("")
    
    ### 
    GranadaDATA = []
    MennickentDATA = []
    Nmax = 100
    for istar in range(0,len(GranadaT1lines)-1):
        GranadaDATA.append( [
                GranadaT1lines[istar][2],           ### [0] HD name
                GranadaT1lines[istar][0]+" "+\
                        GranadaT1lines[istar][1],   ### [1] Star name
                np.zeros((Nmax+1,2)),               ### [2] flux humphreys [cgs]
                np.zeros((Nmax+1,2)),               ### [3] EW humphreys [Angs]
                np.zeros((Nmax+1,2)),               ### [4] FWHM humphreys [Angs]
                np.array([np.nan,np.nan]),          ### [5] flux Bralpha [cgs]
                np.array([np.nan,np.nan]),          ### [6] EW Bralpha [Angs]
                np.array([np.nan,np.nan]),          ### [7] FWHM Bralpha [Angs]
                np.array([np.nan,np.nan]),          ### [8] flux Pfgamma [cgs]
                np.array([np.nan,np.nan]),          ### [9] EW Pfgamma [Angs]
                np.array([np.nan,np.nan]),          ### [10] FWHM Pfgamma [Angs]
                ])
        
        GranadaDATA[istar][2][:,:] = np.nan
        GranadaDATA[istar][3][:,:] = np.nan
        GranadaDATA[istar][4][:,:] = np.nan
    

        for iline in range(0,len(GranadaT2lines)):

            if GranadaT2lines[iline][1] == "6":
                hump_idx = int(GranadaT2lines[iline][0])
            
                if GranadaT2lines[iline][istar+3] == "None" or \
                        GranadaT2lines[iline][istar+3] == "em":
                    GranadaDATA[istar][3][hump_idx,0] = np.nan
                    GranadaDATA[istar][3][hump_idx,1] = np.nan
                else:
                    GranadaDATA[istar][3][hump_idx,0] = \
                            float(GranadaT2lines[iline][istar+3])
                    GranadaDATA[istar][3][hump_idx,1] = \
                            abs(GranadaDATA[istar][3][hump_idx,0])*0.1

            if GranadaT2lines[iline][0] == "5" \
                    and GranadaT2lines[iline][1] == "4":
            
                if GranadaT2lines[iline][istar+3] == "None" or \
                        GranadaT2lines[iline][istar+3] == "em":
                    GranadaDATA[istar][6][0] = np.nan
                    GranadaDATA[istar][6][1] = np.nan
                else:
                    GranadaDATA[istar][6][0] = \
                            float(GranadaT2lines[iline][istar+3])
                    GranadaDATA[istar][6][1] = \
                            abs(GranadaDATA[istar][6][0])*0.1

            if GranadaT2lines[iline][0] == "8" \
                    and GranadaT2lines[iline][1] == "5":
            
                if GranadaT2lines[iline][istar+3] == "None" or \
                        GranadaT2lines[iline][istar+3] == "em":
                    GranadaDATA[istar][9][0] = np.nan
                    GranadaDATA[istar][9][1] = np.nan
                else:
                    GranadaDATA[istar][9][0] = \
                            float(GranadaT2lines[iline][istar+3])
                    GranadaDATA[istar][9][1] = \
                            abs(GranadaDATA[istar][9][0])*0.1    






        for iline in range(0,len(GranadaT3lines)):

            if GranadaT3lines[iline][1] == "6":
                hump_idx = int(GranadaT3lines[iline][0])
            
                if GranadaT3lines[iline][istar+3] == "None" or \
                        GranadaT3lines[iline][istar+3] == "em":
                    GranadaDATA[istar][2][hump_idx,0] = np.nan
                    GranadaDATA[istar][2][hump_idx,1] = np.nan
                else:
                    GranadaDATA[istar][2][hump_idx,0] = \
                            float(GranadaT3lines[iline][istar+3])*1e-13
                    GranadaDATA[istar][2][hump_idx,1] = \
                            abs(GranadaDATA[istar][2][hump_idx,0])*0.1

            if GranadaT3lines[iline][0] == "5" \
                    and GranadaT3lines[iline][1] == "4":
            
                if GranadaT3lines[iline][istar+3] == "None" or \
                        GranadaT3lines[iline][istar+3] == "em":
                    GranadaDATA[istar][5][0] = np.nan
                    GranadaDATA[istar][5][1] = np.nan
                else:
                    GranadaDATA[istar][5][0] = \
                            float(GranadaT3lines[iline][istar+3])*1e-13
                    GranadaDATA[istar][5][1] = \
                            abs(GranadaDATA[istar][5][0])*0.1

            if GranadaT3lines[iline][0] == "8" \
                    and GranadaT3lines[iline][1] == "5":
            
                if GranadaT3lines[iline][istar+3] == "None" or \
                        GranadaT3lines[iline][istar+3] == "em":
                    GranadaDATA[istar][8][0] = np.nan
                    GranadaDATA[istar][8][1] = np.nan
                else:
                    GranadaDATA[istar][8][0] = \
                            float(GranadaT3lines[iline][istar+3])*1e-13
                    GranadaDATA[istar][8][1] = \
                            abs(GranadaDATA[istar][8][0])*0.1    
    
    






        for iline in range(0,len(GranadaT4lines)):

            if GranadaT4lines[iline][1] == "6":
                hump_idx = int(GranadaT4lines[iline][0])
            
                if GranadaT4lines[iline][istar+3] == "None" or \
                        GranadaT4lines[iline][istar+3] == "em":
                    GranadaDATA[istar][4][hump_idx,0] = np.nan
                    GranadaDATA[istar][4][hump_idx,1] = np.nan
                else:
                    GranadaDATA[istar][4][hump_idx,0] = \
                            float(GranadaT4lines[iline][istar+3])
                    GranadaDATA[istar][4][hump_idx,1] = 40.

            if GranadaT4lines[iline][0] == "5" \
                    and GranadaT4lines[iline][1] == "4":
            
                if GranadaT4lines[iline][istar+3] == "None" or \
                        GranadaT4lines[iline][istar+3] == "em":
                    GranadaDATA[istar][7][0] = np.nan
                    GranadaDATA[istar][7][1] = np.nan
                else:
                    GranadaDATA[istar][7][0] = \
                            float(GranadaT4lines[iline][istar+3])
                    GranadaDATA[istar][7][1] = 40.

            if GranadaT4lines[iline][0] == "8" \
                    and GranadaT4lines[iline][1] == "5":
            
                if GranadaT4lines[iline][istar+3] == "None" or \
                        GranadaT4lines[iline][istar+3] == "em":
                    GranadaDATA[istar][10][0] = np.nan
                    GranadaDATA[istar][10][1] = np.nan
                else:
                    GranadaDATA[istar][10][0] = \
                            float(GranadaT4lines[iline][istar+3])
                    GranadaDATA[istar][10][1] = 40.    
    
    
    
    
    
        MennickentDATA.append( [
                "",                                     ### [0] HD name
                MennickentT3lines[istar][0]+" "+\
                        MennickentT3lines[istar][1],    ### [1] Star name
                np.zeros((Nmax+1,2)),                   ### [2] flux humphreys [cgs]
                np.zeros((Nmax+1,2)),                   ### [3] EW humphreys [Angs]
                np.zeros((Nmax+1,2)),                   ### [4] FWHM humphreys [Angs]
                np.array([np.nan,np.nan]),              ### [5] flux Bralpha [cgs]
                np.array([np.nan,np.nan]),              ### [6] EW Bralpha [Angs]
                np.array([np.nan,np.nan]),              ### [7] FWHM Bralpha [Angs]
                np.array([np.nan,np.nan]),              ### [8] flux Pfgamma [cgs]
                np.array([np.nan,np.nan]),              ### [9] EW Pfgamma [Angs]
                np.array([np.nan,np.nan]),              ### [10] FWHM Pfgamma [Angs]
                ])
        
        MennickentDATA[istar][2][:,:] = np.nan
        MennickentDATA[istar][3][:,:] = np.nan
        MennickentDATA[istar][4][:,:] = np.nan    
    




        if MennickentT3lines[istar][2] == "None" or \
                MennickentT3lines[istar][2] == "em":
            MennickentDATA[istar][5][0] = np.nan
            MennickentDATA[istar][5][1] = np.nan
        else:
            MennickentDATA[istar][5][0] = \
                    float(MennickentT3lines[istar][2])
            MennickentDATA[istar][5][1] = \
                    abs(MennickentDATA[istar][5][0])*0.1


        if MennickentT3lines[istar][3] == "None" or \
                MennickentT3lines[istar][3] == "em":
            MennickentDATA[istar][8][0] = np.nan
            MennickentDATA[istar][8][1] = np.nan
        else:
            MennickentDATA[istar][8][0] = \
                    float(MennickentT3lines[istar][3])
            MennickentDATA[istar][8][1] = \
                    abs(MennickentDATA[istar][8][0])*0.1


        if MennickentT3lines[istar][4] == "None" or \
                MennickentT3lines[istar][4] == "em":
            MennickentDATA[istar][2][14,0] = np.nan
            MennickentDATA[istar][2][14,1] = np.nan
        else:
            MennickentDATA[istar][2][14,0] = \
                    float(MennickentT3lines[istar][4])
            MennickentDATA[istar][2][14,1] = \
                    abs(MennickentDATA[istar][2][14,0])*0.1




        if MennickentT3lines[istar][5] == "None" or \
                MennickentT3lines[istar][5] == "em":
            MennickentDATA[istar][6][0] = np.nan
            MennickentDATA[istar][6][1] = np.nan
        else:
            MennickentDATA[istar][6][0] = \
                    float(MennickentT3lines[istar][5])
            MennickentDATA[istar][6][1] = \
                    abs(MennickentDATA[istar][6][0])*0.1


        if MennickentT3lines[istar][6] == "None" or \
                MennickentT3lines[istar][6] == "em":
            MennickentDATA[istar][9][0] = np.nan
            MennickentDATA[istar][9][1] = np.nan
        else:
            MennickentDATA[istar][9][0] = \
                    float(MennickentT3lines[istar][6])
            MennickentDATA[istar][9][1] = \
                    abs(MennickentDATA[istar][9][0])*0.1


        if MennickentT3lines[istar][7] == "None" or \
                MennickentT3lines[istar][7] == "em":
            MennickentDATA[istar][3][14,0] = np.nan
            MennickentDATA[istar][3][14,1] = np.nan
        else:
            MennickentDATA[istar][3][14,0] = \
                    float(MennickentT3lines[istar][7])
            MennickentDATA[istar][3][14,1] = \
                    abs(MennickentDATA[istar][3][14,0])*0.1



        if MennickentT3lines[istar][8] == "None" or \
                MennickentT3lines[istar][8] == "em":
            MennickentDATA[istar][7][0] = np.nan
            MennickentDATA[istar][7][1] = np.nan
        else:
            MennickentDATA[istar][7][0] = \
                    float(MennickentT3lines[istar][8])
            MennickentDATA[istar][7][1] = 40.


        if MennickentT3lines[istar][9] == "None" or \
                MennickentT3lines[istar][9] == "em":
            MennickentDATA[istar][10][0] = np.nan
            MennickentDATA[istar][10][1] = np.nan
        else:
            MennickentDATA[istar][10][0] = \
                    float(MennickentT3lines[istar][9])
            MennickentDATA[istar][10][1] = 40.


        if MennickentT3lines[istar][10] == "None" or \
                MennickentT3lines[istar][10] == "em":
            MennickentDATA[istar][4][14,0] = np.nan
            MennickentDATA[istar][4][14,1] = np.nan
        else:
            MennickentDATA[istar][4][14,0] = \
                    float(MennickentT3lines[istar][10])
            MennickentDATA[istar][4][14,1] = 40.


    ### HD names of the stars, from Simbad.
    MennickentDATA[0][0] = "120324"
    MennickentDATA[1][0] = "120991"
    MennickentDATA[2][0] = "105521"
    MennickentDATA[3][0] = "148259"
    MennickentDATA[4][0] = "186272"
    MennickentDATA[5][0] = "178175"
    MennickentDATA[6][0] = "29441"
    MennickentDATA[7][0] = "187811"


    
    
    return GranadaDATA, MennickentDATA












