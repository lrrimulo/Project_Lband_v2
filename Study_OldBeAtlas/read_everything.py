import glob as glob
import numpy as np
import pyhdust as hdt
import pyhdust.phc as phc
import pyhdust.lrr as lrr
import pyhdust.lrr.roche_singlestar as rss


def domain_PLgrid():
    """
    This function returns the lists with the domain of the grid of HDUST 
    models. 
    """
    npar = ['2.0','2.5','3.0','3.5','4.0','4.5']
    sigpar = ['0.00','0.02','0.05','0.12','0.28','0.68','1.65','4.00']
    Mpar = ['04.20','07.70','10.80','14.60','20.00']
    obpar = ['1.20','1.40']
    cosipar=[   '-4.3711e-08','0.11147','0.22155','0.33381','0.44464',\
            '0.55484','0.66653','0.77824','0.88862','1.0']
    
    return npar, sigpar, Mpar, obpar, cosipar



def read_everything():
    """
    This procedure reads all HDUSTs's fullseds, sources and temperatures
    and stores the important contents of the files in the list 
    'fullsed_contents', which is returned (with other things).
    """

    ### Paths to the fullsed, source and temperature files:
    #fullsed_path = '../OldBeAtlas/fullsed_v3/'
    fullsed_path = '../OldBeAtlas/fullsed_v2/'
    #fullsed_path = '../OldBeAtlas/fullsed/'
    source_path = '../OldBeAtlas/source/'
    temps_path = '../OldBeAtlas/temperatures/'

    ### assumed distance [parsecs] for the calculations
    dist_std = 10.


    ###########################
    
    ### The domain of the power-law grid:
    npar, sigpar, Mpar, obpar, cosipar = domain_PLgrid()
    filepars=[npar,sigpar,Mpar,obpar]

    print("Reading the OldBeAtlas files...")
    print("")

    files_fullsed=sorted(glob.glob(fullsed_path+'*'))	
    files_source=sorted(glob.glob(source_path+'*'))
    files_temps=sorted(glob.glob(temps_path+'*'))

    files_fullsed_new=[]    ### will receive the names of the fullsed
                            ### files to be opened.

    ### It is assumed that the names of the fullsed files are of the form:
    ### fullsed_mod191_PLn4.0_sig0.05_h072_Rd050.0_Be_M04.80_ob1.10_H0.30_Z0.014_bE_Ell.sed2
    ### or
    ### fullsed_mod01_PLn3.5_sig0.00_h060_Rd050.0_Be_M03.80_ob1.20_H0.77_Z0.014_bE_Ell.sed2
    for i in range(0,len(npar)):
        for j in range(0,len(sigpar)):
            for k in range(0,len(Mpar)):
                for l in range(0,len(obpar)):
                    ### Check if there is a fullsed file with some specific
                    ### values of n, Sig, M and ob:
                    for ifile in xrange(0,len(files_fullsed)):
                        if ('PLn{0}_sig{1}_h072_Rd050.0_Be_'\
                            .format(filepars[0][i],filepars[1][j])+\
                            'M{0}_ob{1}_H0.30_Z0.014_bE_Ell'\
                            .format(filepars[2][k],filepars[3][l]) in \
                            files_fullsed[ifile]) \
                            or ('PLn{0}_sig{1}_h060_Rd050.0_Be_'\
                            .format(filepars[0][i],filepars[1][j])+\
                            'M{0}_ob{1}_H0.30_Z0.014_bE_Ell'\
                            .format(filepars[2][k],filepars[3][l]) in \
                            files_fullsed[ifile]):
                                
                                ### elements of 'files_fullsed_new' are = 
                                ### [ [n,sig,M,ob], "fullsed file" ]
                                files_fullsed_new.append([[ filepars[0][i],\
                                                            filepars[1][j],\
                                                            filepars[2][k],\
                                                            filepars[3][l]],\
                                                        files_fullsed[ifile]]) 

    ### Now that we have a 'files_fullsed_new' list complete, the idea is
    ### to create lists of source and temperature files in such a way that, 
    ### for each fullsed file stored in a 'files_fullsed_new' line, 
    ### there is a line with the correspondent source file in 
    ### 'files_source_new' and a line with the correspondent temp file in 
    ### 'files_temps_new'. 

    ### It is assumed that the names of the source files are of the form:
    ### Be_M03.40_ob1.45_H0.54_Z0.014_bE_Ell.txt
    ### (Notice that the it is contained in the name of the fullsed file.)
    files_source_new=[] ### will receive the names of the source
                        ### files to be opened.
    for iffn in xrange(0,len(files_fullsed_new)):
        ### Check if there is a source file whose name is contained in 
        ### the name of the specific fullsed file:
        for ifs in xrange(0,len(files_source)):
            if files_source[ifs].replace(source_path,'').replace('.txt','')\
                        in files_fullsed_new[iffn][1]:
                files_source_new.append(files_source[ifs])
    ### (Notice that I have assumed that there is always a source file 
    ### associated with a fullsed file. That is not the case with the 
    ### temperature files below.)


    ### It is assumed that the names of the temperature files are of the form:
    ### mod126_PLn3.5_sig0.28_h072_Rd050.0_Be_M09.60_ob1.20_H0.30_Z0.014_bE_Ell30_avg.temp
    ### (Notice that the it is contained in the name of the fullsed file.)
    files_temps_new=[]  ### will receive the names of the temperature
                        ### files to be opened.
    for iffn in xrange(0,len(files_fullsed_new)):
        achei=0 ### Some fullsed files may not have correspondent temp files,
                ### like the ones of purely photospherical models.
        ### Check if there is a temperature file whose name is contained in
        ### the name of the specific fullsed file.
        ### If not, add "EMPTY" to the 'files_temps_new' list.
        for ifs in xrange(0,len(files_temps)):
            if files_temps[ifs].replace(temps_path,'').replace(\
                    '30_avg.temp','')\
                    in files_fullsed_new[iffn][1]:
                files_temps_new.append(files_temps[ifs])
                achei=1
        if achei == 0:
            files_temps_new.append('EMPTY')


    ### Now, building the 'fullsed_contents' list. It will contain the 
    ### relevant contents of all available fullsed, source and temperature 
    ### files of the grid.

    fullsed_contents=[] ### This list will receive the important contents
                        ### of all the files
    for ifile in xrange(0,len(files_fullsed_new)):

        ### Reading the fullsed, source and temperature files:
    
        fullsedtest=files_fullsed_new[ifile][1]
        f0=open(fullsedtest,'r')
        f0linhas=f0.readlines()
        f0.close()

        sourcetest=files_source_new[ifile]
        f1=open(sourcetest,'r')
        f1linhas=f1.readlines()
        f1.close()    

        tempstest=files_temps_new[ifile]
        if tempstest != 'EMPTY':
            ### OBS: This pyhdust procedure will print 
            ### "'FILE' completely read!"
            ncr, ncmu, ncphi, nLTE, nNLTE, Rstarz, Raz, betaz, dataz, \
                pcr, pcmu, pcphi = hdt.readtemp(tempstest)
            abttemp=[
                    [dataz[0,i,ncmu/2,0]/Rstarz for i in \
                            xrange(0,len(dataz[0,:,ncmu/2,0]))],
                    [dataz[3,i,ncmu/2,0] for i in \
                            xrange(0,len(dataz[3,:,ncmu/2,0]))]
                    ]
        else:
            abttemp=[
                    [np.nan,np.nan],
                    [np.nan,np.nan]
                    ]


        ### Obtaining each element of the 'fullsed_contents' list

        nobs=int(f0linhas[3].split()[1])    ### number of different cosi
        nlbd=int(f0linhas[3].split()[0])    ### number of lambdas for each cosi
        contents=[    
            fullsedtest,                    ### 0: Name of fullsed file
            np.zeros(nobs),                 ### 1: will receive the cosi's
            np.zeros((nobs,nlbd,3)),        ### 2: will receive the SED
            sourcetest,                     ### 3: Name of source file
            np.zeros(5),                    ### 4: will receive the 
                                            ###    parameters of the star 
                                            ###    (source)
            tempstest,                      ### 5: Name of temperature file
            np.zeros((2,len(abttemp[0]))),  ### 6: will receive the temp 
                                            ###    profile
            [[],[]]
            ]
        contents[1][:] = np.nan
        contents[2][:] = np.nan
        contents[4][:] = np.nan
        contents[6][:] = np.nan


        ### Receiving cosi and SED ("1" and "2")
        for iobs in xrange(0,nobs):
            mu = float(f0linhas[5+iobs*nlbd].split()[0])
            contents[1][iobs] = mu
            for ilbd in xrange(0, nlbd):
                auxi = f0linhas[5+iobs*nlbd+ilbd].split()
                contents[2][iobs, ilbd, 0] = float(auxi[2])
                contents[2][iobs, ilbd, 1] = float(auxi[3])
                contents[2][iobs, ilbd, 2] = float(auxi[7])


        ### Receiving parameters of the star (source) ("4")
        contents[4][0] = float(f1linhas[3].split()[2]) ### M
        contents[4][1] = float(f1linhas[4].split()[2]) ### R_pole
        contents[4][2] = float(f1linhas[5].split()[2]) ### W
        contents[4][3] = float(f1linhas[6].split()[2]) ### L
        contents[4][4] = float(f1linhas[7].split()[2]) ### Beta_GD
    
        ### Receiving the temperature profile ("6")
        for i in xrange(0,len(contents[6][0,:])):
            contents[6][0,i] = abttemp[0][i]
            contents[6][1,i] = abttemp[1][i]
        
        ### elements of 'fullsed_contents':
        fullsed_contents.append([files_fullsed_new[ifile][0],contents])

    print("")

    return files_fullsed_new, files_source_new, files_temps_new, fullsed_contents, \
        fullsed_path, source_path, temps_path, dist_std



def check_completeness(files_fullsed_new):
    """
    This function checks what fraction of the grid was already computed 
    and what are the missing models.
    
    WARNING: It assumes that the grid is hyperrectangular. This however is not 
    the case. Hence, this function needs correction!
    """
    ### The domain of the grid of HDUST models:
    npar, sigpar, Mpar, obpar, cosipar = domain_PLgrid()
    
    
    computed = 0.
    total = 0.
    missing = []
    for i in range(0,len(npar)):
        for j in range(0,len(sigpar)):
            for k in range(0,len(Mpar)):
                for l in range(0,len(obpar)):
                    total += 1.
                    ### Now, check if there is a fullsed file that matches
                    ### the current grid element.
                    found = 0
                    for ifile in range(0,len(files_fullsed_new)):
                        if files_fullsed_new[ifile][0][0] == npar[i] and\
                                files_fullsed_new[ifile][0][1] == sigpar[j] and\
                                files_fullsed_new[ifile][0][2] == Mpar[k] and\
                                files_fullsed_new[ifile][0][3] == obpar[l]:
                            computed += 1.
                            found = 1
                    if found == 0:
                        missing.append([npar[i],sigpar[j],Mpar[k],obpar[l]])
    
    fraction_computed = computed/total

    return fraction_computed, missing






