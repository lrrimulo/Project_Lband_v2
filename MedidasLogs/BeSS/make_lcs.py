import itertools as it
import glob as glob
import numpy as np
import matplotlib.pyplot as plt
import pyhdust.spectools as spt
import pyhdust.phc as phc
from scipy.optimize import curve_fit
import pyhdust.lrr.jdutil as jdu
import pyhdust.lrr as lrr


def gaussian_fit(x,sigma2,A):
    """
    This is a gaussian function centered on x = 0 and with area 'A' and 
    variance 'sigma2'.
    """
    return 0.+A/np.sqrt(2.*np.pi*sigma2)*np.exp(-0.5*(x-0.)**2./sigma2)
    
def linear_fit(x,A,B):
    """
    This is a simple linear function.
    """
    return A+B*x


folders_data = ["HD144/","HD4180/","HD5394/","HD6811/","HD11606/","HD20336/",\
        "HD23302/","HD23480/","HD23630/","HD23862/","HD187811/","HD191610/",\
        "HD193009/","HD194335/","HD194883/","HD195907/","HD197419/","HD200310/",\
        "HD202904/","HD204722/","HD208057/","HD210129/","HD217675/","HD217891/"]

Starnames =     ["10 Cas","$o$ Cas","$\\gamma$ Cas","$\\phi$ And",\
                "V777 Cas","Bk Cam","17 Tau","23 Tau","25 Tau",\
                "28 Tau","12 Vul","28 Cyg","V2113 Cyg","V2119 Cyg",\
                "V2120 Cyg","V2123 Cyg","V568 Cyg","60 Cyg",\
                "$\\upsilon$ Cyg","V2162 Cyg","16 Peg","25 Peg",\
                "$o$ And","$\\beta$ Psc"]

### These are the days, months and years to be used in the loop below
days = ["01","02","03","04","05","06","07","08","09","10",\
        "11","12","13","14","15","16","17","18","19","20",
        "21","22","23","24","25","26","27","28","29","30",
        "31"]
months = ["01","02","03","04","05","06","07","08","09","10","11","12"]
years = ["2007","2008","2009","2010","2011","2012","2013","2014",\
        "2015","2016","2017","2018"]

### All possible dates of light curves
dates = [years,months,days]
dates = [elem for elem in it.product(*dates)]

### Dates of L-band and WISE measurements
dateLband = ["2017","10","01"]
dateWISE = ["2012","03","14"]

### Wavelength of Halpha [microns]
lbc = spt.hydrogenlinewl(3, 2)*1e6

### Loop over the folders of data
for folder_data in folders_data:

    ### Obtaining files' names for this specific 'folder_data'
    files_list = glob.glob(folder_data+"*")

    ### 'files_contents' will receive all the data.
    files_contents = []
    ### Loop over all possible dates
    for idate in range(0,len(dates)):
        current_date = dates[idate][0]+dates[idate][1]+dates[idate][2]
        ### Loop over the files
        for ifile in range(0,len(files_list)):
            current_file = files_list[ifile]
            ### 
            if current_date in current_file:
                ### Julian date
                JD = jdu.date_to_jd(int(dates[idate][0]),\
                        int(dates[idate][1]),int(dates[idate][2]))
                ### Reading the file
                f0 = open(current_file,"r")
                linesf0 = f0.readlines()
                f0.close()
            
                ### Saving the table in 'array_data':
                array_data = np.zeros((len(linesf0),2))
                array_data[:,:] = np.nan
                for iline in range(0,len(linesf0)):
                    array_data[iline,0] = float(linesf0[iline].split()[0])
                    array_data[iline,1] = float(linesf0[iline].split()[1])



                ### Selecting region for evaluating "noise"
                forerr_lambs = []
                forerr_data = []
                if len(array_data[:,0]) >= 2 and \
                        np.nanmin(array_data[:,0]) <= lbc*1e4-40. and \
                        np.nanmax(array_data[:,0]) >= lbc*1e4+40.:
                    ### 
                    for itest in range(0,len(array_data[:,0])):
                        ### 
                        if lbc*1e4+30. <= array_data[itest,0] <= lbc*1e4+40.:
                            forerr_lambs.append(array_data[itest,0])
                            forerr_data.append(array_data[itest,1])

                ### If the data contains a region around the Halpha line 
                ### and a few lines for evaluating the "noise":
                ### (Minimal resolution around 16000.)
                if len(array_data[:,0]) >= 2 and \
                        np.nanmin(array_data[:,0]) <= lbc*1e4-40. and \
                        np.nanmax(array_data[:,0]) >= lbc*1e4+40. and \
                        len(forerr_lambs) >= 25:
                            
                            
                            
                    files_contents.append([
                        current_file,
                        JD,
                        array_data,
                        "",  ### [3] (EW, errEW) [Angstroms]
                        "",  ### [4] (PS, errPS) [km/s]
                        ""]) ### [5] [(FWHM, errFWHM) [km/s], (A, errA) [km/s]]
            

                    
                    hwidth = 2000.
    
                    xlp=array_data[:,0]*1e-4 ### lambda [microns]
                    ylp=array_data[:,1]      ### BeSS's flux [unit = ???]


                    ### For the region where noise will be evaluated:
                    ### Returning an array of velocities [km/s]: 'xplott' 
                    ### and an array with the normalized flux: 'yplott'                    
                    xplott,yplott = spt.lineProf(np.array(forerr_lambs)*1e-4, \
                            np.array(forerr_data), lbc, hwidth=hwidth)
                    ### Fitting of straight line in the region 
                    popt, pcov = curve_fit(linear_fit, xplott, yplott)
                    
                    ### Variance of spectra
                    var = 0.
                    for ifit in range(0,len(xplott)):
                        var += (yplott[ifit]-linear_fit(xplott[ifit],\
                                popt[0],popt[1]))*\
                                (yplott[ifit]-linear_fit(xplott[ifit],\
                                popt[0],popt[1]))
                    var = var/float(len(xplott))



                    ### Returning an array of velocities [km/s]: 'xplot' 
                    ### and an array with the normalized flux: 'yplot'
                    xplot,yplot = spt.lineProf(xlp, ylp, lbc, hwidth=hwidth)
                    ### Equivalent width [Angstroms]
                    ew_ = spt.EWcalc(xplot, yplot, vw=hwidth)
                    ew = ew_*lbc/phc.c.cgs*1e9
                    ### Error of EW [Angstroms]
                    err_ew = 2.*hwidth*lbc/phc.c.cgs*1e9*2.*np.sqrt(var)
                    
                    files_contents[-1][3] = (ew,err_ew)
            
            
                    
                    ### Try to calculate the peak separation in [km/s]
                    try:
                        v1,v2 = spt.PScalc(xplot, yplot, vc=0.0, ssize=0.05, \
                                        gaussfit=True)
                    except:
                        v1 = np.nan; v2 = np.nan
                    
                    files_contents[-1][4] = (v2-v1,np.sqrt(var)*abs(v2-v1))
        
                    
                    
                    ### Trial of calculating the FWHM: A gaussian is ajusted to 
                    ### the absolute value of the line profile. The FWHM of this 
                    ### gaussian is extracted [km/s]. 
                    ### Also, the area [km/s] is extracted.
                    try:
                        popt, pcov = curve_fit(gaussian_fit, xplot, abs(yplot-1.),\
                                p0=[1000.,0.])
                        fwhm = np.sqrt(8.*popt[0]*np.log(2))
                        area = popt[1]
                        #plt.plot(xplot,abs(yplot-1.),color="black")
                        #plt.plot(xplot,gaussian_fit(xplot,popt[0],popt[1]),\
                        #        color="red")
                        #plt.show()
                    except:
                        fwhm = np.nan
                        area = np.nan
                    
                    files_contents[-1][5] = [(fwhm,np.sqrt(var)*abs(fwhm)),\
                            (area,np.sqrt(var)*abs(area))]
                            
                    ### If the modulus of the EW is unrealistically big, 
                    ### remove the element.
                    if abs(files_contents[-1][3][0]) > 100.:
                        del files_contents[-1]
                        

    Madefit = False
    Madefit0 = False
    ### Turn this on to make the linear fit of the region near the 
    ### L-band measurements.
    if 1==1:


        ### time of L-band measurements
        xLband = jdu.date_to_jd(int(dateLband[0]),\
                int(dateLband[1]),int(dateLband[2]))
        ### time of WISE measurements
        xWISE = jdu.date_to_jd(int(dateWISE[0]),int(dateWISE[1]),\
                int(dateWISE[2]))
        
        y0 = 8.5
        
        before_time = 2.5
        near_time = 0.6
        after_time = 0.4
        
        xtime = []
        yEW = []
        yerrEW = []
        ### 
        for ielem in range(0,len(files_contents)):
            if xLband-before_time*365. <= files_contents[ielem][1] <= \
                    xLband+after_time*365.:
                xtime.append(files_contents[ielem][1])
                yEW.append(files_contents[ielem][3][0])
                yerrEW.append(files_contents[ielem][3][1])
        
        ### 
        found_elem = False
        if len(xtime) >= 3:
            found_beyond = False
            for iee in range(0,len(xtime)):
                if xLband < xtime[iee] <= xLband+after_time*365.:
                    found_beyond = True
            found_near = False
            for iee in range(0,len(xtime)):
                if xLband-near_time*365. < xtime[iee] <= xLband:
                    found_near = True                

            
            if found_beyond or found_near:
                found_elem = True
        
        
        if found_elem:
            try:
                ### Fitting of straight line in the region 
                popt, pcov = curve_fit(linear_fit, xtime, yEW)
                A = popt[0]
                B = popt[1]
                
                if B/(linear_fit(xtime[0],A,B)-y0) > 0.0002:
                    variation_type = "BU"
                if B/(linear_fit(xtime[0],A,B)-y0) < 0.0002:
                    variation_type = "DISS"
                if abs(B/(linear_fit(xtime[0],A,B)-y0)) <= 0.0002:
                    variation_type = "STD"
                                    
                Madefit = True
            except:
                variation_type = "???"
                A = np.nan
                B = np.nan
        else:
            variation_type = "???"
            A = np.nan
            B = np.nan
            
            
            
            
        xtime0 = []
        yEW0 = []
        yerrEW0 = []
        ### 
        for ielem in range(0,len(files_contents)):
            if xWISE-before_time*365. <= files_contents[ielem][1] \
                    <= xWISE+after_time*365.:
                xtime0.append(files_contents[ielem][1])
                yEW0.append(files_contents[ielem][3][0])
                yerrEW0.append(files_contents[ielem][3][1])
                
        found_elem = False
        if len(xtime0) >= 3:
            found_beyond = False
            for iee in range(0,len(xtime0)):
                if xWISE < xtime0[iee] <= xWISE+after_time*365.:
                    found_beyond = True
            found_near = False
            for iee in range(0,len(xtime0)):
                if xWISE-near_time*365. < xtime0[iee] <= xWISE:
                    found_near = True                

            
            if found_beyond or found_near:
                found_elem = True  
        
        if found_elem:
            try:
                ### Fitting of straight line in the region 
                popt, pcov = curve_fit(linear_fit, xtime0, yEW0)
                A0 = popt[0]
                B0 = popt[1]
                
                if B0/(linear_fit(xtime0[0],A0,B0)-y0) > 0.0002:
                    variation_type0 = "BU"
                if B0/(linear_fit(xtime0[0],A0,B0)-y0) < 0.0002:
                    variation_type0 = "DISS"
                if abs(B0/(linear_fit(xtime0[0],A0,B0)-y0)) <= 0.0002:
                    variation_type0 = "STD"
                                    
                Madefit0 = True
            except:
                variation_type0 = "???"
                A0 = np.nan
                B0 = np.nan
        else:
            variation_type0 = "???"
            A0 = np.nan
            B0 = np.nan
            
        
                
        
        


                        
                        
    ### Turn this on to plot the light curves
    if 1==1:

        ### 
        yearlabel = ["2006","2008","2010","2012","2014","2016","2018",\
                "2020"]
        yearpos = [jdu.date_to_jd(int(yearlabel[iy]),1,1) \
                for iy in range(0,len(yearlabel))]

        colors = ["red","orange","green","blue","purple"]


        ### 
        plt.figure(figsize=(11,4))
        
        
        ### Plate containing EW light curve
        plt.subplot(211)
        
        ### time
        xlc = [files_contents[ifile][1] for ifile \
                in range(0,len(files_contents))]
        ### EW [Anstroms]
        ylc = [files_contents[ifile][3][0] for ifile \
                in range(0,len(files_contents))]
        ### error EW [Anstroms]
        errylc = [files_contents[ifile][3][1] for ifile \
                in range(0,len(files_contents))]
        
        ### time of L-band measurements
        xLband = jdu.date_to_jd(int(dateLband[0]),\
                int(dateLband[1]),int(dateLband[2]))
        ### time of WISE measurements
        xWISE = jdu.date_to_jd(int(dateWISE[0]),int(dateWISE[1]),\
                int(dateWISE[2]))
        
        ### 
        ylims = [np.nanmax(ylc)+0.2*(np.nanmax(ylc)-np.nanmin(ylc)),
                np.nanmin(ylc)-0.2*(np.nanmax(ylc)-np.nanmin(ylc))]
        
        ### Plotting L-band and WISE times
        plt.plot([xLband,xLband],ylims,color = "black",linewidth=2.)
        plt.plot([xWISE,xWISE],ylims,color = "black",linewidth=2.)
        
        ### Plotting EW light curve
        plt.plot(xlc,ylc,color = "black", linewidth = 0.5)
        for ilc in range(0,len(xlc)):
            plt.errorbar(xlc[ilc],ylc[ilc],yerr = errylc[ilc],\
            color = colors[ilc%len(colors)], linewidth = 0.5)
        
        if Madefit:
            if variation_type == "BU":
                colore = "purple"
            if variation_type == "STD":
                colore = "green"
            if variation_type == "DISS":
                colore = "red"
            plt.plot(xtime,[linear_fit(x,A,B) for x in xtime],\
                    color=colore,linewidth=2.)

        if Madefit0:
            if variation_type0 == "BU":
                colore = "purple"
            if variation_type0 == "STD":
                colore = "green"
            if variation_type0 == "DISS":
                colore = "red"
            plt.plot(xtime0,[linear_fit(x,A0,B0) for x in xtime0],\
                    color=colore,linewidth=2.)

        ### 
        titleadd = ""
        if Madefit0:
            titleadd += "      "+\
                "$\\tau_0$ = "+\
                str(lrr.round_sig((linear_fit(xtime0[0],A0,B0)-y0)/B0,4))+" [days]"
        if Madefit:
            titleadd += "      "+\
                "$\\tau_1$ = "+\
                str(lrr.round_sig((linear_fit(xtime[0],A,B)-y0)/B,4))+" [days]"
        ### 
        plt.xticks(yearpos,yearlabel)
        plt.xlim([np.nanmin(yearpos),np.nanmax(yearpos)])
        plt.ylabel("$EW\,[\mathrm{A}]$")
        plt.ylim(ylims)
        idx_fd = folders_data.index(folder_data)
        plt.title(folder_data.replace("/","").replace("HD","HD ")+\
                " ("+Starnames[idx_fd]+")"+titleadd)
    


        ### Plate containing the evolution of Halpha line profiles
        plt.subplot(212)
        
        xplots = []
        yplots = []
        
        ### Loop over spectra
        for ielem in range(0,len(files_contents)):
            
        
            hwidth = 1000.
            array_data = files_contents[ielem][2]
    
            xlp=array_data[:,0]*1e-4 ### lambda [microns]
            ylp=array_data[:,1]      ### BeSS's flux [unit = ???]        
        
            ### Returning an array of velocities [km/s]: 'xplot' 
            ### and an array with the normalized flux: 'yplot'
            xplot,yplot = spt.lineProf(xlp, ylp, lbc, hwidth=hwidth)        
        
            xplots.append(xplot)
            yplots.append(yplot)
        
        ### A factor to make all spectra of similar size
        fac = np.nanmax(np.array([np.nanmax(yplots[i]) \
                for i in range(0,len(yplots))])-1)-\
                np.nanmin(np.array([np.nanmin(yplots[i]) \
                for i in range(0,len(yplots))])-1)
        
        ### Loop over spectra
        for ielem in range(0,len(files_contents)):
            ### Julian date
            JD = files_contents[ielem][1]
        
            ### Plotting the spectrum
            plt.plot((yplots[ielem]-1.)*400./fac+JD,xplots[ielem],\
            linewidth = 0.5, color=colors[ielem%len(colors)])
        
        ### Bar with size one
        JD = jdu.date_to_jd(2019,10,1)
        plt.plot([JD,JD-1.*400./fac],[800.,800.],color="black",linewidth=3.)

        
        ### 
        plt.xticks(yearpos,yearlabel)
        plt.xlim([np.nanmin(yearpos),np.nanmax(yearpos)])
        plt.ylabel("vel. [km/s]")
        plt.ylim([-hwidth,hwidth])        
        ### 
        plt.tight_layout()
        plt.savefig(folder_data.replace("/","")+".png")
        
        plt.close()



    ### Turn this on to produce the output files
    if 1==1:
    
    
        ### 
        times = [files_contents[idate][1] for idate \
                in range(0,len(files_contents))]
                
        EWs = [files_contents[idate][3][0] for idate \
                in range(0,len(files_contents))]
        errEWs = [files_contents[idate][3][1] for idate \
                in range(0,len(files_contents))]        

        PSs = [files_contents[idate][4][0] for idate \
                in range(0,len(files_contents))]
        errPSs = [files_contents[idate][4][1] for idate \
                in range(0,len(files_contents))]                

        FWHMs = [files_contents[idate][5][0][0] for idate \
                in range(0,len(files_contents))]
        errFWHMs = [files_contents[idate][5][0][1] for idate \
                in range(0,len(files_contents))]                        
        

        xLband = jdu.date_to_jd(int(dateLband[0]),\
                int(dateLband[1]),int(dateLband[2]))
        
        xWISE = jdu.date_to_jd(int(dateWISE[0]),int(dateWISE[1]),\
                int(dateWISE[2]))
    
    
        EW_tLb = lrr.interpLinND([xLband],[times],EWs,tp="linear")
        errEW_tLb = lrr.interpLinND([xLband],[times],errEWs,tp="linear")
        EW_tWISE = lrr.interpLinND([xWISE],[times],EWs,tp="linear")
        errEW_tWISE = lrr.interpLinND([xWISE],[times],errEWs,tp="linear")

        PS_tLb = lrr.interpLinND([xLband],[times],EWs,tp="linear")
        errPS_tLb = lrr.interpLinND([xLband],[times],errEWs,tp="linear")
        PS_tWISE = lrr.interpLinND([xWISE],[times],EWs,tp="linear")
        errPS_tWISE = lrr.interpLinND([xWISE],[times],errEWs,tp="linear")
        
        FWHM_tLb = lrr.interpLinND([xLband],[times],EWs,tp="linear")
        errFWHM_tLb = lrr.interpLinND([xLband],[times],errEWs,tp="linear")
        FWHM_tWISE = lrr.interpLinND([xWISE],[times],EWs,tp="linear")
        errFWHM_tWISE = lrr.interpLinND([xWISE],[times],errEWs,tp="linear")
        
        
        
        
        f0 = open(folder_data.replace("/","")+".dat","w")
        f0.write("### [0] times [JD]\n")
        f0.write("### [1] EW [Angstrom]\n")
        f0.write("### [2] errEW [Angstrom]\n")
        f0.write("### [3] Peak separation [km/s]\n")
        f0.write("### [4] error Peak separation [km/s]\n")
        f0.write("### [5] FWHM [km/s]\n")
        f0.write("### [6] error FWHM [km/s]\n")
        for idate in range(0,len(times)):
            f0.write(str(times[idate])+" "+\
                    str(EWs[idate])+" "+\
                    str(errEWs[idate])+" "+\
                    str(PSs[idate])+" "+\
                    str(errPSs[idate])+" "+\
                    str(FWHMs[idate])+" "+\
                    str(errFWHMs[idate])+"\n")
        f0.write("##### time in L-band [JD] ; EW [A] ; errEW [A] ; PS [km/s] ; errPS [km/s] ; FWHM [km/s] ; errFWHM [km/s]  \n")
        f0.write(str(xWISE)+" "+str(EW_tWISE)+" "+str(errEW_tWISE)+" "+\
                str(PS_tWISE)+" "+str(errPS_tWISE)+" "+str(FWHM_tWISE)+" "+str(errFWHM_tWISE)+"\n")
        f0.write(str(xLband)+" "+str(EW_tLb)+" "+str(errEW_tLb)+" "+\
                str(PS_tLb)+" "+str(errPS_tLb)+" "+str(FWHM_tLb)+" "+str(errFWHM_tLb)+"\n")
        f0.write("####### timescale of variation [days] ; variation label \n")
        if len(xtime0) > 0:
            f0.write(str((linear_fit(xtime0[0],A0,B0)-y0)/B0)+" "+variation_type0+"\n")
        else:
            f0.write(str(np.nan)+" "+variation_type0+"\n")
        if len(xtime) > 0:
            f0.write(str((linear_fit(xtime[0],A,B)-y0)/B)+" "+variation_type+"\n")
        else:
            f0.write(str(np.nan)+" "+variation_type+"\n")
        f0.close()
    
    



        f1 = open(folder_data.replace("/","")+"_LC.dat","w")
        
        for idate in range(0,len(times)):
            ### 
            f1.write(str(times[idate])+"\n")
            ### 
            lineswrite = []
            [lineswrite.append(el) for el in files_contents[idate][2][:,0]]
            [f1.write(str(el)+" ") for el in lineswrite]
            f1.write("\n")
            ### 
            lineswrite = []
            [lineswrite.append(el) for el in files_contents[idate][2][:,1]]
            [f1.write(str(el)+" ") for el in lineswrite]
            f1.write("\n")
        
        f1.close()
 
 
 
 
 
 
        
    
    
    
    
    

