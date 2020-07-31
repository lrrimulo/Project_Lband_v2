### EXAMPLE OF USING lrr.interpLinND
import numpy as np
import pyhdust.lrr as lrr
import itertools as it


lrr.makeitso() ### Just to mark the beginning of the run :)

### Defining the exact function to be used as example.
def function_ex1(x,y,z):
    ### A non-linear function.
    return x**2.*y**3.*z
    ### A linear function.
    #return 2.*x+3.*y+4.*z


### Creating a 3D domain:
x=[1.,2.]
y=[3.,4.,5.]
z=[6.,7.,8.,9.]

### Atributing values to every point in the grid.
values=[]
for i in range(0,len(x)):
    for j in range(0,len(y)):
        for k in range(0,len(z)):
            function_example=function_ex1(x[i],y[j],z[k])
            values.append(function_example)
            ### Checking values:
            if 1==2:
                print(np.array([i,j,k]),\
                    np.array([x[i],y[j],z[k]]),function_example)




### The list 'axis', which enters the routine:
axis=[x,y,z]


if 1==2:

    ### Several examples to be printed on screen:
    print("")
    point=np.array([1.6,4.8,8.8]) 
    print("EXAMPLE: Point in 3D space inside the grid.")
    print("Point: ",point)
    interp=lrr.interpLinND(point,axis,values)
    print("The interpolated value:",interp)
    interp=lrr.interpLinND(point,axis,values,allow_extrapolation="no")
    print("The interpolated value (not allowing extrapolation):",interp)
    f_example=function_ex1(point[0],point[1],point[2])
    print("The real value:",f_example)
    

    print("")
    point=np.array([1.0,3.0,6.0]) 
    print("EXAMPLE: Point in 3D space coinciding with one of the points of the grid.") 
    print("Point: ",point)
    interp=lrr.interpLinND(point,axis,values)
    print("The interpolated value:",interp)
    interp=lrr.interpLinND(point,axis,values,allow_extrapolation="no")
    print("The interpolated value (not allowing extrapolation):",interp)
    f_example=function_ex1(point[0],point[1],point[2])
    print("The real value:",f_example)
    

    print("")
    point=np.array([3.0,4.0,8.0]) 
    print("EXAMPLE: Point in 3D space that requires extrapolation.")
    print("Point: ",point)
    interp=lrr.interpLinND(point,axis,values)
    print("The interpolated value:",interp)
    interp=lrr.interpLinND(point,axis,values,allow_extrapolation="no")
    print("The interpolated value (not allowing extrapolation):",interp)
    f_example=function_ex1(point[0],point[1],point[2])
    print("The real value:",f_example)

    print("")
    point=np.array([0.0,0.0,0.0]) 
    print("EXAMPLE: Point in 3D space that requires extrapolation.")
    print("Point: ",point)
    interp=lrr.interpLinND(point,axis,values)
    print("The interpolated value:",interp)
    interp=lrr.interpLinND(point,axis,values,allow_extrapolation="no")
    print("The interpolated value (not allowing extrapolation):",interp)
    f_example=function_ex1(point[0],point[1],point[2])
    print("The real value:",f_example)


##################
### Other tests

if 1==2:

    print("\n\n")
    print("STUDYING low_high: \n")


    point=np.array([1.6,4.8,8.8]) 
    low,high,ind,extrapolated=lrr.low_high(point,axis)
    print("Point: ",point)
    print("For the domain defined by: ")
    print(axis)
    print("low,high,ind = ")
    print(low,high,ind)
    print("")


    point=np.array([1.0,3.0,6.0]) 
    low,high,ind,extrapolated=lrr.low_high(point,axis)
    print("Point: ",point)
    print("For the domain defined by: ")
    print(axis)
    print("low,high,ind = ")
    print(low,high,ind)
    print("")
        
    point=np.array([3.0,4.0,8.0]) 
    low,high,ind,extrapolated=lrr.low_high(point,axis)
    print("Point: ",point)
    print("For the domain defined by: ")
    print(axis)
    print("low,high,ind = ")
    print(low,high,ind)
    print("")

    point=np.array([0.0,0.0,0.0]) 
    low,high,ind,extrapolated=lrr.low_high(point,axis)
    print("Point: ",point)
    print("For the domain defined by: ")
    print(axis)
    print("low,high,ind = ")
    print(low,high,ind)
    print("")
    
    point=np.array([-1.0,-1.0,10.0]) 
    low,high,ind,extrapolated=lrr.low_high(point,axis)
    print("Point: ",point)
    print("For the domain defined by: ")
    print(axis)
    print("low,high,ind = ")
    print(low,high,ind)
    print("")
    
    
if 1==2:
    
    numbers=[0,1,2,3,5,9,11,50]
    for i in range(0,len(numbers)):
        bina=lrr.dec_2_binary(numbers[i])
        print(numbers[i],bina)
    
    
    
    
    
if 1==2:
    
    print("\n\n")
    print("STUDYING build_Fx: \n")


    point=np.array([1.6,4.8,8.8]) 
    low,high,ind,extrapolated=lrr.low_high(point,axis)
    print("Point: ",point)
    print("For the domain defined by: ")
    print(axis)
    print("low,high,ind = ")
    print(low,high,ind)
    print("")
    print(values)
    Fx=lrr.build_Fx(axis,values,ind)
    print(Fx)
    print("")
    
    














### Filling NaNs routine: 

if 1==1:

    ### Defining the exact function to be used as example.
    def function_EXa(x,y,z,t):
        ### A non-linear function.
        return x*x+y*y*y+z+t*t
        #### A linear function
        #return 3.*x+4.*y+z+3.*t

    def make_hole(pos,center,radius,htype):
        """
        Returns TRUE if the point is inside the hole.
        Returns FALSE if not. 
        """
        if htype == "none":
            return False
        
        if htype == "spherical":
            dist2 = 0.
            for i in range(0,len(pos)):
                dist2 += (pos[i]-center[i])*(pos[i]-center[i])
            if np.sqrt(dist2) <= radius:
                return True
            else:
                return False
                

    ### Creating a 4D domain:
    x=[4.,5.,6.,7.]
    y=[4.,5.,6.,7.]
    z=[5.,6.,7.]
    t=[4.,5.,6.]

    holetype = "spherical"
    holeradius = 1.2
    holecenter = [4.3,6.1,6.1,4.3]

    ### Atributing values to every point in the grid.
    values=[]
    values_ref = []
    for i in range(0,len(x)):
        for j in range(0,len(y)):
            for k in range(0,len(z)):
                for l in range(0,len(t)):
                    function_example=function_EXa(x[i],y[j],z[k],t[l])
                    values_ref.append(function_example)
                    if make_hole([x[i],y[j],z[k],t[l]],\
                            holecenter,holeradius,holetype):
                        values.append(np.nan)
                    else:
                        values.append(function_example)
                    ### Checking values:
                    if 1==2:
                        print(np.array([i,j,k,l]),\
                            np.array([x[i],y[j],z[k],t[l]]),function_example)



    ###########################

    point = [4.4,6.2,6.0,4.4]
    ### The list 'axis', which enters the routine:
    axis=[x,y,z,t]
    allow_extrapolation="no"
    tp = "arcsinh"
    prints = "yes"

    if 1==2:
        
        interp = lrr.interpLinNDpowerful(point,axis,values,tp,allow_extrapolation)

        print(interp)
        print(function_EXa(point[0],point[1],point[2],point[3]))



    if 1==1:
        
        new_values = lrr.fill_NaNs_interp(axis,values,tp,allow_extrapolation,prints)
        #new_values = lrr.fill_NaNs_interp(axis,values)
    
        for i in range(0,len(values)):
            print(values[i],new_values[i],values_ref[i])

        




    
    
    
