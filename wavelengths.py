import pyhdust.spectools as spt

print("Halpha [Angs] = "+str(spt.hydrogenlinewl(3., 2.)*1e10))
print("")
print("Bralpha [Angs] = "+str(spt.hydrogenlinewl(5., 4.)*1e10))
print("Humphrey14 [Angs] = "+str(spt.hydrogenlinewl(14., 6.)*1e10))
print("Pfgamma [Angs] = "+str(spt.hydrogenlinewl(8., 5.)*1e10))
print("")
print("Humphrey15 [Angs] = "+str(spt.hydrogenlinewl(15., 6.)*1e10))
print("Humphrey16 [Angs] = "+str(spt.hydrogenlinewl(16., 6.)*1e10))
print("Humphrey17 [Angs] = "+str(spt.hydrogenlinewl(17., 6.)*1e10))
print("Humphrey18 [Angs] = "+str(spt.hydrogenlinewl(18., 6.)*1e10))
print("Humphrey19 [Angs] = "+str(spt.hydrogenlinewl(19., 6.)*1e10))
print("Humphrey20 [Angs] = "+str(spt.hydrogenlinewl(20., 6.)*1e10))
print("Humphrey21 [Angs] = "+str(spt.hydrogenlinewl(21., 6.)*1e10))
print("Humphrey22 [Angs] = "+str(spt.hydrogenlinewl(22., 6.)*1e10))
print("Humphrey23 [Angs] = "+str(spt.hydrogenlinewl(23., 6.)*1e10))
print("Humphrey24 [Angs] = "+str(spt.hydrogenlinewl(24., 6.)*1e10))
print("Humphrey25 [Angs] = "+str(spt.hydrogenlinewl(25., 6.)*1e10))




N=100
selectedlines=[]
selectedlines2=[]
lambs=[]
ji=[]
for i in range(1,N):
    for j in range(i+1,N+1):
        lamb = spt.hydrogenlinewl(j, i)*1e6
        if 3.2 <= lamb <= 4.1:
            selectedlines.append([j,i,lamb])
            lambs.append(lamb)
            ji.append([j,i])
for ii in range(0,len(selectedlines)):
    print(selectedlines[ii])

print(lambs)
print(sorted(lambs))


