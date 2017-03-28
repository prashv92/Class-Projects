import randomstate
import numpy as np
from scipy.stats import chi2
from scipy.stats import norm
import math


#Random Number Generator 1
def ran1(n):
	Z0 = 0.84611665575369589265
	lamb = 1 - (10**(-11))
	rnp = []

	for i in range (n):
		if i+1 == 1:
			rnp.append(2*lamb*(0.5 - abs(Z0 - 0.5)))
		
		else:
			rnp.append(2*lamb*(0.5 - abs(rnp[i-1] - 0.5)))
	
	r = np.array(rnp)
	return r

#Implementation of Chi-Squared Test
def chisq(r,n,k):
	dof = k-1
	a = np.sort(r)
	Nj = []
	chisum = 0.0
	for i in range (k):
		lowerBoundK = (i-1)/k
		upperBoundK = i/k
		count = 0.0
		for j in range (n):
			if a[j]>=lowerBoundK and a[j]<upperBoundK:
				count=count + 1

		chisum = chisum + ((count - n/k)**2)

	chi = (k/n)*chisum
	thchi2 = chi2.ppf(0.95,dof)
	print("chi", chi)
	print("thchi", thchi2)
	print("dof",k)

#Implementation of Kolmogorov-Smirnov Test
def ksTest(r,n):
	a = np.sort(r)
	maxDnplus = 0
	maxDnminus = 0
	for i in range (n):
		dnplus = (i/n) - a[i]
		if dnplus > maxDnplus:
			maxDnplus = dnplus
		dnminus = a[i] - (i-1)/n
		if dnminus > maxDnminus:
			maxDnminus = dnminus
	dn = max(maxDnplus, maxDnminus)

	c05 = 1.358
	stat = (math.sqrt(n) + 0.12 + 0.11/math.sqrt(n))*dn
	print("ksStat",stat)
	print("c05",c05)
	if stat > c05:
		return 1
	else:
		return 0

#Implementation of Runs Test
def runsTest(r,n):
	runLength = 1
	runDict = {1:0,2:0,3:0,4:0,5:0,6:0}
	for i in range(n):
		if(i+1<n and r[i] < r[i+1]):
			runLength = runLength + 1
		else:
			if(runLength>=6):
				runDict[6] = runDict[6] + 1
			else:
				runDict[runLength] = runDict[runLength] + 1
			runLength = 1

	A = [(4529.4,9044.9,13568,18091,22615,27892),(9044.9,18097,27139,36187,45234,55789),(13568,27139,40721,54281,67852,83685),(18091,36187,54281,72414,90470,111580),(22615,45234,67852,90470,113262,139476),(27892,55789,83685,111580,139476,172860)]

	bVector = [1/6,5/24,11/120,19/720,29/540,1/840]
	rn = 0   
	for i in range (6):
		for j in range(6):
			rn = rn + A[i][j]*(runDict[j+1] - n*bVector[j])*(runDict[i+1] - n*bVector[i])

	rn = rn/n
	thchisq = chi2.ppf(0.95,6)
	print("rn",rn)
	print("chisq",thchisq)

	if rn > thchisq: return 0
	else: return 1

#Implementation of Autocorrelation Test
def autoCorr(r,n,j):
	low = math.floor((n-1)/j)
	h = low - 1
	rhoJ = 0
	for i in range(h):
		if(i+1<h and i+2<h):
			rhoJ = rhoJ + r[1+i*j]*r[1+(i+1)*j]
	rhoJ = (rhoJ*12/(h+1)) - 3
	print("rhoJ",rhoJ)
	varRhoJ = (13*h + 7)/((h+1))**2
	print("varRhoJ",varRhoJ)
	aj = rhoJ/math.sqrt(varRhoJ)
	zValue = norm.ppf(.975)
	print("aj",aj)
	print("zVale", zValue)
	if(aj>=-zValue and aj<=zValue): return 1
	else: return 0

#Implementation of Serial Test
def serialTest(r,n,k,d):
	r = r*k
	ui = np.split(r,n)
	ui=np.floor(ui)
	s = k**d

#The following code is the logic for binning:
#	1. I've multiplied the uniform stream by k to scale the data to 0-k instead of creating 
#	   a hypercube between 0-1 with 1/k coordinates
#	2. I've then split the stream into n Uis of length d
#	3. Then, I've floored the value of each Ui such that the values stored in the Ui's
#	   reflect the coordinates of the bin into which they fall
#	4. This is followed by creating a matrix Nj which contains the number of Ui's that fall
#	   into each N(j1,j2,j3,...,jd)
#	5. Then, I proceed to calculate the chi-squared statistic and comparing it to the 
#	   true chi-squared statistic
	if d==2:
		Nj = np.zeros((k,k))
		for i in range(n):
			l = int(ui[i][0])
			m = int(ui[i][1])
			Nj[l][m] += 1

	
	if d==3:
		Nj = np.zeros((k,k,k))
		for i in range(n):
			l = int(ui[i][0])
			m = int(ui[i][1])
			n = int(ui[i][2])
			Nj[l][m][n] += 1

	Nj = (Nj - (n/s))**2
	chisq = (s/n)*np.sum(Nj)
	thchisq = chi2.ppf(0.95,s-1)
	print("chiSq",chisq)
	print("thchiSq",thchisq)
	if chisq > thchisq:
		return 1
	else :
		return 0 


rs = randomstate.prng.mrg32k3a.RandomState(12345)
#Chi-square test
num = 2**15
k = 2**12
seq = np.array(rs.random_sample(num))
s = ran1(num)
chisq(seq,num,k)
chisq(s,num,k)

#Kolmogorov-Smirnov test
testSeq = ksTest(seq,num)
testS = ksTest(s,num)

#Serial Test
d1 = 2
d2 = 3
k1 =  64
k2 = 16

#Generating the random numbers for Serial Test for k=64, d=2
serialSeq1 = np.array(rs.random_sample(num*d1))
serialS1 = ran1(num*d1)

#Generating the random numbers for Serial Test for k=16, d=3
serialSeq2 = np.array(rs.random_sample(num*d2))
serialS2 = ran1(num*d2)

serialTSeq1 = serialTest(serialSeq1,num,k1,d1)
serialTS1 = serialTest(serialS1,num,k1,d1)

serialTSeq2 = serialTest(serialSeq2,num,k2,d2)
serialTS1 = serialTest(serialS2,num,k2,d2)

#Runs Test
runsN = 5000
runsSeq = np.array(rs.random_sample(runsN))
runsS = ran1(runsN)
runsTSeq = runsTest(runsSeq,runsN)
runsTS = runsTest(runsS, runsN)

#Autocorrelation test for the Generator 1
autoTestSeq1 = autoCorr(runsSeq,runsN,1)
autoTestSeq2 = autoCorr(runsSeq,runsN,2)
autoTestSeq3 = autoCorr(runsSeq,runsN,3)
autoTestSeq4 = autoCorr(runsSeq,runsN,4)
autoTestSeq5 = autoCorr(runsSeq,runsN,5)
autoTestSeq6 = autoCorr(runsSeq,runsN,6)

#Autocorrelation test for MRG32k3a
autoTestS1 = autoCorr(runsS,runsN,1)
autoTestS2 = autoCorr(runsS,runsN,2)
autoTestS3 = autoCorr(runsS,runsN,3)
autoTestS4 = autoCorr(runsS,runsN,4)
autoTestS5 = autoCorr(runsS,runsN,5)
autoTestS6 = autoCorr(runsS,runsN,6)