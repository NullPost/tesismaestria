import numpy as np
import matplotlib.pyplot as plt
import scienceplots
from scipy.optimize import curve_fit
import scipy.integrate as integrate

plt.style.use(['science'])

# Constants
fmtoMeV = (1/197.3)
umatoMeV = 931.5
Ar_m = 39.95*umatoMeV # uma -> MeV
G_f = 1.1663787E-11 # MeV-2
weibergAngle = 0.22337
flux = 3.89E-10 # 10^12 cm^-2 s^-1 in MeV^2 / s

# CEvNS Diff Cross Section




def diffCrossdep(N,Z,E,T,M,sin2thetaw):
    return ((G_f)**2/(2*np.pi))*((N-Z*(1-4*(sin2thetaw)))**2)*M*(1-(M*T)/(2*(E**2)))


# Reactor Spectra (everything seems to point to use MeV)

alpha = {
    '235U' : [3.217, -3.111,  1.395,-0.369,   0.04445, -0.002053 ],
    '238U' : [0.4833,0.1927,-0.1283,-0.006762,0.002233,-0.0001536],
    '239Pu': [6.413, -7.432,3.535,  -0.882,   0.1025,  -0.00455  ],
    '241Pu': [3.251, -3.204, 1.428, -0.3675,  0.04254, -0.001896 ]
}

fractions = {    
    '235U' : 0.55,
    '238U' : 0.07,
    '239Pu': 0.32,
    '241Pu': 0.06
}

# fractions = {    
#     '235U' : 0,
#     '238U' : 1,
#     '239Pu': 0,
#     '241Pu': 0
# }



def SpecTot(E): # Units MeV-1
    return fractions['235U']*(np.exp(alpha['235U'][0] 
                                   + alpha['235U'][1]*E 
                                   + alpha['235U'][2]*(E**2) 
                                   + alpha['235U'][3]*(E**3) 
                                   + alpha['235U'][4]*(E**4) 
                                   + alpha['235U'][5]*(E**5))) \
         + fractions['238U']*(np.exp(alpha['238U'][0] 
                                   + alpha['238U'][1]*E 
                                   + alpha['238U'][2]*(E**2) 
                                   + alpha['238U'][3]*(E**3) 
                                   + alpha['238U'][4]*(E**4) 
                                   + alpha['238U'][5]*(E**5))) \
         + fractions['239Pu']*(np.exp(alpha['239Pu'][0] 
                                   + alpha['239Pu'][1]*E 
                                   + alpha['239Pu'][2]*(E**2) 
                                   + alpha['239Pu'][3]*(E**3) 
                                   + alpha['239Pu'][4]*(E**4) 
                                   + alpha['239Pu'][5]*(E**5))) \
         + fractions['241Pu']*(np.exp(alpha['241Pu'][0] 
                                   + alpha['241Pu'][1]*E 
                                   + alpha['241Pu'][2]*(E**2) 
                                   + alpha['241Pu'][3]*(E**3) 
                                   + alpha['241Pu'][4]*(E**4) 
                                   + alpha['241Pu'][5]*(E**5)))
    

# 40-Argon

def fluxdep(T, sin2thetaw):        
      return integrate.quad(lambda E: diffCrossdep(N,Z,E,T,Ar_m,sin2thetaw)*SpecTot(E), np.sqrt(Ar_m * T /2), E_max)[0]

Z = 14
N = 40 - 14

E_max = 9.5 #MeV
Tmax = (2*E_max**2)/(Ar_m-E_max) # ~5 keV
print("Max recoil energy:", Tmax*1000, "keV")

times = {"100 días": 8.64E6,
         "200 días": 2*(8.64E6)} # s

masses = {"20 kg": 3.015E26, 
          "100 kg": 1.5E27,
          "200 kg": 3.015E27} #MeV


chiSqR = {}
minAngle = {}
lim1 = {}
lim2 = {}


plot, ax = plt.subplots(1,1,figsize=(5,5))


for mass in masses:

    total_targets = times["200 días"]*masses[mass]*flux

    Ntheo = total_targets * integrate.quad(lambda t: fluxdep(t, weibergAngle), 30*(1/1000)*(1/1000),Tmax)[0]
    
    print(Ntheo)

    angles = np.linspace(weibergAngle-0.0005,weibergAngle+0.0005,500)
    Nexp = []

    for i in angles:
        Nexp.append(total_targets*integrate.quad(lambda t: fluxdep(t, i), 30*(1/1000)*(1/1000),Tmax)[0])
    
    #print(Nexp[50])
    chiSq = []

    for i in Nexp:
        chiSq.append(((Ntheo-i)**2)/(Ntheo + (0*i)**2))
    
    print(chiSq)
    chiSqR[mass] = chiSq

    minAngle[mass] = angles[chiSq.index(min(chiSq))]

    closesttoone = min(chiSq, key= lambda x: abs(x-1))
    index = chiSq.index(closesttoone)
    lim1[mass] = angles[index]

    chiSq.pop(chiSq.index(closesttoone))

    closesttoone2 = min(chiSq, key=lambda x:abs(x-1))
    lim2[mass] = angles[chiSq.index(closesttoone2)]

    chiSq.insert(index, closesttoone)

ax.set_ylabel(r"$\Delta \chi^2$")
ax.set_xlabel(r"$\sin^2 \theta_W$")

ax.hlines(1.00, xmin=min(angles), xmax=max(angles), colors="black") #label=r"$1 \sigma$") 
ax.hlines(2.71 , xmin=min(angles), xmax=max(angles), colors="black") #label=r"$90 \% \text{C.L.}$") 
ax.hlines(4.00, xmin=min(angles), xmax=max(angles), colors="black") # label=r"$2 \sigma$") 
ax.hlines(6.63, xmin=min(angles), xmax=max(angles), colors="black") # label=r"$2 \sigma$") 

ax.set_xlim(min(angles),max(angles))
ax.set_ylim(0, 12)

ax_right = ax.twinx()
ax_right.set_ylim(ax.get_ylim())

ax_right.set_yticks([1, 2.71, 4,6.63])
ax_right.set_yticklabels([r"$1 \sigma$", r"$90 \%$C.L.", r"$2 \sigma$", r"$99 \%$ C.L."])  

for key in chiSqR.keys():

    if lim1[key] > lim2[key]:   
        ax.plot(angles, chiSqR[key], label = key + r"; $\sin^2 \theta_W = "+f"{minAngle[key]:.5f}"+r"^{+"+f"{lim1[key] - minAngle[key]:.5f}"
                           +r"}_{"+f"-{minAngle[key] - lim2[key]:.5f}"+r"}$")
    else:
        ax.plot(angles, chiSqR[key], label = key + r"; $\sin^2 \theta_W = "+f"{minAngle[key]:.5f}"+r"^{+"+f"{lim2[key] - minAngle[key]:.5f}"
                           +r"}_{"+f"-{minAngle[key] - lim1[key]:.5f}"+r"}$")

plot.suptitle(r"Diferencia estadística $\chi^2$ para diferenetes angulos de mezcla electrodébil $\sin^2 \theta_W$ en CE$\nu$NS de neutrinos "+ "\n"+ r" provenientes de reactores nucleares interactuando con diferentes massas de $^{40}$Ar por 200 días")

ax.legend()
plt.savefig("/home/nullpost/Scripts/college stuf/CEvNS sequel/Figure_sin_200.png",dpi=200)


# ax1.plot(angles, chiSq)


# ax1.set_ylabel(r"$\Delta \chi^2$")
# ax1.set_xlabel(r"$\sin^{2}(\theta_W)$")

# ax2.yaxis.tick_right()
# ax2.yaxis.set_label_position("right")
# ax2.set_ylabel(r"$\Delta \chi^2$", )
# ax2.set_xlabel(r"$\sin^{2}(\theta_W)$")

# plot.suptitle(r"Diferencia estadística $\chi^2$ para diferenetes ángulos de mezcla débil "+ "\n"+ r"en CE$\nu$NS de neutrinos provenientes de reactores nucleares interactuando con $^{40}$Ar")
# plot.tight_layout()

# minAngle = angles2[chiSq2.index(min(chiSq2))]

# closesttoone = min(chiSq2, key= lambda x: abs(x-1))
# index = chiSq2.index(closesttoone)
# lim1 = angles2[index]

# chiSq2.pop(chiSq2.index(closesttoone))

# closesttoone2 = min(chiSq2, key=lambda x:abs(x-1))
# lim2 = angles2[chiSq2.index(closesttoone2)]

# print(lim1, lim2)

# chiSq2.insert(index, closesttoone)
# def parabola(x,a):
#     return a*(x - minAngle)**2

# popt, pcov = curve_fit(parabola, angles, diffSquared)
# #print(diffSquared)
# ax2.plot(angles, parabola(angles, *popt), label=r"Parabola fit, $\sin^2 \theta_{W} = 0.22 \pm 0.03$")

# print(diffSquared)

# upperlim  = angles2[chiSq2.index(0.9463472606695976)]
# lowerlim = angles2[chiSq2.index(0.9807083243824412)]



# #print("weinberg angle:", minAngle, "+/-", np.sqrt(1/popt[0]))
# print("weinberg angle:", minAngle, "+", upperlim - minAngle, "-", minAngle - lowerlim )

# if lim1 > lim2:

    
#     ax2.plot(angles2, chiSq2, label = r"$\sin^2 \theta _ W = "+f"{minAngle:.3f}"+r"^{+"+f"{lim1 - minAngle:.3f}"
#                            +r"}_{"+f"-{minAngle - lim2:.3f}"+r"}$")
# else:
#     ax2.plot(angles2, chiSq2, label = r"$\sin^2 \theta _ W = "+f"{minAngle:.3f}"+r"^{+"+f"{lim2 - minAngle:.3f}"
#                            +r"}_{"+f"-{minAngle - lim1:.3f}"+r"}$")

# plt.legend()
# plt.savefig("/home/nullpost/Scripts/college stuf/CEvNS sequel/Figure_2.png")


# print(Tmax)
# rang = np.linspace(30*(1/1000)*(1/1000),Tmax,100)
# res = []

# for i in rang:
#     res.append(flux(i)[0])
#     #print(flux(i)[0])
# res = np.array(res)

# #res = np.array(res)

# 

# 
# plt.show()