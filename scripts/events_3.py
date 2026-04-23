import numpy as np
import matplotlib.pyplot as plt
import scienceplots
from scipy.optimize import curve_fit
import scipy.integrate as integrate
from scipy.optimize import brentq

plt.style.use(['science'])

# Constants
fmtoMeV = (1/197.3)
umatoMeV = 931.5
Ar_m = 39.95*umatoMeV # uma -> MeV
G_f = 1.1663787E-11 # MeV-2
weibergAngle = 0.22337
flux = 3.89E-10 # 10^12 cm^-2 s^-1 in MeV^2 / s
Mass_W = 80.379 # GeV/c^2
#ncr_SM = -2.125E-5 # GeV^(-2) 0.83e-32 cm^2 
ncr_SM = -1.82206*(1.1663787E-5)*(100/5.07E15)**2 # 0.83E-33 cm^-2

#ncr_SM = 0 

GeVm2tocm2 = (100/5.07E15)**2



# CEvNS Diff Cross Section




def diffCrossdep(N,Z,E,T,M,sin2thetaw,ncr):
    g_vp = (1/2-2*(sin2thetaw*(1+(1/3)*(Mass_W**2)*(ncr/GeVm2tocm2))))
    g_vn = -1/2
    return ((((G_f)**2) *M)/(np.pi))*((g_vn*N-g_vp*Z)**2)*(1-(M*T)/(2*(E**2)))


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

Z = 14
N = 40 - 14

E_max = 9.5 #MeV

# QUENCHING FACTOR

k_AR   = 0.1333*(18**(2/3))*(40**(-1/2)) #k parameter argon


def epsilon(T): # T in keV
    return 11.5 * Z**(-7/3) * T

def g_lindhard(eps):
    return 3 * eps**0.15 + 0.7 * eps**0.6 + eps

def QF(T):
    return (k_AR*(g_lindhard(epsilon(T))))/(1 + k_AR*(g_lindhard(epsilon(T))))

# Range for root search
ER_MIN = (1e-4) 
ER_MAX = (6)   

def ERfromEI(E_I):
    func = lambda E_R: QF(E_R) * E_R - E_I
    return brentq(func, ER_MIN,ER_MAX)




def fluxdep(T, sin2thetaw,ncr):
      E_min =  (T/2)*(1+np.sqrt(1+(2*Ar_m/T))) 
      return integrate.quad(lambda E: diffCrossdep(N,Z,E,T,Ar_m,sin2thetaw,ncr)*SpecTot(E), E_min, E_max)[0]



times = {"100 días": 8.64E6,
         "200 días": 2*(8.64E6)} # s

masses = {"20 kg": 3.015E26, 
          "100 kg": 5*3.015E26,
          "200 kg": 3.015E27} #MeV

exposure = "200 días"

chiSqR = {}
minAngle = {}
lim1 = {}
lim2 = {}


plot, ax = plt.subplots(1,1,figsize=(5,5))

T_min = ERfromEI(0.1)/1000 # 100 eVee threshold to eVnr (in GeV) ~ 400 eVnr
T_max = (2*E_max**2)/(Ar_m-2*E_max) # ~5 keV

print("Max recoil energy:", T_max*1000, "keV")
print("Threshold recoil energy:", T_min*1000, "keV")

zoom = ncr_SM*0.2
for mass in masses:

    total_targets = times[exposure]*masses[mass]*flux

    Ntheo = total_targets * integrate.quad(lambda t: fluxdep(t, weibergAngle,ncr_SM), T_min ,T_max)[0]
    
    print(f"##### EXPECTED NUMBER OF EVENTS for {mass}",Ntheo,"#######")
    print("##### EVENTS PER DAY:",Ntheo/(times[exposure]/86400),"#######")

    angles = np.linspace(ncr_SM-zoom,ncr_SM+zoom,500)
    Nexp = []

    for i in angles:
        Nexp.append(total_targets*integrate.quad(lambda t: fluxdep(t, weibergAngle,i), T_min,T_max)[0])
    
    #print(Nexp[50])
    chiSq = []

    for i in Nexp:
        chiSq.append(((Ntheo-i)**2)/(Ntheo + (0*i)**2))
    
    #print(chiSq)
    chiSqR[mass] = chiSq

    minAngle[mass] = angles[chiSq.index(min(chiSq))]
    print(angles[chiSq.index(min(chiSq))])

    closesttoone = min(chiSq, key= lambda x: abs(x-1))
    index = chiSq.index(closesttoone)
    lim1[mass] = angles[index]

    chiSq.pop(chiSq.index(closesttoone))

    closesttoone2 = min(chiSq, key=lambda x:abs(x-1))
    lim2[mass] = angles[chiSq.index(closesttoone2)]

    chiSq.insert(index, closesttoone)

ax.set_ylabel(r"$\Delta \chi^2$")
ax.set_xlabel(r"$\left< r^{2}_{\nu_e} \right> ~ [\text{cm}^{2}] $")

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

    #ax.plot(angles, chiSqR[key], label = key)
    if lim1[key] > lim2[key]:   
        ax.plot(angles, chiSqR[key], label = key + r"; $\left< r^{2}_{\nu_e} \right> = "+f"{minAngle[key]*1E33:.3f}"+r"^{+"+f"{(lim1[key] - minAngle[key])*1E33:.3f}"
                           +r"}_{"+f"-{(minAngle[key] - lim2[key])*1E33:.3f}"+r"}\times 10^{-33} \text{cm}^{2}$")
    else:
        ax.plot(angles, chiSqR[key], label = key + r"; $\left< r^{2}_{\nu_e} \right> = "+f"{minAngle[key]*1E33:.3f}"+r"^{+"+f"{(lim2[key] - minAngle[key])*1E33:.3f}"
                           +r"}_{"+f"-{(minAngle[key] - lim1[key])*1E33:.3f}"+r"}\times 10^{-33} \text{cm}^{2}$")

plot.suptitle(r"Diferencia estadística $\chi^2$ para diferenetes radios de carga $\left< r^{2}_{\nu_e} \right>$ en CE$\nu$NS de neutrinos "+ "\n"+ r" provenientes de reactores nucleares interactuando con diferentes massas de $^{40}$Ar "+f"por {exposure}")

ax.legend()
plt.savefig(f"/home/nullpost/Scripts/college stuf/CEvNS sequel/Figure_phi_{exposure[:3]}.png",dpi=200)


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