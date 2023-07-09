
import os
import sys
import importlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
### import casa tasks (but useless for this code actually...)
from taskinit import *

### import my modules
sys.path.append("/Users/nagashima/")
save_dir = 'paperfigure/' 

from mycasatest import hexbin_sampling_ex
from mycasa_sampling import hexbin_sampling
from imhead import imhead
from imval import imval
from scipy.optimize import curve_fit
import math


##　使用データ ##
image_Paa       = "NGC1068_11080_NIC3_Paalpha_flux-recentered-reproject-units.fits"
image_ff        = "ngc1068_b3_12m+7m_cont.image.tt0.pbcor.fits"
image_ffrms     = "ngc1068_b3_12m+7m_cont.noise.tt0.fits"
image_ffb6      = "ngc1068_b6_12m+7m_cont_pbcor_sm_testest.fits"
abr             = "ngc1068_magnum_muse_extinction.fits"
sync            = "ngc1068_vla_14p9_regrid.fits"

snr             = 3.
ra_deg          = 40.66963 # ngc1068 center in degree
dec_deg         = -0.01330 # ngc1068 center in degree
beamarea_100GHz = 70.5944  # number of pixel per area of the 100GHz map
cut_radius_kpc  = 0.8      # radius cut to focus on the outer part of ngc1068 (or ignore the center)
nu_ff  = 99.7015  

## サンプリング ##
ORIx_deg,ORIy_deg,ORIz_Paa = hexbin_sampling(image_Paa,ra_deg,dec_deg,units="other")
_,_,ORIz_abr         = hexbin_sampling(abr,ra_deg,dec_deg,units="K")
_,_,ORIz_ffrms       = hexbin_sampling(image_ffrms,ra_deg,dec_deg,err=True,units="other")
_,_,ORIz_ff          = hexbin_sampling(image_ff,ra_deg,dec_deg,units="other")
_,_,ORIz_ffb6          = hexbin_sampling(image_ffb6,ra_deg,dec_deg,units="other")
_,_,ORIz_sync          = hexbin_sampling(sync,ra_deg,dec_deg,units="other")



## 標準化 ##
ORIr_kpc = np.sqrt(ORIx_deg**2+ORIy_deg**2) * 72/1000.
ORIx_deg = ORIx_deg * 72./1000.
ORIy_deg = ORIy_deg * 72./1000.
### convert units of Paalpha hexmap from 10E^-16 erg/s/cm^2 to log10(erg/s/cm^2) [see FITS header]
ORIz_Paa = ORIz_Paa * 10**-16
ORIzlog_Paa = np.log10(ORIz_Paa)

### convert units of 100GHz hexmap from Jy/beam to log10(Jy)
ORIz_ff = ORIz_ff / beamarea_100GHz
ORIzlog_ff = np.log10(ORIz_ff)

ORIzerr_ffrms = ORIz_ffrms / beamarea_100GHz #not log
ORIzlog_ffrms = 1 / np.log(10) * ORIzerr_ffrms / ORIz_ff

ORIz_ffb6 = ORIz_ffb6 / beamarea_100GHz
ORIzlog_ffb6 = np.log10(ORIz_ffb6)

ORIz_sync = ORIz_sync / beamarea_100GHz
ORIzlog_sync = np.log10(ORIz_sync)


syncsigma =  7.766*10**-4  /beamarea_100GHz
ffb6sigma =  2.740*10**-4/beamarea_100GHz
Paasigma = 3.46*10**-17

#####################
##  figure1を作る   ##
#####################
snr = 3.

cut0_1       = ((ORIz_Paa >= snr*Paasigma)&(ORIr_kpc>=cut_radius_kpc))
cut0_2       = ((ORIr_kpc>=cut_radius_kpc)&(abs(ORIr_kpc)<1.5))
cut0_3       = (( ( (ORIx_deg-(-1.0))**2 + (ORIy_deg-(-0.5))**2 ) <1 ) &  (ORIr_kpc>=cut_radius_kpc)) 
cut0_4       = (( ( (ORIx_deg-(1.25))**2 + (ORIy_deg-(0.5))**2 ) <(0.75)**2 ) & (ORIr_kpc>=cut_radius_kpc)) 
cut0_5       = ( np.logical_or(cut0_1, cut0_2) )
cut0_6       = ( np.logical_or(cut0_3, cut0_4) )

cut0         = ( np.logical_or(cut0_5, cut0_6) )
cut1       = ((ORIr_kpc<cut_radius_kpc))
cut        = np.where( np.logical_or(cut0, cut1) )

x_deg      = ORIx_deg[cut]
y_deg      = ORIy_deg[cut]
r_kpc      = ORIr_kpc[cut]
z_ff       = ORIz_ff[cut]
zlog_ff    = ORIzlog_ff[cut]
zerr_ffrms = ORIzerr_ffrms[cut]

zlog_ffrms = ORIzlog_ffrms[cut]
ff_sigma_total=np.sqrt(zerr_ffrms**2 + (z_ff * 0.05)**2)
zlog_ff_sigma_total = 1 / np.log(10) * ff_sigma_total / z_ff




fig=plt.figure(figsize=(15,10))
gs=gridspec.GridSpec(nrows=10,ncols=10)
ax=plt.subplot(gs[0:10,0:10])
ax.set_aspect('equal', adjustable='box')
ax.invert_xaxis()
plt.rcParams["font.size"]=30
plt.rcParams["legend.fontsize"]=30
outfile         = "figure1_100GHzffc.png"
os.system("rm -rf "+os.path.join(save_dir, outfile))

ax.set_xlabel("x [kpc]")
ax.set_ylabel("y [kpc]")
ax.set_title(r"$NGC \, 1068 \, \, 100\, GHz \,free-free \, continuum\,  map$",y=1.05)
ax.set_xlim(2.0,-2.0)
ax.set_ylim(-2.0,2.0)
xlim_bg = [2.0, -2.0] # matplotlibのxlimと同じ値でok
ylim_bg = [-2.0, 2.0] # matplotlibのylimと同じ値でok

sc=ax.scatter(x_deg,y_deg,c=zlog_ff,edgecolor='None',marker="h",s=100)
cber=fig.colorbar(sc,extend="both")
c = patches.Circle(xy=(0, 0), radius=0.8, linewidth=8,fill=False, ec='white')
ax.add_patch(c)

cber.set_label(r"$log_{10}\, intensity \, \mathrm{[Jy]} $")
ax.grid()
plt.tick_params(labelsize=30)
save_dir = 'paperfigure/' 
plt.savefig(os.path.join(save_dir, outfile),dpi=200) 

#########culclate###################
interarea = np.where((r_kpc<0.8))
outarea  = np.where((r_kpc>0.8))
inff_totalflux = np.sum(z_ff[interarea])
outff_totalflux = np.sum(z_ff[outarea])
print("inff_totalflux:{}\n".format(inff_totalflux))
print("outff_totalflux:{}".format(outff_totalflux))
#####################################



###########################################################
fig1cut_b6  = ((~np.isnan(ORIzlog_ffb6))&(~np.isinf(ORIzlog_ffb6))&(ORIz_ffb6>ffb6sigma*snr))

cut0_1       = ((ORIz_Paa >= snr*Paasigma)&(ORIr_kpc>=cut_radius_kpc))
cut0_2       = ((ORIr_kpc>=cut_radius_kpc)&(abs(ORIr_kpc)<1.5))
cut0_3       = (( ( (ORIx_deg-(-1.0))**2 + (ORIy_deg-(-0.5))**2 ) <1 ) &  (ORIr_kpc>=cut_radius_kpc)) 
cut0_4       = (( ( (ORIx_deg-(1.25))**2 + (ORIy_deg-(0.5))**2 ) <(0.75)**2 ) & (ORIr_kpc>=cut_radius_kpc)) 

cut0_5       = ( np.logical_or(cut0_1, cut0_2) )
cut0_6       = ( np.logical_or(cut0_3, cut0_4) )
cut0         = ( np.logical_or(cut0_5, cut0_6) )
cut1       = ((ORIr_kpc<cut_radius_kpc))
cut        = np.where( np.logical_or(cut0, cut1) )

#cut = np.where(fig1cut_b6) 
x_deg      = ORIx_deg[cut]
y_deg      = ORIy_deg[cut]
r_kpc      = ORIr_kpc[cut]
z_ffb6     = ORIz_ffb6[cut]
zlog_ffb6  = ORIzlog_ffb6[cut]

fig=plt.figure(figsize=(15,10))
gs=gridspec.GridSpec(nrows=10,ncols=10)
ax=plt.subplot(gs[0:10,0:10])
ax.set_aspect('equal', adjustable='box')

ax.invert_xaxis()

plt.rcParams["font.size"]=30
plt.rcParams["legend.fontsize"]=30
outfile         = "figure1_ffb6.png" #閾プロット数 & 非線形モデルの温度の上限 _日付 (プロットする数値) _satter.png

os.system("rm -rf "+os.path.join(save_dir, outfile))

ax.set_xlabel("x [kpc]")
ax.set_ylabel("y [kpc]")
ax.set_title(r"$NGC \, 1068 \, \, 230\, GHz \, continuum\,  map$",y=1.05)
ax.set_xlim(2.0,-2.0)
ax.set_ylim(-2.0,2.0)
xlim_bg = [2.0, -2.0] # matplotlibのxlimと同じ値でok
ylim_bg = [-2.0, 2.0] # matplotlibのylimと同じ値でok

sc=ax.scatter(x_deg,y_deg,c=zlog_ffb6,edgecolor='None',marker="h",s=100)
cber=fig.colorbar(sc,extend="both")
c = patches.Circle(xy=(0, 0), radius=0.8, linewidth=8,fill=False, ec='white')
ax.add_patch(c)
cber.set_label(r"$log_{10}\, intensity \, \mathrm{[Jy]} $")
ax.grid()
plt.tick_params(labelsize=30)
save_dir = 'paperfigure/' 
plt.savefig(os.path.join(save_dir, outfile),dpi=200) 

interarea = np.where((ORIr_kpc<0.8))
outarea  = np.where((ORIr_kpc>0.8))
inffb6_totalflux = np.sum(ORIz_ffb6[interarea])
outffb6_totalflux = np.sum(ORIz_ffb6[outarea])
print("inffb6_totalflux:{}\n".format(inffb6_totalflux))
print("outffb6_totalflux:{}".format(outffb6_totalflux))

#########################################################################


cut0_1       = ((ORIz_Paa >= snr*Paasigma)&(ORIr_kpc>=cut_radius_kpc))
cut0_2       = ((ORIr_kpc>=cut_radius_kpc)&(abs(ORIr_kpc)<1.5))
cut0_3       = (( ( (ORIx_deg-(-1.0))**2 + (ORIy_deg-(-0.5))**2 ) <1 ) &  (ORIr_kpc>=cut_radius_kpc)) 
cut0_4       = (( ( (ORIx_deg-(1.25))**2 + (ORIy_deg-(0.5))**2 ) <(0.75)**2 ) & (ORIr_kpc>=cut_radius_kpc)) 

cut0_5       = ( np.logical_or(cut0_1, cut0_2) )
cut0_6       = ( np.logical_or(cut0_3, cut0_4) )
cut0         = ( np.logical_or(cut0_5, cut0_6) )
cut1       = ((ORIr_kpc<cut_radius_kpc))
cut        = np.where( np.logical_or(cut0, cut1) )

x_deg      = ORIx_deg[cut]
y_deg      = ORIy_deg[cut]
r_kpc      = ORIr_kpc[cut]
z_sync     = ORIz_sync[cut]
zlog_sync  = ORIzlog_sync[cut]

fig=plt.figure(figsize=(15,10))
gs=gridspec.GridSpec(nrows=10,ncols=10)
ax=plt.subplot(gs[0:10,0:10])
ax.set_aspect('equal', adjustable='box')

ax.invert_xaxis()

plt.rcParams["font.size"]=30
plt.rcParams["legend.fontsize"]=30
outfile         = "figure1_vla_sync.png" #閾プロット数 & 非線形モデルの温度の上限 _日付 (プロットする数値) _satter.png

os.system("rm -rf "+os.path.join(save_dir, outfile))

ax.set_xlabel("x [kpc]")
ax.set_ylabel("y [kpc]")
ax.set_title(r"$NGC \, 1068 \, \, 14.9\, GHz \, continuum\,  map$",y=1.05)
ax.set_xlim(2.0,-2.0)
ax.set_ylim(-2.0,2.0)
xlim_bg = [2.0, -2.0] # matplotlibのxlimと同じ値でok
ylim_bg = [-2.0, 2.0] # matplotlibのylimと同じ値でok

sc=ax.scatter(x_deg,y_deg,c=zlog_sync,edgecolor='None',marker="h",s=100)
cber=fig.colorbar(sc,extend="both")
c = patches.Circle(xy=(0, 0), radius=0.8, linewidth=8,fill=False, ec='white')
ax.add_patch(c)
cber.set_label(r"$log_{10}\, intensity \, \mathrm{[Jy]} $")
ax.grid()
plt.tick_params(labelsize=30)
save_dir = 'paperfigure/' 
plt.savefig(os.path.join(save_dir, outfile),dpi=200) 





interarea = np.where((r_kpc<0.8))
outarea  = np.where((r_kpc>0.8))
insync_totalflux = np.sum(z_sync[interarea])
outsync_totalflux = np.sum(z_sync[outarea])
print("insync_totalflux:{}\n".format(insync_totalflux))
print("outsync_totalflux:{}".format(outsync_totalflux))


#########################################################################

### remove nan/inf and remove data points from the central region (~np.isnan(K_13co))&(~np.isinf(K_13co))&


cut0_1       = ((ORIz_Paa >= snr*Paasigma)&(ORIr_kpc>=cut_radius_kpc))
cut0_2       = ((ORIr_kpc>=cut_radius_kpc)&(abs(ORIr_kpc)<1.5))
cut0_3       = (( ( (ORIx_deg-(-1.0))**2 + (ORIy_deg-(-0.5))**2 ) <1 ) &  (ORIr_kpc>=cut_radius_kpc)) 
cut0_4       = (( ( (ORIx_deg-(1.25))**2 + (ORIy_deg-(0.5))**2 ) <(0.75)**2 ) & (ORIr_kpc>=cut_radius_kpc)) 

cut0_5       = ( np.logical_or(cut0_1, cut0_2) )
cut0_6       = ( np.logical_or(cut0_3, cut0_4) )
cut0         = ( np.logical_or(cut0_5, cut0_6) )
cut1       = ((ORIr_kpc<cut_radius_kpc))
cut        = np.where( np.logical_or(cut0, cut1) )


x_deg      = ORIx_deg[cut]
y_deg      = ORIy_deg[cut]
z_Paa      = ORIz_Paa[cut]

zlog_Paa   = ORIzlog_Paa[cut]

r_kpc      = ORIr_kpc[cut]
z_abr      = ORIz_abr[cut]


sigma_stat = 0.05
zerr_Paarms = sigma_stat * z_Paa
zlog_Paarms = 1 / np.log(10) * zerr_Paarms / z_Paa



fig=plt.figure(figsize=(15,10))
gs=gridspec.GridSpec(nrows=10,ncols=10)
ax=plt.subplot(gs[0:10,0:10])
ax.set_aspect('equal', adjustable='box')
ax.invert_xaxis()
plt.rcParams["font.size"]=30
plt.rcParams["legend.fontsize"]=30
plt.rcParams['axes.facecolor'] = 'indigo'


outfile         = "figure1_Paa.png" #閾プロット数 & 非線形モデルの温度の上限 _日付 (プロットする数値) _satter.png

os.system("rm -rf "+os.path.join(save_dir, outfile))

ax.set_xlabel("x [kpc]")
ax.set_ylabel("y [kpc]")
ax.set_title(r"$NGC \, 1068 \, \, Pa \,alpha \,  map$",y=1.05)
ax.set_xlim(2.0,-2.0)
ax.set_ylim(-2.0,2.0)
xlim_bg = [2.0, -2.0] # matplotlibのxlimと同じ値でok
ylim_bg = [-2.0, 2.0] # matplotlibのylimと同じ値でok

sc=ax.scatter(x_deg,y_deg,c=zlog_Paa,edgecolor='None',marker="h",s=100)
cber=fig.colorbar(sc,extend="both")
c = patches.Circle(xy=(0, 0), radius=0.8, linewidth=8,fill=False, ec='white')
ax.add_patch(c)
cber.set_label(r"$log_{10}\, intensity \, \mathrm{[erg/s/cm^{2}]} $")
ax.grid()

plt.tick_params(labelsize=30)
save_dir = 'paperfigure/' 

plt.savefig(os.path.join(save_dir, outfile),dpi=200) 

plt.rcParams['axes.facecolor'] = 'indigo'
plt.rcParams.update(plt.rcParamsDefault)

interarea = np.where((ORIr_kpc<0.8))
outarea  = np.where((ORIr_kpc>0.8))
inPaa_totalflux = np.sum(ORIz_Paa[interarea])
outPaa_totalflux = np.sum(ORIz_Paa[outarea])
print("inPaa_totalflux:{} ± {}\n".format( inPaa_totalflux,inPaa_totalflux*0.05 ))
print("outPaa_totalflux:{} ± {}".format( outPaa_totalflux,inPaa_totalflux*0.05 ))
