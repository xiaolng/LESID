import os
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splrep, splev

import light_echo as le


def get_snjsonfiles():
    """get all json files in astrocat"""
    # all json files
    snjsonfiles = glob.glob("../astrocats/*/*")
    snjsonfiles.sort()

    return snjsonfiles
    
def get_sndata(snid, snjsonfiles):
    """get sndata from json files in astrocat"""
    # path to inside astrocat
    snjsonfile = [s for s in snjsonfiles if snid in s][0]

    # copy json files to SN_data from astrocat
    if not os.path.isdir(f"./SN_data/{snid}"):
        os.system(f"mkdir SN_data/{snid}/")
        os.system(f"cp {snjsonfile} SN_data/{snid}/SN{snid}.json")
    assert os.path.exists(f"./SN_data/{snid}/SN{snid}.json")

    # load json
    with open(f"./SN_data/{snid}/SN{snid}.json") as f:
        sndata = json.load(f)[f'SN{snid}']

    return sndata


def write_photometry_spectra(snid, sndata):
    """write phoeometry and spectra to txt files"""
    redshift = float(sndata['redshift'][0]['value'])
    # extinction
    ext_dict = {'B': 0.081, 'V': 0.061, 'R': 0.048, 'I': 0}  # data source?

    with open(f"./SN_data/{snid}/{snid}_phot.txt", 'w') as f:
        # z=0.002058
        # A_B=0.081,A_V=0.061,A_R=0.048
        # FLT,MJD,MAG,MAGERR
        f.write(f"# z={redshift}\n")
        f.write(f"# A_B={ext_dict['B']},A_V={ext_dict['V']}, A_R={ext_dict['B']},A_I={ext_dict['I']}\n")
        f.write(f"# FLT,MJD,MAG,MAGERR\n")
        for ph in sndata['photometry']:
            if 'band' in ph.keys():
                if ph['band'] in ['B', 'V', 'R', 'I']:
                    f.write(f"{ph['band']},{ph['time']},{ph['magnitude']},0.01\n")
    print(f'write photometry {snid}')

    # write spectra
    for spectra_data in sndata['spectra']:
        if 'time' in spectra_data.keys():
            mjd = spectra_data['time']
            with open(f"./SN_data/{snid}/{snid}_{mjd}.flm", 'w') as f:
                f.write(f"# MJD: {mjd}\n")
                f.write(f"# REDSHIFT: {redshift}\n")
                # write each line
                for sp in spectra_data['data']:
                    f.write(f"{sp[0]} {sp[1]}\n")
    print(f"write spectra {snid}")


def read_spectra(snid, mjd_max, A_V=None):
    # read spectra, deredden, and deredshift
    spec_files = glob.glob("SN_data/{}/{}*.flm".format(snid, snid))
    spec_data = []
    ext = le.F99(Rv=3.1)
    for spec in spec_files:
        with open(spec) as spec_file:
            lines = spec_file.readlines()
            mjd = float(lines[0].split(':')[1])
            redshift = float(lines[1].split(':')[1])
            wave, flux = np.genfromtxt(spec, unpack=True)
            if A_V:
                red = ext.extinguish(wave * u.AA, Av=A_V)
                flux /= red
            # spec_data.append([mjd-mjd_max/(1.+redshift),wave/(1.+redshift),flux])
            spec_data.append([(mjd - mjd_max) / (1. + redshift), wave / (1. + redshift), flux])

    return spec_data, redshift


def get_photometry_spectra(snid, plot=True):
    sndata = get_sndata(snid=snid)

    write_photometry_spectra(snid=snid, sndata=sndata)

    sntype = [s['value'] for s in sndata['claimedtype'] ]


    #snid = '1994d'

    phot_data, filts, ext_dict, redshift = le.read_photometry(snid)
    filts = [f for f in filts] # change from set to list
    
    if plot:
        plot_photometry(snid, sntype, phot_data, filts)

    # find idx of max mag in B band
    if 'B' in filts:
        mjd_max_idx = np.argmin(phot_data[ filts.index('B') ][1])
        mjd_max = phot_data[filts.index('B')][0][mjd_max_idx]
    else:
        mjd_max = phot_data[0][0][0]  # need to modify 
    
    ### get spectra

    spec_data, redshift = read_spectra(snid, mjd_max, A_V=None) # use A_V = ext_dict['V'], 1994d spectra malready dereddened

    phases = np.transpose(spec_data)[0]

    # sort phases
    phases = [sp[0] for sp in spec_data]

    spec_data_sorted = []

    for i in np.argsort(phases):
        spec = spec_data[i]
        spec_data_sorted.append(spec)

    spec_data = spec_data_sorted

    if plot:
        plot_spectra(snid, spec_data)
    
    return phot_data, spec_data
    


def fit_light_curve(data, filts, filts_order=['B', 'V', 'R', 'I'], spec_phase=None, linear=False, plot=False,
                    xlim=[-25, 200], ylim=[16, 10]):
    """ fit light curves, 
        return a list of splines for each filter
    """
    #if plot:
    #   le.basic_format()

    ext1 = -5
    ext2 = 20

    splines = []
    for i, band in enumerate(filts_order):
        mjds, mags, errs = data[filts.index(band)] # get photometry at band
        #print(filts.index(band))
        if len(mjds) > 1:
            if spec_phase is not None:
                if np.amin(spec_phase) < mjds[0]:
                    ext1 = (np.amin(spec_phase) - mjds[0]) - 5
                if np.amax(spec_phase) > mjds[-1]:
                    ext2 = (np.amax(spec_phase) - mjds[-1]) + 5
            tnew = np.arange(mjds[0], mjds[-1], dtype=float, step=0.1)
            tnew_ext1 = np.arange(mjds[0] + ext1, mjds[0], dtype=float, step=0.1)
            tnew_ext2 = np.arange(mjds[-1], mjds[-1] + ext2, dtype=float, step=0.1)
            m_spline, m_spline_ext1, m_spline_ext2 = le.interp_light_curve(mjds, mags, errs, linear=linear)
            if m_spline is not None:
                m_smooth = splev(tnew, m_spline)
                m_smooth_ext1 = splev(tnew_ext1, m_spline_ext1)
                m_smooth_ext2 = splev(tnew_ext2, m_spline_ext2)
                if plot:
                    plt.plot(tnew, m_smooth, color=le.color_dict[band], )
                    plt.plot(tnew_ext1, m_smooth_ext1, color=le.color_dict[band], linestyle='--')
                    plt.plot(tnew_ext2, m_smooth_ext2, color=le.color_dict[band], linestyle='--')
                splines.append([m_spline, m_spline_ext1, m_spline_ext2])
        if plot:
            plt.errorbar(mjds, mags, yerr=errs, label=band, fmt='o', markersize=6, color=le.color_dict[band],
                         alpha=0.5)

    if plot:
        if spec_phase is not None:
            for i, d in enumerate(spec_phase):
                if i == 0:
                    plt.axvline(x=d, color='k', alpha=.7, label='spectra phase')
                else:
                    plt.axvline(x=d, color='k', alpha=.7, linestyle='--', linewidth=.5)

        plt.gca().invert_yaxis()
        plt.xlabel('phase (days)', )
        plt.ylabel('magnitude', )
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.legend()
        # plt.show()
    return splines



def lc_model_vacca(x, p0, p1, p2, p3, p4, p5, p6,):
    """ Vacca & Leibundgut 1996 
    
    - exponential rise, for early
    - Gaussian atop, for peak phase
    - linear decay, for late-time decline
    
    """
    p = [p0, p1, p2, p3, p4, p5, p6,]
    
    # gaussian 
    g = p[0] * np.exp(-(x-p[1])**2 / p[2]**2)
    
    # linear decay
    g += p[4] * x + p[3]
    
    # exponential rise
    g *= (np.exp(-p[5] * (x-p[6]))+1)
    
    return g


def plot_photometry(snid, sntype, phot_data, filts, savefig=False):

    fig, ax = plt.subplots(1,1)

    for band, dat in zip(filts, phot_data):
        mjds = dat[0]
        mags = dat[1]

        ax.plot(mjds, mags, '.-', label=band, color=le.color_dict[band])

    ax.invert_yaxis()
    ax.set_xlabel('MJD')
    ax.set_ylabel('mag')
    ax.legend()
    ax.set_title(f'{snid}, {sntype}')

    if savefig:
        fig.savefig(f"./plots/core_collapse/{snid}_phot.png")
    return fig


def plot_spectra(snid, spec_data, ncol=2, saveto="LEtemp/"):

    nrow = int(np.ceil(len(spec_data) / ncol))

    fig, axs = plt.subplots(nrow, ncol, 
                            figsize=(4*ncol, 2*nrow),
                            sharex=True)

    axs = axs.flatten()
    for i, spec in enumerate(spec_data[:]):
        ax = axs[i]
        x = spec[1]
        y = spec[2]
        ax.plot(spec[1], spec[2])    

        ax.set_ylabel('flux')
        ax.set_title(f'idx {i}, phase {spec[0]:0.2f}')

    ax.set_xlabel('wavelength')
    ax.set_xlim([3000, 10000])

    #ax.set_title(f'{snid}, {sntype}')
    fig.suptitle(f'{snid}')
    fig.savefig(f"{saveto}/{snid}_spectra.png")
    
    return fig

def plot_spectra_dict(snid, spec_data, ncol=2, 
                      wavestr='wavelength_interp', fluxstr='flux_interp', 
                        saveto="LEtemp/"):

    nrow = int(np.ceil(len(spec_data) / ncol))

    fig, axs = plt.subplots(nrow, ncol, 
                            figsize=(4*ncol, 2*nrow),
                            sharex=True)

    axs = axs.flatten()
    for i, ph in enumerate(spec_data):
        ax = axs[i]
        x = spec_data[ph][wavestr]
        y = spec_data[ph][fluxstr]
        ax.plot(x, y)    

        ax.set_ylabel('flux')
        ax.set_title(f'idx {i}, phase {ph:0.2f}')

    ax.set_xlabel('wavelength')
    ax.set_xlim([3000, 10000])

    #ax.set_title(f'{snid}, {sntype}')
    fig.suptitle(f'{snid}')
    fig.savefig(f"{saveto}/{snid}_spectra.png")
    
    return fig


