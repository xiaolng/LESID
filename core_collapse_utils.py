import numpy as np
import pylab as plt
import pysynphot
import light_echo as le
import pandas as pd
import glob
from scipy.interpolate import splrep, splev, interp2d, LSQBivariateSpline
from cc_configs import *

def read_spectra(snid, mjd_max, A_V=None):
    """ FBB needs a doc string"""

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


def fit_light_curve(data, filts, spec_phase=None, linear=False, plot=False):
    """ FBB needs a doc string"""

    if plot:
        le.basic_format()

    ext1 = -5
    ext2 = 20

    splines = []
    for i, band in enumerate(filts):
        mjds, mags, errs = data[i]
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
            plt.errorbar(mjds, mags, yerr=errs, label=band, fmt='o', markersize=10, color=le.color_dict[band])

    if plot:
        if spec_phase is not None:
            for i, d in enumerate(spec_phase):
                if i == 0:
                    plt.axvline(x=d, color='k', alpha=.7, label='Spectrum Epoch')
                else:
                    plt.axvline(x=d, color='k', alpha=.7)

        plt.gca().invert_yaxis()
        plt.xlabel('MJD (days)', fontsize=35)
        plt.ylabel('Magnitude', fontsize=35)
        # plt.xlim([-25, 200])
        # plt.ylim([16, 10])
        plt.legend()
        # plt.show()
    return splines


def remove_peak(x, y, peak_idx=[], width_idx=10, height=1, ):
    """remove the telluric peak
    Parameters:
        x, y: array
        width_idx: int, width of the points to remove

    """
    if len(peak_idx) == 0:
        # find the peak location use scipy.signal
        peak_idx, peak_dict = signal.find_peaks(y, height=height, width=0)

    # find the index of peak area
    peak_left = np.concatenate([peak_idx - i for i in range(width_idx)])
    peak_right = np.concatenate([peak_idx + i for i in range(width_idx)])
    peak_area = np.concatenate([peak_left, peak_right])
    peak_area.sort()

    # linear interpolate peak area
    y_nopeak = y.copy()
    no_peak_area = np.delete(np.arange(0, len(x)), peak_area)
    y_nopeak[peak_area] = np.interp(x[peak_area],
                                    x[no_peak_area],
                                    y[no_peak_area])

    return y_nopeak


def app2absMag(appm, lumdist, udist=10e6):
    """convert apparent magnitude to absolute magnitude
    lumdist: luminosity distance D_L, in parsecs
    udist: unit of distance, Mpc, default
    """

    absM = appm - 5 * np.log10(lumdist / 10)

    return absM


def abs2appMag(absM, lumdist, udist=10e6):
    """ FBB needs a doc string"""

    appm = absM + 5 * np.log10(lumdist * udist / 10)

    return appm


def generate_photometry_for_epoch(phase, bands, splines):
    """ FBB needs a better doc string"""

    """get magnitude from fitted splines"""
    phot_dict = {}
    for i, band in enumerate(bands):
        m_smooth = splev(phase, splines[i][0],ext=1)
        if m_smooth == 0 and phase < 0:
            m_smooth = splev(phase, splines[i][1])
        if m_smooth == 0 and phase > 0:
            m_smooth = splev(phase, splines[i][2])
        # print (phase, band, m_smooth)
        phot_dict[band] = m_smooth
    return phot_dict


def obs_spec(wave, flux, mags_lc, lumdist=94.4, absmag=False):
    """ FBB needs a doc string"""

    sp = pysynphot.ArraySpectrum(wave, flux, fluxunits='flam')
    wave, flux = sp.getArrays()

    bp_dict = {}
    mags_dict = {}

    df_obs = pd.DataFrame(columns=['band',
                                   'bp', 'spec_obs',
                                   'mag_obs', 'mag_lc',
                                   'efflam', 'scale'])

    fig, ax = plt.subplots(1, 1)

    for band in ['B', 'V', 'I', 'R', ]:
        bp = pysynphot.ObsBandpass(band_dict[band]);

        spec_obs = pysynphot.Observation(sp, bp, force='extrap');
        mag = spec_obs.effstim('vegamag')
        # convert to apparent mags
        if absmag:
            mag = abs2appMag(absM=mag, lumdist=lumdist, udist=10e6)

        try:
            mag_lc = mags_lc[band].item()
        except KeyError:
            #FBB no exceptions without a specification of which exception you are catching
           continue

        df_obs.loc[len(df_obs)] = {'band': band,
                                   'bp': bp, 'spec_obs': spec_obs,
                                   'mag_obs': mag, 'mag_lc': mag_lc,
                                   'efflam': spec_obs.efflam(),
                                   'scale': 10 ** ((mag_lc - mag) / 2.5)}
        wave_obs, flux_obs = spec_obs.getArrays()
        ax.plot(wave_obs, flux_obs, label=f"{band} {mag:.2f}; mag_lc {mag_lc: .2f}",
                 color=le.color_dict[band])
        # plt.plot(bp.wave, bp.throughput)
        ax.axvline(spec_obs.efflam(),
                   color=le.color_dict[band], linewidth=.5)

    plt.legend()
    plt.xlabel(r'wavelength ($\AA$)')
    plt.ylabel('flux')

    return df_obs


def swarp(df_obs, wave, flux, wave_original, flux_original):

    """ FBB needs a doc string"""
    x = df_obs.efflam.values
    y = df_obs.scale.values
    x = np.concatenate([[10], x, [20000]])

    y = np.concatenate([[y[0]], y, [y[-1]]])

    print('scales', y)
    # spline representation
    x_new = wave
    # x_new = np.linspace(0, 10000)
    sort = np.argsort(x)
    x = x[sort]
    y = y[sort]
    y_new = splev(x_new, splrep(x, y, k=2))

    fig = plt.figure(figsize=(15,5))
    ax = fig.add_subplot(121)
    ax.plot(x, y, '.')
    ax.plot(x_new, y_new)
    ax.set_xlabel('wavelength')
    ax.set_ylabel('scale factor')
    ax.set_ylim([y_new.min() - .1, y_new.max() + .1])
    ax.set_xlim(x_new.min(), x_new.max())

    scale = y_new
    ax = fig.add_subplot(122)

    medianflux = np.median(flux_original)
    ax.plot(wave_original, flux_original, label='original')
    ax.plot(wave, flux / np.median(flux) * medianflux, '--',
            label='previous', alpha=0.7)

    sp = pysynphot.ArraySpectrum(wave, flux / scale, fluxunits='flam')

    wave, flux = sp.getArrays()
    ax.plot(wave, flux / np.median(flux) * medianflux ,
            label='scaled spec', alpha=0.7)

    bp_dict = {}
    mags_dict = {}

    # for band in ['B', 'V', 'R', 'I']:
    #    bp = pysynphot.ObsBandpass(band_dict[band]);
    #    spec_obs = pysynphot.Observation(sp, bp, force='extrap');
    #    mag = spec_obs.effstim('vegamag')
    #    mag_lc = mags_lc[band].item()
    #
    #
    #    wave_obs, flux_obs = spec_obs.getArrays()
    #    plt.plot(wave_obs, flux_obs, label=f"{band} {mag:.2f}; mag_lc {mag_lc: .2f}",
    #             color=le.color_dict[band])
    #    #plt.plot(bp.wave, bp.throughput)
    #    plt.vlines(spec_obs.efflam(), ymin=0, ymax=5e-15,
    #               color=le.color_dict[band], linewidth=.5)
    #
    plt.legend()
    ax.set_xlabel(r'wavelength ($\AA$)')
    ax.set_ylabel('flux')
    return flux, y
