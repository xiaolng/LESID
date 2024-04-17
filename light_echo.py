import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splrep, splev, interp2d, LSQBivariateSpline
import glob
from dust_extinction.parameter_averages import F99
from astropy import units as u
import pysynphot
import scipy.optimize as opt

color_dict = {'U': 'magenta',
          'B': 'blue',
          'V': 'green',
          'R': 'red',
          'I': 'brown',

          'u': 'purple',
          'g': 'seagreen',
          'r': 'crimson',
          'i': 'orangered',
          'z': 'darkred',

          "u'": 'purple',
          "g'": 'seagreen',
          "r'": 'crimson',
          "i'": 'orangered',
          "z'": 'darkred',

          'H': 'yellow',
          'K': 'teal',
          'J': 'crimson',
          'V': 'orange',
          'V0': 'orange',
          'Y': 'pink',

          'W1': 'gray',
          'W2': 'black',
          'M2': 'darkgray',

          'C': 'limegreen',
          'Js': 'gold',
          'Ks': 'lightgray',
          'Jrc2': 'lavender',
          'Hdw': 'darkblue',
          'Ydw': 'darkgreen',
          'Jdw': 'darkgray',
             }


def basic_format(size=[16,12]):
    plt.rc('font', family='serif')
    fig, ax = plt.subplots(1,1)
    fig.set_size_inches(size[0], size[1], forward = True)
    plt.minorticks_on()
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    plt.tick_params(
        which='major', 
        bottom='on', 
        top='on',
        left='on',
        right='on',
        direction='in',
        length=20)
    plt.tick_params(
        which='minor', 
        bottom='on', 
        top='on',
        left='on',
        right='on',
        direction='in',
        length=10)
    return fig, ax


def spectres(new_wavs, spec_wavs, spec_fluxes, spec_errs=None, fill=None,
             verbose=True):

    """
    Function for resampling spectra (and optionally associated
    uncertainties) onto a new wavelength basis.
    Parameters
    ----------
    new_wavs : numpy.ndarray
        Array containing the new wavelength sampling desired for the
        spectrum or spectra.
    spec_wavs : numpy.ndarray
        1D array containing the current wavelength sampling of the
        spectrum or spectra.
    spec_fluxes : numpy.ndarray
        Array containing spectral fluxes at the wavelengths specified in
        spec_wavs, last dimension must correspond to the shape of
        spec_wavs. Extra dimensions before this may be used to include
        multiple spectra.
    spec_errs : numpy.ndarray (optional)
        Array of the same shape as spec_fluxes containing uncertainties
        associated with each spectral flux value.
    fill : float (optional)
        Where new_wavs extends outside the wavelength range in spec_wavs
        this value will be used as a filler in new_fluxes and new_errs.
    verbose : bool (optional)
        Setting verbose to False will suppress the default warning about
        new_wavs extending outside spec_wavs and "fill" being used.
    Returns
    -------
    new_fluxes : numpy.ndarray
        Array of resampled flux values, first dimension is the same
        length as new_wavs, other dimensions are the same as
        spec_fluxes.
    new_errs : numpy.ndarray
        Array of uncertainties associated with fluxes in new_fluxes.
        Only returned if spec_errs was specified.
    """

    # Rename the input variables for clarity within the function.
    old_wavs = spec_wavs
    old_fluxes = spec_fluxes
    old_errs = spec_errs

    # Arrays of left hand sides and widths for the old and new bins
    old_lhs = np.zeros(old_wavs.shape[0])
    old_widths = np.zeros(old_wavs.shape[0])
    old_lhs = np.zeros(old_wavs.shape[0])
    old_lhs[0] = old_wavs[0]
    old_lhs[0] -= (old_wavs[1] - old_wavs[0])/2
    old_widths[-1] = (old_wavs[-1] - old_wavs[-2])
    old_lhs[1:] = (old_wavs[1:] + old_wavs[:-1])/2
    old_widths[:-1] = old_lhs[1:] - old_lhs[:-1]
    old_max_wav = old_lhs[-1] + old_widths[-1]

    new_lhs = np.zeros(new_wavs.shape[0]+1)
    new_widths = np.zeros(new_wavs.shape[0])
    new_lhs[0] = new_wavs[0]
    new_lhs[0] -= (new_wavs[1] - new_wavs[0])/2
    new_widths[-1] = (new_wavs[-1] - new_wavs[-2])
    new_lhs[-1] = new_wavs[-1]
    new_lhs[-1] += (new_wavs[-1] - new_wavs[-2])/2
    new_lhs[1:-1] = (new_wavs[1:] + new_wavs[:-1])/2
    new_widths[:-1] = new_lhs[1:-1] - new_lhs[:-2]

    # Generate output arrays to be populated
    new_fluxes = np.zeros(old_fluxes[..., 0].shape + new_wavs.shape)

    if old_errs is not None:
        if old_errs.shape != old_fluxes.shape:
            raise ValueError("If specified, spec_errs must be the same shape "
                             "as spec_fluxes.")
        else:
            new_errs = np.copy(new_fluxes)

    start = 0
    stop = 0

    # Calculate new flux and uncertainty values, looping over new bins
    for j in range(new_wavs.shape[0]):

        # Add filler values if new_wavs extends outside of spec_wavs
        if (new_lhs[j] < old_lhs[0]) or (new_lhs[j+1] > old_max_wav):
            new_fluxes[..., j] = fill

            if spec_errs is not None:
                new_errs[..., j] = fill

            if (j == 0) and verbose:
                print("\nSpectres: new_wavs contains values outside the range "
                      "in spec_wavs. New_fluxes and new_errs will be filled "
                      "with the value set in the 'fill' keyword argument (nan "
                      "by default).\n")
            continue

        # Find first old bin which is partially covered by the new bin
        while start+1 < len(old_lhs) - 1 and old_lhs[start+1] <= new_lhs[j]:
            start += 1
        
        # Find last old bin which is partially covered by the new bin
        while stop+1 < len(old_lhs) - 1 and old_lhs[stop+1] < new_lhs[j+1]:
            stop += 1

        # If new bin is fully inside an old bin start and stop are equal
        if stop == start:
            new_fluxes[..., j] = old_fluxes[..., start]
            if old_errs is not None:
                new_errs[..., j] = old_errs[..., start]

        # Otherwise multiply the first and last old bin widths by P_ij
        else:
            start_factor = ((old_lhs[start+1] - new_lhs[j])
                            / (old_lhs[start+1] - old_lhs[start]))

            end_factor = ((new_lhs[j+1] - old_lhs[stop])
                          / (old_lhs[stop+1] - old_lhs[stop]))

            old_widths[start] *= start_factor
            old_widths[stop] *= end_factor

            # Populate new_fluxes spectrum and uncertainty arrays
            f_widths = old_widths[start:stop+1]*old_fluxes[..., start:stop+1]
            new_fluxes[..., j] = np.sum(f_widths, axis=-1)
            new_fluxes[..., j] /= np.sum(old_widths[start:stop+1])

            if old_errs is not None:
                e_wid = old_widths[start:stop+1]*old_errs[..., start:stop+1]

                new_errs[..., j] = np.sqrt(np.sum(e_wid**2, axis=-1))
                new_errs[..., j] /= np.sum(old_widths[start:stop+1])

            # Put back the old bin widths to their initial values
            old_widths[start] /= start_factor
            old_widths[stop] /= end_factor

    # If errors were supplied return both new_fluxes and new_errs.
    if old_errs is not None:
        return np.array([new_wavs, new_fluxes, (1./new_errs)**2.])
        # return new_fluxes, new_errs

    # Otherwise just return the new_fluxes spectrum array
    else:
        # return new_fluxes
        return np.array([new_wavs, new_fluxes])


def read_photometry(snid):
    #reads photomtery and filter info from 'snid_phot.txt'
    data_file = np.genfromtxt("SN_data/{}/{}_phot.txt".format(snid,snid), 
                        delimiter = ',', dtype=str,skip_header=2)
    data_file = np.transpose(data_file)
    bands = set(data_file[0])
    data = []

    with open("SN_data/{}/{}_phot.txt".format(snid,snid)) as snphot:
        lines = snphot.readlines()
        redshift = float(lines[0].split('=')[1])
        head = lines[1].split(',')
        ext_dict = {}
        for ext in head:
            band = ext.split('\n')[0].split('=')[0][-1]
            ext_val = float(ext.split('\n')[0].split('=')[1])
            ext_dict[band] = ext_val
            
    for band in bands:
        inds = data_file[0] == band
        mjds = data_file[1][inds].astype(float)
        mags = data_file[2][inds].astype(float) - ext_dict[band] #extinction correction
        errs = data_file[3][inds].astype(float)
        data.append([mjds,mags,errs])

    return data, bands, ext_dict, redshift


def read_spectra(snid, mjd_max, A_V = None):
    #read spectra, deredden, and deredshift
    spec_files = glob.glob("SN_data/{}/{}*.flm".format(snid,snid))
    spec_data = []
    ext = F99(Rv=3.1)
    for spec in spec_files:
        with open(spec) as spec_file:
            lines = spec_file.readlines()
            mjd = float(lines[0].split(':')[1])
            redshift = float(lines[1].split(':')[1])
            wave, flux = np.genfromtxt(spec, unpack=True)
            if A_V:
                red = ext.extinguish(wave*u.AA, Av = A_V)
                flux /= red
            spec_data.append([mjd-mjd_max/(1.+redshift),wave/(1.+redshift),flux])
    return spec_data, redshift


def interp_light_curve(mjds, mags, errs, s = 100, linear=False):
    weights = 1./errs
    if not linear:
        m_spline = splrep(mjds, mags, w = weights, k=3)
        m_spline_ext1 = splrep(mjds[:2], mags[:2], w = 1/errs[:2], k=1)
        m_spline_ext2 = splrep(mjds[-5:], mags[-5:], w = 1/errs[-5:], k=1)
    else:
        m_spline = splrep(mjds, mags, w = weights, k=1)
        m_spline_ext1 = splrep(mjds[:2], mags[:2], w = 1/errs[:2], k=1)
        m_spline_ext2 = splrep(mjds[-5:], mags[-5:], w = 1/errs[-5:], k=1)
    return m_spline, m_spline_ext1, m_spline_ext2


def fit_light_curve(data, filts, spec_phase=None, linear=False, plot=False):

    if plot:
        basic_format()

    ext1 = -5
    ext2 = 20

    splines = []
    for i, band in enumerate(filts):
        mjds, mags, errs = data[i]
        if len(mjds) > 1:
            if spec_phase is not None:
                if np.amin(spec_phase) < mjds[0]:
                    ext1 = (np.amin(spec_phase) - mjds[0])-5
                if np.amax(spec_phase) > mjds[-1]:
                    ext2 = (np.amax(spec_phase) - mjds[-1])+5
            tnew = np.arange(mjds[0], mjds[-1], dtype=float, step=0.1)
            tnew_ext1 = np.arange(mjds[0] + ext1, mjds[0], dtype=float, step=0.1)
            tnew_ext2 = np.arange(mjds[-1], mjds[-1] + ext2, dtype=float, step=0.1)
            m_spline, m_spline_ext1, m_spline_ext2 = interp_light_curve(mjds, mags, errs, linear=linear)
            if m_spline is not None:
                m_smooth = splev(tnew, m_spline)
                m_smooth_ext1 = splev(tnew_ext1, m_spline_ext1)
                m_smooth_ext2 = splev(tnew_ext2, m_spline_ext2)
                if plot:
                    plt.plot(tnew, m_smooth, color = color_dict[band])
                    plt.plot(tnew_ext1, m_smooth_ext1, color = color_dict[band], linestyle='--')
                    plt.plot(tnew_ext2, m_smooth_ext2, color = color_dict[band], linestyle='--')
                splines.append([m_spline, m_spline_ext1, m_spline_ext2])
        if plot:
            plt.errorbar(mjds, mags, yerr = errs, label = band, fmt='o', markersize=10, color = color_dict[band])
            if spec_phase is not None:
                for i, d in enumerate(spec_phase):
                    if i == 0:
                        plt.axvline(x=d, color = 'k', alpha = .7, label='Spectrum Epoch')
                    else:
                        plt.axvline(x=d, color = 'k', alpha = .7)
    if plot:
        plt.gca().invert_yaxis()
        plt.xlabel('MJD (days)', fontsize = 35)
        plt.ylabel('Magnitude', fontsize = 35)
        plt.show()

    return splines


def scale_flux_to_photometry_pysyn(phase, wave, flux, splines, valid_bands, template=None):
    # lib = pyphot.get_library()
    band_dict = {'U': 'johnson,u', 'B': 'johnson,b', 'V': 'johnson,v',  'V0': 'johnson,v',
                 'R': 'johnson,r', 'I': 'johnson,i', 
                 'u': 'sdss,u', 'g': 'sdss,g', 'r': 'sdss,r', 'i': 'sdss,i', 'z': 'sdss,z'}
#     band_dict = {b'U': 'johnson,u', b'B': 'johnson,b', b'V': 'johnson,v',  b'V0': 'johnson,v',
#                  b'R': 'johnson,r', b'I': 'johnson,i', 
#                  b'u': 'sdss,u', b'g': 'sdss,g', b'r': 'sdss,r', b'i': 'sdss,i', b'z': 'sdss,z'}
    
    if len(valid_bands) > 0:
        mags_from_phot = generate_photometry_for_epoch(phase, valid_bands, splines)
        # print (mags_from_phot)
        valid_mjds = []
        for band in mags_from_phot:
            if ~np.isnan(mags_from_phot[band]):
                valid_mjds.append(band)
        if len(valid_mjds) > 0:
            filts = {}
            for b in valid_mjds:
                # filts[b] = lib[band_dict.get(b)]
                filts[b] =  pysynphot.ObsBandpass(band_dict.get(b))

            guess = 1.e-14
            scale = opt.minimize(total_phot_offset_pysyn, guess, args = (wave, flux, filts, mags_from_phot), method = 'Nelder-Mead').x
            scale = scale[0]

            spec_phots = {}
        else:
            scale = np.nan
            print (scale, "mjd of spectrum outside photometric coverage")
    else:
        scale = np.nan
        mags_from_phot = None

    return scale, mags_from_phot



def total_phot_offset_pysyn(scale, wave, flux, filters, target_mags):
    
    diff_sum = 0.

    if scale < 0:
        scale = 1.e-14
    sp = pysynphot.ArraySpectrum(wave, scale*flux, fluxunits='flam')

    for f in filters:
        bp = filters[f]
        name = bp.name.split('/')[-1].split('_')[0] + ','+ bp.name.split('/')[-1].split('_')[1]
        if target_mags[f] is not None:
            spec_obs = pysynphot.Observation(sp, bp, force='extrap')
            if 'johnson' in name:
                mag_from_spec = spec_obs.effstim('vegamag')
            elif 'sdss' in name:
                mag_from_spec = spec_obs.effstim('abmag')

            # print (f, mag_from_spec, target_mags[f])
            diff = (mag_from_spec - target_mags[f])**2.
            diff_sum += diff
    return diff_sum


def generate_photometry_for_epoch(phase, bands, splines):
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


def prepare_spec(wrange, calibrated_spec_array, plot=True,dw=2, subnans = False):
    spec_arr_new = []
    minwaves = []
    maxwaves = []
    
    for spec in calibrated_spec_array:
        if spec[1][0] < wrange[0] and spec[1][-1] > wrange[1]:
            spec_arr_new.append([spec[0],spec[1],spec[2]])

    waverange = wrange
    specnew = {}
    phases = []
    
    # new_wavs = np.arange(2000, 12000,
    #                        dtype=float, step=dw)
    new_wavs = np.arange(3500, 10000,
                           dtype=float, step=dw)

    new_fluxes = np.nan*np.ones(len(new_wavs))
    new_weights = np.ones(len(new_wavs))
    for spec in spec_arr_new:
        snew = {}
        wave = spec[1]
        flux = spec[2]
        if plot:
            plt.plot(wave,flux)

        wave,flux= spectres(new_wavs, wave, flux)
        weight = new_weights
        
        idx = np.array((wave>waverange[0]) & (wave<waverange[1]))
        idx_nan = ~np.isnan(flux)
        
        idx_nan_w1 = (wave<spec[1][0])
        flux[idx_nan_w1] = np.median(flux[idx_nan][0:10])
        weight[idx_nan_w1] = .001
        
        idx_nan_w2 = (wave>spec[1][-1])
        flux[idx_nan_w2] = np.median(flux[idx_nan][-10:])
        weight[idx_nan_w2] = .001
        
#         snew['wavelength_interp'] = wave[idx]
#         snew['flux_interp'] = flux[idx]
        
        snew['wavelength_interp'] = wave
        snew['flux_interp'] = flux
        snew['weight_interp'] = weight
        
        
        phase = spec[0]
        phases.append(phase)
        
        specnew[phase] = snew
        if plot:
            plt.plot(wave,flux)
        plt.title(str(phase))
        plt.show()
    return specnew, np.sort(phases), wave



def linear_interpolation(spec, waves, phases, pint_range=[-20,90]):
    nwave = np.shape(spec[list(spec.keys())[0]]['wavelength_interp'])[0]
    nphase = len(list(spec.keys()))
    arr2d = []
    for i,phase in enumerate(np.sort(list(spec.keys()))):
        arr2d.append(spec[phase]['flux_interp'])

    weight2d = []
    for i,phase in enumerate(np.sort(list(spec.keys()))):
        weight2d.append(spec[phase]['weight_interp'])
    x = waves
    y = phases
    X, Y = np.meshgrid(x, y)
    Z = np.asarray(arr2d)
    W = np.asarray(weight2d)

    plt.figure(figsize=[10,10])
    plt.imshow(Z,aspect=20, origin='lower', interpolation='none')
    plt.show()
    # Z = np.sin(np.pi*X/2) * np.exp(Y/2)
    # print (np.shape(x))
    # print(np.shape(y))
    # print (np.shape(Z))
    # print (np.shape(W))

    x2 = waves
    y2 = np.arange(pint_range[0], pint_range[1], 1)
    # y2 = np.arange(0, 90, 1)

    # f = interp2d(x, y, Z, kind='linear')
    # Z2 = f(x2, y2)

    tx = x
    ty = y
    splrep = LSQBivariateSpline(X.ravel(), Y.ravel(), Z.ravel(), tx, ty, w=W.ravel(), kx=1, ky=1)
    Z2 = np.transpose(splrep(x2,y2))

    X2, Y2 = np.meshgrid(x2, y2)

    return X2,Y2,Z2,X,Y,Z


def mklecho_tophat(X2, Y2, Z2, p_int = [-20,90]):
    inds = []
    plow = p_int[0]
    phigh = p_int[1]
    phases_interp = []
    for i, y in enumerate(Y2):
        if y[0] > plow and y[0] < phigh:
            inds.append(i)
        phases_interp.append(y[0])
        
    phase_diff = np.diff(phases_interp)[0]

    lecho = np.nansum(Z2[inds]*phase_diff,axis=0)

    return lecho






