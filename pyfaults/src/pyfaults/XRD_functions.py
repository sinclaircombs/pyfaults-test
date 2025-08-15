"""
XRD_functions.py

Module containing functions related to simulating powder X-ray diffraction (PXRD) patterns

tt_to_q --> converts 2theta values to Q values
importFile --> import text file of PXRD data
importExpt --> import file of experimental PXRD data and adjust 2theta range to match simulated PXRD
fullSim --> calculates a single PXRD pattern from a CIF
simulate --> calculates a set of PXRD patterns from all CIFs in a directory
"""

#---------- import packages ----------
import Dans_Diffraction as df
import numpy as np
import os, glob

#-------------------------------------
#--------- FUNCTION: tt_to_q ---------
#-------------------------------------
def tt_to_q(twotheta, wavelength):
    """
    Converts 2theta (degrees) values to Q values

    Parameters
    ----------
    twotheta : nparray
        2Theta values in units of degrees
    wavelength : float
        Instrument wavelength in Angstroms

    Returns
    -------
    Q : nparray
        Q values in units of inverse Angstroms
    """
    Q = 4 * np.pi * np.sin((twotheta * np.pi)/360) / wavelength
    return Q



#-------------------------------------
#------- FUNCTION: importFile --------
#-------------------------------------
def importFile(path, filename, *, ext='.txt', norm=True):
    """
    Imports a text file containing PXRD data

    Parameters
    ----------
    path : str
        Directory where data file is stored
    filename : str
        Name of data file
    ext : str, optional
        file extension, by default '.txt'
    norm : bool, optional
        Set to true to normalize intensity values and False otherwise, by default True

    Returns
    -------
    q : nparray
        Imported Q values
    ints : nparray
        Imported intensity values
    """
    
    q, ints = np.loadtxt(path + filename + ext, unpack=True, dtype=float)
    return q, ints



#-------------------------------------
#------- FUNCTION: importExpt --------
#-------------------------------------
def importExpt(path, filename, wl, maxTT, *, ext='.txt'):
    """
    Imports experimental PXRD data and adjusts to match 2theta range of simulated PXRD data

    Parameters
    ----------
    path : str
        Directory where data file is stored
    filename : str
        Name of data file
    wl : float
        Instrument wavelength in Angstroms
    maxTT : float
        Maximum 2theta of simulated PXRD in degrees
    ext : _type_, optional
        _description_, by default None

    Returns
    -------
    exptQ : nparray
        Imported Q values, truncated as necessary
    truncInts : nparray
        Imported intensity values, truncated as necessary
    """

    # import experimental data
    exptTT, exptInts = importFile(path, filename, ext=ext)
    
    # truncate 2theta range according to maxTT
    truncTT = []
    truncInts = []
    for i in range(len(exptTT)):
        if exptTT[i] <= maxTT:
            truncTT.append(exptTT[i])
            truncInts.append(exptInts[i])
    truncTT = np.array(truncTT)
    truncInts = np.array(truncInts)
    
    # convert to Q
    exptQ = tt_to_q(truncTT, wl)
    
    return exptQ, truncInts



#-------------------------------------
#--------- FUNCTION: fullSim ---------
#-------------------------------------
def fullSim(path, cif, wl, tt_max, savePath, *, pw=0.0, bg=0):
    """
    Simulates a powder X-ray diffraction pattern from a CIF and exports data

    Parameters
    ----------
    path : str
        File path to directory where CIF is stored
    cif : str
        Name of CIF
    wl : float
        Simulated instrument wavelength in units of Angstroms
    tt_max : float
        Maximum 2theta in units of degrees
    savePath : str
        File path to directory to save diffraction data to
    pw : float, optional
        Artificial peak broadening term, by default None
    bg : float, optional
        Average of normal instrument background, by default None

    Returns
    -------
    q : nparray
        Diffraction pattern Q values in units of inverse Angstroms
    ints : nparray
        Normalized diffraction pattern intensity values in arbitrary units / counts
    """

    # load CIF as crystal structure readable by Dans_Diffraction
    struct = df.Crystal(path + cif + '.cif')
    
    # calculate energy in keV from wavelength
    energy_kev = df.fc.wave2energy(wl)
    # set scattering source type to X-rays
    struct.Scatter.setup_scatter('xray')
    # calculate maximum wavevector from maximum 2theta and energy
    wavevector_max = df.fc.calqmag(tt_max, energy_kev)
    
    # calculate PXRD pattern
    q, ints = struct.Scatter.generate_powder(wavevector_max, 
                                             peak_width=pw, 
                                             background=bg, 
                                             powder_average=True)
    
    # export diffraction pattern to text file
    with open(savePath + cif + '_sim.txt', 'w') as f:
        for (q, ints) in zip(q, ints):
            f.write('{0} {1}\n'.format(q, ints))
    f.close() 
    
    return q, ints 



#-------------------------------------
#-------- FUNCTION: simulate ---------
#-------------------------------------
def simulate(path):
    """
    Simulates powder X-ray diffraction patterns of all CIFs in a given directory

    Parameters
    ----------
    path : str
        File path of directory where CIFs are stored
    """
    
    import pyfaults as pf
    
    unitcell, ucDF, gsDF, scDF, simDF = pf.pfInput.pfInput(path)

    wl = simDF.loc[0, 'wl']
    maxTT = simDF.loc[0, 'maxTT']
    pw = simDF.loc[0, 'pw']
    
    # creates folder to store generated data
    if os.path.exists('./simulations') == False:
        os.mkdir('./simulations')

    fileList = glob.glob('./supercells/*')
    for f in range(len(fileList)):
        fileList[f] = fileList[f].replace('.cif', '')
        fileList[f] = fileList[f].replace('./supercells\\', '')
    
    for f in fileList:
        q, ints = pf.simXRD.fullSim('./supercells/', f, wl.iloc[0], maxTT.iloc[0], pw=pw.iloc[0], savePath='./simulations/')