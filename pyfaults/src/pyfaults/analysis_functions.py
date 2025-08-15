"""
analysis_functions.py

Module containing functions related to analysis of simulated and/or experimental PXRD data

getNormVals --> helper function to get values at intensity maximum for use in normalization function
normalizeToExpt --> normalize a PXRD pattern to experimental data
diffCurve --> calculates a difference curve between two sets of PXRD data
r2val --> calculates an R^2 value between two sets of PXRD data
diff_r2 --> calculates both a difference curve and an R^2 value between two sets of PXRD data
fitDiff --> calculates the difference between two difference curves
simR2vals --> calculates R^2 values for each simulated PXRD pattern in a file directory against experimental PXRD data, generates text file report
stepGridSearch --> generates a step-wise set of stacking vectors and fault probabilities
randGridSearch --> generates a random set of stacking vectors and fault probabilities
"""

#---------- import packages ----------
import numpy as np
import sklearn.metrics as skl
import glob, random



#-------------------------------------
#------- FUNCTION: getNormVals -------
#-------------------------------------
def getNormVals(q, ints):
    """
    Helper function to get values at intensity maximum for use in normalization function

    Parameters
    ----------
    q : nparray
        Q values in inverse Angstroms
    ints : nparray
        Intensity values

    Returns
    -------
    intsMax : float
        Maximum intensity value
    qAtIntsMax : float
        Q value corresponding to maximum intensity
    maxIndex : float
        Array index corresponding to maximum intensity
    intsMin: float
        Minimum intensity value
    """
    intsMax = 0
    maxIndex = 0
    
    for i in range(len(ints)):
        if ints[i] > intsMax:
            intsMax = ints[i]
            maxIndex = i
            
    qAtIntsMax = q[maxIndex]
    
    intsMin = np.min(ints)
    
    return intsMax, qAtIntsMax, maxIndex, intsMin


#-------------------------------------
#----- FUNCTION: normalizeToExpt -----
#-------------------------------------
def normalizeToExpt(exptQ, exptInts, q, ints):
    """
    Normalize a PXRD pattern to experimental data

    Parameters
    ----------
    exptQ : nparray
        Experimental Q values in inverse Angstroms
    exptInts : nparray
        Experimental intensity values
    q : nparray
        Simulated Q values in inverse Angstroms
    ints : nparray
        Simulated intensity values

    Returns
    -------
    normInts: nparray
        Simulated intensity values normalized to experimental data
    """
    intsMax, qAtIntsMax, maxIndex, intsMin = getNormVals(exptQ, exptInts)
    
    qRange = [qAtIntsMax-0.1, qAtIntsMax+0.1]
    
    normMax = 0
    for i in range(len(q)):
        if q[i] >= qRange[0] and q[i] <= qRange[1]:
            if ints[i] > normMax:
                normMax = ints[i]
    
    normInts = []

    for i in range(len(ints)):
        if ints[i] > 0:
            normInts.append(ints[i] / normMax)
        else:
            normInts.append(0.0)
            
    return normInts


#-------------------------------------
#-------- FUNCTION: diffCurve --------
#-------------------------------------
def diffCurve(q1, q2, ints1, ints2):
    """
    Calculates a difference curve between two sets of PXRD data

    Parameters
    ----------
    q1 : nparray
        Dataset 1 Q values in inverse Angstroms
    q2 : nparray
        Dataset 2 Q values in inverse Angstroms
    ints1 : nparray
        Dataset 1 intensity values
    ints2 : nparray
        Dataset 2 intensity values

    Returns
    -------
    diff_q : nparray
        Q values of difference curve
    diff_ints : nparray
        Intensity values of difference curve
    """
    diff_q_list = []
    diff_ints_list = []
    for i in range(0, len(q1)):
        q1_val = float('%.3f'%(q1[i]))
        for j in range(0, len(q2)):
            q2_val = float('%.3f'%(q2[j]))
            if q1_val == q2_val:
                diff_q_list.append(q1_val)
                diff_ints_list.append(ints1[i]-ints2[j])
    diff_q = np.array(diff_q_list)
    diff_ints = np.array(diff_ints_list)
        
    return diff_q, diff_ints



#-------------------------------------
#---------- FUNCTION: r2val ----------
#-------------------------------------
def r2val(q1, q2, ints1, ints2):
    """
    Calculates an R^2 value between two sets of PXRD data

    Parameters
    ----------
    q1 : nparray
        Dataset 1 Q values in inverse Angstroms
    q2 : nparray
        Dataset 2 Q values in inverse Angstroms
    ints1 : nparray
        Dataset 1 intensity values
    ints2 : nparray
        Dataset 2 intensity values

    Returns
    -------
    r2 : float
        Calculated R^2 value
    """    
    q_list = []
    ints1_list = []
    ints2_list = []
    for i in range(len(q1)):
        q1_val = float('%.3f'%(q1[i]))
        for j in range(len(q2)):
            q2_val = float('%.3f'%(q2[j]))
            if q1_val == q2_val:
                q_list.append(q1_val)
                ints1_list.append(ints1[i])
                ints2_list.append(ints2[j])
    ints1_arr = np.array(ints1_list)
    ints2_arr = np.array(ints2_list)

    r2 = skl.r2_score(ints1_arr, ints2_arr)

    return r2



#-------------------------------------
#--------- FUNCTION: diff_r2 ---------
#-------------------------------------
def diff_r2(q1, q2, ints1, ints2):
    """
    Calculates both a difference curve and an R^2 value between two sets of PXRD data

    Parameters
    ----------
    q1 : nparray
        Dataset 1 Q values in inverse Angstroms
    q2 : nparray
        Dataset 2 Q values in inverse Angstroms
    ints1 : nparray
        Dataset 1 intensity values
    ints2 : nparray
        Dataset 2 intensity values

    Returns
    -------
    r2 : float
        Calculated R^2 value
    diff_q : nparray
        Q values of difference curve
    diff_ints : nparray
        Intensity values of difference curve
    """
    
    q_list = []
    ints1_list = []
    ints2_list = []
    for i in range(len(q1)):
        q1_val = float('%.3f'%(q1[i]))
        for j in range(len(q2)):
            q2_val = float('%.3f'%(q2[j]))
            if q1_val == q2_val:
                q_list.append(q1_val)
                ints1_list.append(ints1[i])
                ints2_list.append(ints2[j])
    ints1_arr = np.array(ints1_list)
    ints2_arr = np.array(ints2_list)
    diff_ints = np.subtract(ints1_arr, ints2_arr)

    r2 = skl.r2_score(ints1_arr, ints2_arr)

    return r2, q_list, diff_ints



#-------------------------------------
#--------- FUNCTION: fitDiff ---------
#-------------------------------------
def fitDiff(diff_ints1, diff_ints2):
    """
    Calculates the difference between two difference curves

    Parameters
    ----------
    diff_ints1 : nparray
        Difference curve intensity values from dataset 1
    diff_ints2 : nparray
        Difference curve intensity values from dataset 2

    Returns
    -------
    fitDiff : nparray
        Intensity values of the difference between two difference curves
    """
    fitDiff = np.subtract(diff_ints1, diff_ints2)
    return fitDiff



#-------------------------------------
#-------- FUNCTION: simR2vals --------
#-------------------------------------
def simR2vals(exptPath, exptFN, exptWL, maxTT):
    """
    Calculates R^2 values for each simulated PXRD pattern in a file directory against experimental PXRD data, generates text file report

    Parameters
    ----------
    exptPath : str
        File path of experimental PXRD data directory
    exptFN : str
        Experimental data file name
    exptWL : float
        Instrument wavelength in Angstroms
    maxTT : float
        Maximum two theta in degrees

    Returns
    -------
    r2vals : nparray
        List of calculated R^2 values
    """
    from pyfaults.XRD_functions import importExpt, importFile

    r2vals = []
    
    expt_q, expt_ints = importExpt(exptPath, exptFN, exptWL, maxTT)
    
    sims = glob.glob('./simulations/*.txt')
    
    for f in sims:
        removeExt = f.split('.')
        fn = removeExt[0].split('\\')
        
        q, ints = importFile('./simulations/', fn[-1])
        r2 = r2val(expt_q, q, expt_ints, ints)
        
        r2vals.append([fn[-1], r2])
        
    with open('./r2vals.txt', 'w') as x:
        for i in range(len(r2vals)):
            for (fn, r2) in zip(r2vals[i][0], r2vals[i][1]):
                x.write('{0} {1}\n'.format(fn, r2))
        x.close()

    return r2vals



#-------------------------------------
#----- FUNCTION: stepGridSearch ------
#-------------------------------------
def stepGridSearch(pRange, sxRange, syRange):
    """
    Generates a step-wise set of stacking vectors and fault probabilities

    Parameters
    ----------
    pRange : nparray
        List of minimum fault probability, maximum fault probability, and step size
    sxRange : nparray
        List of minimum stacking vector x-component, maximum stacking vector x-component, and step size
    syRange : nparray
        List of minimum stacking vector y-component, maximum stacking vector y-component, and step size

    Returns
    -------
    pList : nparray
        Set of fault probabilities
    sList : nparray
        Set of stacking vectors
    """
    # generate fault probabilities
    p = pRange[0]
    pList = []
    while p <= pRange[1]:
        pList.append(round(p, 3))
        p = p + pRange[2]
    
    # generate stacking vectors
    sx = sxRange[0]
    sy = syRange[0]
    sxList = []
    syList = []
    sList = []
    while sx <= sxRange[1]:
        sxList.append(round(sx, 5))
        sx = sx + sxRange[2]
    while sy <= syRange[1]:
        syList.append(round(sy, 5))
        sy = sy + syRange[2]
    for i in range(len(sxList)):
        for j in range(len(syList)):
            s = [sxList[i], syList[j], 0]
            sList.append(s)
            
    return np.array(pList), np.array(sList)



#-------------------------------------
#----- FUNCTION: randGridSearch ------
#-------------------------------------
def randGridSearch(pRange, sxRange, syRange, numVec):
    """
    Generates a random set of stacking vectors and fault probabilities

    Parameters
    ----------
    pRange : nparray
        List of minimum fault probability, maximum fault probability, and step size
    sxRange : nparray
        List of minimum stacking vector x-component and maximum stacking vector x-component
    syRange : nparray
        List of minimum stacking vector y-component and maximum stacking vector y-component
    numVec : int
        Number of randomized stacking vectors to generate

    Returns
    -------
    pList : nparray
        Set of fault probabilities
    sList : nparray
        Set of stacking vectors
    """
    # generate fault probabilities
    p = pRange[0]
    pList = []
    while p <= pRange[1]:
        pList.append(round(p, 3))
        p = p + pRange[2]
    
    # generate stacking vectors
    sList = []
    for i in range(numVec):
        sx = random.randrange(sxRange[0], sxRange[1])
        sy = random.randrange(syRange[0], syRange[1])
        sList.append(sx, sy, 0)
    
    return np.array(pList), np.array(sList)