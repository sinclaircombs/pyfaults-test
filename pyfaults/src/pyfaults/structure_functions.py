""" 
structure_functions.py

Module containing functions related to building unit cell or supercell structures

toCif --> exports CIF file
getLayers --> Imports layer information from formatted DataFrame and returns a list of Layer objects
importCSV --> generates a unit cell from a CSV of formatted atomic parameters
--------------
CSV FORMATTING
--------------
Layer, Atom, Element, x, y, z, Occupancy, Biso
A, H1, H1+, 0, 0, 0, 1, 2.0
A, H2, H1+, 0.5, 0.5, 0, 1, 2.0
...
genSupercells --> Generates Supercell instances for all possible combinations in a defined parameter space and exports CIFs in new 'supercells' directory
"""

#---------- import packages ----------
import pandas as pd
import os



#-------------------------------------
#---------- FUNCTION: toCif ----------
#-------------------------------------
def toCif(cell, path, filename):
    """
    Generates CIF of a unit cell or supercell structure

    Parameters
    ----------
    cell : Unitcell or Supercell
        Unit cell or supercell structure to convert to CIF format
    path : str
        File directory to save CIF
    filename : str
        Name of CIF file
    """
    
    lines = []
    # space group info
    lines.extend([
        '%-31s %s' % ('_symmetry_space_group_name_H-M', 'P1'),
        '%-31s %s' % ('_symmetry_Int_Tables_number', '1'),
        '%-31s %s' % ('_symmetry_cell_setting', 'triclinic'),
        ''])
    
    # lattice parameters
    lines.extend([
        '%-31s %.6g' % ('_cell_length_a', cell.lattice.a),
        '%-31s %.6g' % ('_cell_length_b', cell.lattice.b),
        '%-31s %.6g' % ('_cell_length_c', cell.lattice.c),
        '%-31s %.6g' % ('_cell_angle_alpha', cell.lattice.alpha),
        '%-31s %.6g' % ('_cell_angle_beta', cell.lattice.beta),
        '%-31s %.6g' % ('_cell_angle_gamma', cell.lattice.gamma),
        ''])
    
    # symmetry operations
    lines.extend([
        'loop_',
        '_space_group_symop_operation_xyz',
        '  \'x, y, z\' ',
        ''])
    
    # loop info
    lines.extend([
        'loop_',
        '  _atom_site_label',
        '  _atom_site_type_symbol',
        '  _atom_site_fract_x',
        '  _atom_site_fract_y',
        '  _atom_site_fract_z',
        '  _atom_site_B_iso_or_equiv',
        '  _atom_site_adp_type',
        '  _atom_site_occupancy' ])

    # atoms
    for lyr in cell.layers:
        for a in lyr.atoms:
            label = a.atomLabel
            elem = a.element
            x = a.x
            y = a.y
            z = a.z
            occ = a.occupancy
            biso = a.biso
            aline = ' %-5s %-3s %11.6f %11.6f %11.6f %11.6f %-5s %.4f' % (label, elem, x, y, z, biso, 'Biso', occ)
            lines.append(aline)

    with open(path + filename + '.cif', 'w') as cif:
        for i in lines:
            cif.write(i + '\n')
    cif.close()
    
    return



#-------------------------------------
#-------- FUNCTION: getLayers --------
#-------------------------------------
def getLayers(df, lattice, layerNames):
    """
    Imports layer information from formatted DataFrame and returns a list of Layer objects

    Parameters
    ----------
    df : DataFrame
        Pandas DataFrame containing formatted layer information
    lattice : Lattice
        Unit cell lattice parameters
    layerNames : list of str
        Unique identifiers for layers, must match those defined in DataFrame

    Returns
    -------
    list of Layer
        Layer objects generated from DataFrame information
    """
    
    from pyfaults.structure_classes import LayerAtom, Layer, Lattice
    
    # generate new Lattice object
    newLatt = Lattice(a=lattice.a, 
                          b=lattice.b, 
                          c=lattice.c,
                          alpha=lattice.alpha, 
                          beta=lattice.beta, 
                          gamma=lattice.gamma)

    layers = []
    # loop through definied layer names
    for i in range(len(layerNames)):
        alist = []
        # loop through dataframe rows
        for index, row in df.iterrows():
            # if row corresponds to current layer name
            if row['Layer'] == layerNames[i]:
                # grab atomic position values
                xyz = [row['x'], row['y'], row['z']]
                    
                # create new LayerAtom instance
                newAtom = LayerAtom(layerNames[i], 
                                    row['Atom'], 
                                    row['Element'], 
                                    xyz, 
                                    row['Occupancy'],
                                    row['Biso'],
                                    newLatt)
                # add new atom to list of layer atoms
                alist.append(newAtom)
                
        # create new Layer instance
        newLayer = Layer(alist, newLatt, layerNames[i])
        layers.append(newLayer)
    return layers



#-------------------------------------
#-------- FUNCTION: importCSV --------
#-------------------------------------
def importCSV(path, filename, lattParams, lyrNames):
    """
    Generates a new Unitcell instance from CSV containing atomic parameters

    Parameters
    ----------
    path : str
        File path of directory where CSV is stored
    filename : str
        Name of CSV file
    lattParams : nparray
        Unit cell lattice parameters formatted as [a, b, c, alpha, beta, gamma]
    lyrNames : list of str
        List of unique identifiers for layers, must match those defined in CSV

    Returns
    -------
    unitcell : Unitcell
        Instance of Unitcell generated with parameters from CSV
    """

    from pyfaults.structure_classes import Unitcell, Lattice

    csv = pd.read_csv(path + filename + '.csv')
    
    latt = Lattice(a=lattParams[0],
                              b=lattParams[1],
                              c=lattParams[2],
                              alpha=lattParams[3],
                              beta=lattParams[4],
                              gamma=lattParams[5])
    
    lyrs = getLayers(csv, latt, lyrNames)
    
    unitcell = Unitcell(filename, lyrs, latt)
    unitcell.toCif(path)
    
    return unitcell


#-------------------------------------
#------ FUNCTION: genSupercells ------
#-------------------------------------
def genSupercells(unitcell, nStacks, fltLayer, probList, sVecList):
    """
    Generates Supercell instances within a defined parameter space and exports corresponding CIFs

    Parameters
    ----------
    unitcell : Unitcell
        Unit cell used to construct supercell
    nStacks : int
        Number of unit cells stacked to generate supercell
    fltLayer : str
        Name of layer to apply stacking fault parameters to
    probList : list of float
        List of probabilities of stacking fault occurrence, defines one dimension of parameter space
    sVecList : list of nparray
        List of in-plane displacement vector components in [x,y] format and fractional coordinates, defines one dimension of parameter space
    """

    from pyfaults.structure_classes import Supercell
    
    # create 'supercells' folder in working directory
    if os.path.exists('./supercells/') == False:
        os.mkdir('./supercells/')
    
    cellList = []
    
    # generate unfaulted supercell
    UF = Supercell(unitcell, nStacks)
    cellList.append([UF, 'Unfaulted'])
    
    # generate faulted supercells over parameter space
    for p in range(len(probList)):
        for s in range(len(sVecList)):
            FLT = Supercell(unitcell, nStacks, fltLayer=fltLayer, stackVec=sVecList[s], stackProb=probList[p])
            # creates file name tag with vector number and probability percentage
            cellTag = 'S' + str(s+1) + '_P' + str(int(probList[p]*100))
            cellList.append([FLT, cellTag])
    
    # export CIF for each supercell
    for c in range(len(cellList)):
        toCif((cellList[c][0]), './supercells/', cellList[c][1])
    
    return               