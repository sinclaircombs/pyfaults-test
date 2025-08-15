"""
structure_classes.py
----------
Unitcell --> An object representing the unit cell of a crystal structure
Supercell --> An object constructed by repeatedly stacking individual unit cells along the c-axis and subsequently used to create stacking fault models

----------
Structure properties are stored in the following lower level classes:

LayerAtom --> Contains information about a specific atom in a unit cell or supercell layer
Layer --> Contains information about a layer in a unit cell or supercell
Lattice --> Contains unit cell lattice parameters
"""

#---------- import packages ----------
import copy as cp
import numpy as np
import random as r



#-------------------------------------
#---------- CLASS: Unitcell ----------
#-------------------------------------
class Unitcell(object):

    #---------- properties ----------
    name = property(lambda self: self._name, lambda self, val: self.setParam(name=val),
                    doc='str : Unique identifier for Unitcell object')
    
    layers = property(lambda self: self._layers, lambda self, val: self.setParam(layers=val),
                      doc='list of Layer : List of named layers that make up the unit cell')
    
    lattice = property(lambda self: self._lattice, lambda self, val: self.setParam(lattice=val),
                       doc='Lattice : Lattice object describing lattice parameters of the unit cell')
    
    #---------- functions ----------
    def __init__(self, name, layers, lattice):
        """
        Initializes a new instance of Unitcell

        Parameters
        ----------
        name : str
            Unique identifier for Unitcell object
        layers : list of Layer
            List of named layers that make up the unit cell
        lattice : Lattice
            Lattice object describing lattice parameters of the unit cell
        """
        self._name = None
        self._layers = None
        self._lattice = None

        # feed input parameters to setParam method
        self.setParam(name, layers, lattice)
        return
    
    def setParam(self, *, name=None, layers=None, lattice=None):
        """
        Sets parameters of Unitcell object from __init__ parameters
        """
        if name is not None:
            self._name = name
        if layers is not None:
            self._layers = layers
        if lattice is not None:
            self._lattice = lattice
        return
    
    def info(self):
        """
        Prints information about the unit cell
        """
        print("Name: " + self.name)
        print("----------")
        print("a: " + self.lattice.a)
        print("b: " + self.lattice.b)
        print("c: " + self.lattice.c)
        print("alpha: " + self.lattice.alpha)
        print("beta: " + self.lattice.beta)
        print("gamma: " + self.lattice.gamma)
        print("----------")
        layerList = self.layers
        layerStr = []
        for i in range(len(layerList)):
            layerStr.append(layerList[i].layerName)
        print("Layers: " + layerStr)
        return
    


#-------------------------------------
#--------- CLASS: Supercell ----------
#-------------------------------------
class Supercell(object):
    
    #---------- properties ----------
    unitcell = property(lambda self: self._unitcell,
                        doc='Unitcell : Unit cell used to construct supercell')
    
    lattice = property(lambda self: self._lattice,
                       doc='Lattice : Lattice parameters of unit cell')
    
    nStacks = property(lambda self: self._nStacks,
                       doc='int : Number of unit cells stacked to generate supercell')
    
    layers = property(lambda self: self._layers,
                      doc='list of Layer : List of named layers that make up the supercell')
    
    fltLayer = property(lambda self: self._fltLayer,
                        doc='str : Name of layer to apply stacking fault parameters to')
    
    stackVec = property(lambda self: self._stackVec,
                        doc='nparray : In-plane displacement vector components in [x,y] format')
    
    stackProb = property(lambda self: self._stackProb, lambda self, val: self.setParam(stackProb=val),
                         doc='float : Probability stacking fault will occur')
        
    zAdj = property(lambda self: self._zAdj, lambda self, val: self.setParam(zAdj=val),
                    doc='float : Out-of-plane displacement vector component (z)')
        
    intLayer = property(lambda self: self._intLayer, lambda self, val: self.setLayers(intLayer=val),
                        doc='Layer : Layer to be inserted as an intercalation layer')
    
    #---------- functions ----------
    def __init__(self, unitcell, nStacks, *, fltLayer=None, stackVec=[0,0], stackProb=0.0, zAdj=0, intLayer=None):
        """
        Initializes new instance of Supercell

        Parameters
        ----------
        unitcell : Unitcell
            Unit cell used to construct supercell
        nStacks : int
            Number of unit cells stacked to generate supercell
        fltLayer : str, optional
            Name of layer to apply stacking fault parameters to, by default None
        stackVec : nparray, optional
            In-plane displacement vector components in [x,y] format, by default [0,0]
        stackProb : float, optional
            Probability stacking fault will occur, by default 0.0
        zAdj : float, optional
            Out-of-plane displacement vector component (z), by default 0
        intLayer : Layer, optional
            Layer to be inserted as an intercalation layer, by default None
        """
        
        from pyfaults.lattice import Lattice
        self._unitcell = unitcell
        # redefines length of c based on number of stacks
        newLatt = Lattice(unitcell.lattice.a,
                          unitcell.lattice.b,
                          (unitcell.lattice.c * nStacks),
                          unitcell.lattice.alpha,
                          unitcell.lattice.beta,
                          unitcell.lattice.gamma)
        self._lattice = newLatt
        self._nStacks = None
        self._layers = None
        self._fltLayer = None
        self._stackVec = None
        self._stackProb = None
        self._zAdj = None
        self._intLayer = None

        assignProb = self.assignProb()
        countFaults = self.countFaults(assignProb)
        self.adjustForZ(countFaults)
        self.generateLayers(assignProb)

        # feed parameters to setParam method
        self.setParam(nStacks, fltLayer, stackVec, stackProb, zAdj, intLayer)
        return
    
    def setParam(self, *, nStacks=None, fltLayer=None, stackVec=None, stackProb=None, zAdj=None, intLayer=None):
        """
        Sets parameters of Supercell object from __init__ parameters
        """
        if nStacks is not None:
            self._nStacks = nStacks
        if fltLayer is not None:
            self._fltLayer = fltLayer
        if stackVec is not None:
            self._stackVec = stackVec
        if stackProb is not None:
            self._stackProb = stackProb
        if zAdj is not None:
            self._zAdj = zAdj
        if intLayer is not None:
            self._intLayer = intLayer
        return
    
    def assignProb(self):
        """ Generates a random integer from 0 to 100 for each unit cell stack in the supercell; used in implementing fault probability

        Returns
        -------
        assignProb : list of int
            Randomly generated probability values for each unit cell stack in supercell
        """
        assignProb = []
        while n < self.nStacks:
            assignProb.append(r.randint(0,100))
            n += 1
        return assignProb
    
    def countFaults(self, assignProb):
        """
        Counts the total number of faulted stacks across all unit cells in supercell

        Parameters
        ----------
        assignProb : list of int
            Randomly generated probability values for each unit cell stack in supercell

        Returns
        -------
        countFaults
            Total number of stacks containing stacking faults
        """
        countFaults = 0
        stackProbPercent = self.stackProb*100
        for n in range(len(assignProb)):
            if assignProb[n] <= stackProbPercent:
                countFaults += 1
        return countFaults
    
    def adjustForZ(self, countFaults):
        """
        Adjusts lattice vector c for number of z-direction displacements

        Parameters
        ----------
        countFaults : int
            Total number of stacks containing stacking faults
        """
        self.lattice.c = self.lattice.c + (self.zAdj*countFaults)

    def generateLayers(self, assignProb):
        """
        Generates supercell layers based on stacking fault parameters

        Parameters
        ----------
        assignProb : list of int
            Randomly generated probability values for each unit cell stack in supercell
        """
        newLayers = []

        while n < self.nStacks:
            # tag denotes which stack layer belongs to
            tag = '_n' + str(n+1)

            for lyr in self.unitcell.layers:
                newLayer = cp.deepcopy(lyr)
                newLayer.setParam(lattice=self.lattice)

                stackProbPercent = self.stackProb*100
                if assignProb[n] <= stackProbPercent and lyr.layerName == self.fltLayer:
                    newLayer.setParam(layerName=lyr.layerName + tag + '_fault')
                    newLayers.append(self.adjustAtomPos(newLayer, n, True))

                    if self.intLayer is not None:
                        newIntLayer = self.addIntercalationLayer(n, tag)
                        newLayers.append(newIntLayer)
                
                else:
                    newLayer.setParam(layerName=lyr.layerName + tag)
                    newLayers.append(self.adjustAtomPos(newLayer, n, True))
            n += 1

        self._layers = newLayers
        return

    def adjustAtomPos(self, layer, nCurrent, isFaulted):
        """
        Adjusts [x,y,z] position of atoms based on position in supercell and stacking fault parameters

        Parameters
        ----------
        layer : Layer
            Layer containing atoms to be adjusted
        nCurrent : int
            Number of current stack in supercell
        isFaulted : bool
            Set to True if current stack is faulted, set to False if not faulted

        Returns
        -------
        layer : Layer
            Layer with adjusted atomic positions
        """
        for atom in layer.atoms:
            atomLabel = atom.atomLabel.split('_')

            if isFaulted == True:
                position = [atom.x, atom.y, ((atom.z + nCurrent + self.zAdj) / self.nStacks)]
                position = np.add(position, self.stackVec)
                tag = layer.layerName + '_n' + str(nCurrent+1) + '_fault'

            elif isFaulted == False:
                position = [atom.x, atom.y, ((atom.z + nCurrent) / self.nStacks)]
                tag = layer.layerName + '_n' + str(nCurrent+1)
            
            atom.setParam(layerName=tag, atomLabel=atomLabel[0], xyz=position, lattice=self.lattice)
        return layer

    def addIntercalationLayer(self, n, tag):
        """
        Inserts an intercalation layer at fault location

        Parameters
        ----------
        n : int
            Stack location in supercell
        tag : str
            Denotes which stack intercalation layer belongs to

        Returns
        -------
        Layer
            Intercalation layer to be inserted into supercell
        """
        newLayer = cp.deepcopy(self.intLayer)
        newLayer.setParam(layerName='I' + tag, lattice=self.lattice)
        
        for atom in newLayer.atoms:
            alabel = atom.atomLabel.split('_')
            newXYZ = [atom.x, atom.y, ((atom.z + n) / self.nStacks)]
            
            atom.setParam(layerName='I' + tag, atomLabel=alabel[0], xyz=newXYZ, lattice=self.lattice)
        return newLayer
    
    def info(self):
        """
        Prints information about the supercell
        """
        print("a: " + self.lattice.a)
        print("b: " + self.lattice.b)
        print("c: " + self.lattice.c)
        print("alpha: " + self.lattice.alpha)
        print("beta: " + self.lattice.beta)
        print("gamma: " + self.lattice.gamma)
        print("----------")
        print("Stacking Vector: " + str(self.stackVec))
        print("----------")
        print("Fault Probability: " + str(self.stackProb))
        print("----------")
        layerStr = []
        for lyr in self.layers:
            if lyr.layerName.contains("fault"):
                getCurrN = lyr.layerName.split('_')
                layerStr.append(getCurrN[1])
        print("Faulted Layers: " + layerStr)
        return



#-------------------------------------
#--------- CLASS: LayerAtom ----------
#-------------------------------------
class LayerAtom(object):

    #---------- properties ----------
    layerName = property(lambda self: self._layerName, lambda self, val: self.setParam(layerName=val),
                         doc='str : Unique identifier for layer containing LayerAtom')
        
    atomLabel = property(lambda self: self._atomLabel, lambda self, val: self.setParam(atomLabel=val),
                         doc='str : Unique identifier for LayerAtom object')
        
    element = property(lambda self: self._element, lambda self, val: self.setParam(element=val),
                       doc='str : Chemical element abbreviation and oxidation state')
        
    xyz = property(lambda self: self._xyz, lambda self, val: self.setParam(xyz=val),
                   doc='nparray : Position of atom in fractional coordinates of unit cell vectors')
        
    x = property(lambda self: self._xyz[0], lambda self, val: self.setParam(0, val),
                 doc='float : x-component of atomic position')
        
    y = property(lambda self: self._xyz[1], lambda self, val: self.setParam(1, val),
                 doc='float : y-component of atomic position')
        
    z = property(lambda self: self._xyz[2], lambda self, val: self.setParam(2, val),
                 doc='float : z-component of atomic position')
        
    occupancy = property(lambda self: self._occupancy, lambda self, val: self.setParam(occupancy=val),
                         doc='float : Site occupancy, must be greater than zero and maximum 1')
        
    biso = property(lambda self: self._biso, lambda self, val: self.setParam(biso=val),
                    doc='float : Isotropic atomic displacement parameter (B-factor) in units of square Angstroms')
        
    lattice =  property(lambda self: self._lattice, lambda self, val: self.setParam(lattice=val),
                        doc='Lattice : Unit cell lattice parameters')
    
    #---------- functions ----------
    def __init__(self, layerName, atomLabel, element, xyz, occupancy, biso, lattice):
        """
        Initializes a new instance of LayerAtom

        Parameters
        ----------
        layerName : str
            Unique identifier for layer containing LayerAtom
        atomLabel : str
            Unique identifier for LayerAtom object
        element : str
            Chemical element abbreviation and oxidation state
        xyz : nparray
            Position of atom in fractional coordinates of unit cell vectors
        occupancy : float
            Site occupancy, must be greater than zero and maximum 1
        biso : float
            Isotropic atomic displacement parameter (B-factor) in units of square Angstroms
        lattice : Lattice
            Unit cell lattice parameters
        """

        self._layerName = None
        self._atomLabel = None
        self._element = None
        self._xyz = None
        self._x = None
        self._y = None
        self._z = None
        self._lattice = None
        self._occupancy = None
        self._biso = None

        # feed input parameters to setParam method
        self.setParam(layerName, atomLabel, element, xyz, lattice, occupancy, biso)
        return
    
    def setParam(self, *, layerName=None, atomLabel=None, element=None, xyz=None, lattice=None, occupancy=None, biso=None):
        """
        Sets parameters of Unitcell object from __init__ parameters
        """
        if layerName is not None:
            self._layerName = layerName
        if atomLabel is not None:
            self._atomLabel = atomLabel + '_' + layerName
        if element is not None:
            self._element = element
        if xyz is not None:
            self._xyz = xyz
            # define individual parameters for x-, y-, and z-components of position
            self._x = xyz[0]
            self._y = xyz[1]
            self._z = xyz[2]
        if lattice is not None:
            self._lattice = lattice
        if occupancy is not None:
            self._occupancy = occupancy
        if biso is not None:
            self._biso = biso
        return
    


#-------------------------------------
#---------- CLASS: Layer -------------
#-------------------------------------
class Layer(object):

    #---------- properties ----------
    atoms = property(lambda self: self._atoms, lambda self, val: self.setParam(atoms=val),
                     doc='list of LayerAtom : All atoms contained in the layer')
    
    lattice = property(lambda self: self._lattice, lambda self, val: self.setParam(lattice=val),
                       doc='Lattice : Unit cell lattice parameters')
    
    layerName = property(lambda self: self._layerName, lambda self, val: self.setParam(layerName=val),
                         doc='str : Unique identifier for Layer object')
    
    #---------- functions ----------
    def __init__(self, atoms, lattice, layerName):
        """
        Initializes new instance of Layer

        Parameters
        ----------
        atoms : list of LayerAtom
            All atoms contained in the layer
        lattice : Lattice
             Unit cell lattice parameters
        layerName : str
            Unique identifier for Layer object
        """
        self._layerName = None
        self._atoms = None
        self._lattice = None

        # feed input parameters to setParam method
        self.setParam(atoms, lattice, layerName)
        return
    
    def setParam(self, atoms=None, lattice=None, layerName=None):
        """
        Sets parameters of Layer object from __init__ parameters
        """
        if atoms is not None:
            self._atoms = atoms
        if lattice is not None:
            self._lattice = lattice
        if layerName is not None:
            self._layerName = layerName
        return
    
    def genChildLayer(self, childName, transVec):
        """
        Generates a copy (child) of a Layer displaced by a give translation vector; useful for defining layers with identical atomic compositions and positions at different locations in the unit cell

        Parameters
        ----------
        childName : str
            Unique identifier for newly generated child layer
        transVec : nparray
            Translation vector to apply to original parent layer position to generate child layer

        Returns
        -------
        childLayer : Layer
            New child layer generated from parent
        """
    
        childAtoms = []
        # loop through all atoms in parent layer
        for a in self.atoms:
            # create copy of atom
            pAtom = cp.deepcopy(a)
            splitLabel = pAtom.atomLabel.split('_')
            
            # apply translation vector to atomic position
            newPos = np.add(pAtom.xyz, transVec)
            for i in range(len(newPos)):
                if newPos[i] >= 1:
                    newPos[i] = newPos[i] - 1
                    
            # create new LayerAtom instance
            cAtom = LayerAtom(childName, 
                              splitLabel[0], 
                              pAtom.element, 
                              newPos, 
                              pAtom.occupancy,
                              pAtom.biso,
                              self.lattice)
            # add new atom to list of atoms in child layer
            childAtoms.append(cAtom)
            
        # create child layer with new Layer instance
        childLayer = Layer(childAtoms, self.lattice, childName)
        return childLayer
    


#-------------------------------------
#--------- CLASS: Lattice ------------
#-------------------------------------
class Lattice(object):

    #---------- properties ----------
    a = property(lambda self: self._a, lambda self, val: self.setParam(a=val),
                 doc='float : Unit cell vector a in units of Angstroms')
    
    b = property(lambda self: self._b, lambda self, val: self.setParam(b=val),
                 doc='float : Unit cell vector b in units of Angstroms')
    
    c = property(lambda self: self._c, lambda self, val: self.setParam(c=val),
                 doc='float : Unit cell vector c in units of Angstroms')
    
    alpha = property(lambda self: self._alpha, lambda self, val: self.setParam(alpha=val),
                     doc='float : Unit cell angle alpha (angle between vectors b and c) in units of degrees')
    
    beta = property(lambda self: self._beta, lambda self, val: self.setParam(beta=val),
                    doc='float : Unit cell angle beta (angle between vectors a and c) in units of degrees')
    
    gamma = property(lambda self: self._gamma, lambda self, val: self.setParam(gamma=val),
                     doc='float : Unit cell angle gamma (angle between vectors a and b) in units of degrees')
    
    #---------- functions ----------
    def __init__(self, a, b, c, alpha, beta, gamma):
        """
        Initializes a new instance of Lattice

        Parameters
        ----------
        a : float
            Unit cell vector a in units of Angstroms
        b : float
            Unit cell vector b in units of Angstroms
        c : float
            Unit cell vector c in units of Angstroms
        alpha : float
            Unit cell angle alpha (angle between vectors b and c) in units of degrees
        beta : float
            Unit cell angle beta (angle between vectors a and c) in units of degrees
        gamma : float
            Unit cell angle gamma (angle between vectors a and b) in units of degrees
        """
        self._a = None
        self._b = None
        self._c = None
        self._alpha = None
        self._beta = None
        self._gamma = None

        # feed input parameters to setParam method
        self.setParam(a, b, c, alpha, beta, gamma)
        return
    
    def setParam(self, *, a=None, b=None, c=None, alpha=None, beta=None, gamma=None):
        """
        Sets parameters of Lattice object from __init__ parameters
        """
        if a is not None:
            self._a = a
        if b is not None:
            self._b = b
        if c is not None:
            self._c = c
        if alpha is not None:
            self._alpha = alpha
        if beta is not None:
            self._beta = beta
        if gamma is not None:
            self._gamma = gamma
        return