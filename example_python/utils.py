import numpy as np

def qsat(t,p):
    ''' TAKEN FROM COARE PACKAGE. Usage: es = qsat(t,p)
        Returns saturation vapor pressure es (mb) given t(C) and p(mb).
        After Buck, 1981: J.Appl.Meteor., 20, 1527-1532
        Returns ndarray float for any numeric object input.
    '''

    t2 = np.copy(np.asarray(t, dtype=float))  # convert to ndarray float
    p2 = np.copy(np.asarray(p, dtype=float))
    es = 6.1121 * np.exp(17.502 * t2 / (240.97 + t2))
    es = es * (1.0007 + p2 * 3.46e-6)
    return es

def rhcalc(t,p,q):
    ''' TAKEN FROM COARE PACKAGE. usage: rh = rhcalc(t,p,q)
        Returns RH(%) for given t(C), p(mb) and specific humidity, q(kg/kg)
        Returns ndarray float for any numeric object input.
    '''
    
    q2 = np.copy(np.asarray(q, dtype=float))    # conversion to ndarray float
    p2 = np.copy(np.asarray(p, dtype=float))
    t2 = np.copy(np.asarray(t, dtype=float))
    es = qsat(t2,p2)
    em = p2 * q2 / (0.622 + 0.378 * q2)
    rh = 100.0 * em / es
    return rh