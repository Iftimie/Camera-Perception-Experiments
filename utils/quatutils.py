import numpy as np
import math
#######################################################################################################
#######################################################################################################
# taken from DMS Cam Capture starting from DmsCamCapture.cpp line 5862
X = 0
Y = 1
Z = 2
W = 3
EulParEven = 0
EulFrmR = 1
EulParOdd = 1
EulRepNo = 0 
EulFrmS = 0

def EulOrd(i,p,r,f):
    return (((((((i)<<1)+(p))<<1)+(r))<<1)+(f))

def EulOrdYXZr():
    return EulOrd(Z,EulParEven,EulRepNo,EulFrmR)

def EulOrdYXZs():
    return EulOrd(Y,EulParOdd,EulRepNo,EulFrmS)

def EulOrdZXYr():
    return EulOrd(Y,EulParOdd,EulRepNo,EulFrmR)

def EulOrdZXYs():
    return EulOrd(Z,EulParEven,EulRepNo,EulFrmS)

def Eul_FromQuat(q, order):
    M = np.zeros((4,4), np.float32)
    Nq = q[0]*q[0]+q[1]*q[1]+q[2]*q[2]+q[3]*q[3]
    s = (2.0 / Nq) if (Nq > 0.0) else 0.0
    xs = q[0]*s
    ys = q[1]*s
    zs = q[2]*s
    wx = q[3]*xs
    wy = q[3]*ys
    wz = q[3]*zs
    xx = q[0]*xs
    xy = q[0]*ys
    xz = q[0]*zs
    yy = q[1]*ys
    yz = q[1]*zs
    zz = q[2]*zs
    M[X][X] = 1.0 - (yy + zz)
    M[X][Y] = xy - wz
    M[X][Z] = xz + wy
    M[Y][X] = xy + wz
    M[Y][Y] = 1.0 - (xx + zz)
    M[Y][Z] = yz - wx
    M[Z][X] = xz - wy
    M[Z][Y] = yz + wx
    M[Z][Z] = 1.0 - (xx + yy)
    M[W][X]=M[W][Y]=M[W][Z]=M[X][W]=M[Y][W]=M[Z][W]=0.0
    M[W][W]=1.0
    return Eul_FromHMatrix(M, order)

EulSafe = [0,1,2,0]
EulNext = [1,2,0,1]
def EulGetOrd(order):
    o=order
    f=o&1
    o>>=1
    s=o&1
    o>>=1
    n=o&1
    o>>=1
    i=EulSafe[o&3]
    j=EulNext[i+n]
    k=EulNext[i+1-n]
    h = k if s!=0 else i
    return i,j,k,h,n,s,f

EulRepYes  = 1
EulRepNo   = 0
FLT_EPSILON = np.finfo(float).eps
EulParOdd  = 1
def Eul_FromHMatrix(M, order):
    ea = np.zeros(4, np.float32)
    i,j,k,h,n,s,f = EulGetOrd(order)
    if (s==EulRepYes):
        sy = math.sqrt(M[i][j]*M[i][j] + M[i][k]*M[i][k])
        if (sy > 16*FLT_EPSILON):
            ea[0] = math.atan2(M[i][j], M[i][k])
            ea[1] = math.atan2(sy, M[i][i])
            ea[2] = math.atan2(M[j][i], -M[k][i])
        else:
            ea[0] = math.atan2(-M[j][k], M[j][j])
            ea[1] = math.atan2(sy, M[i][i])
            ea[2] = 0
    else:
        cy = math.sqrt(M[i][i]*M[i][i] + M[j][i]*M[j][i])
        if (cy > 16*FLT_EPSILON):
            ea[0] = math.atan2(M[k][j], M[k][k])
            ea[1] = math.atan2(-M[k][i], cy)
            ea[2] = math.atan2(M[j][i], M[i][i])
        else:
            ea[0] = math.atan2(-M[j][k], M[j][j])
            ea[1] = math.atan2(-M[k][i], cy)
            ea[2] = 0
    if (n==EulParOdd):
        ea[0] = -ea[0]
        ea[1] = - ea[1]
        ea[2] = -ea[2]
    if (f==EulFrmR): 
        t = ea[0]
        ea[0] = ea[2]
        ea[2] = t
    ea[0] = math.degrees(ea[0])
    ea[1] = math.degrees(ea[1])
    ea[2] = math.degrees(ea[2])
    ea[3] = order
    return ea
#######################################################################################################
#######################################################################################################