import matplotlib.pyplot as plt
import numpy as np
import math
from scipy import fftpack


default = '01-027/i15-1-18935_tth_det2_0.'

class PDFSelfScatteringBKG:
    def getBkg(self, r, dR, rMin, qt):
        # Soper[2009](53,54)
        pofr = self.getPofR(r,qt)
        bR   = np.zeros(r.size)
        g0r  = 0.0 # Soper[2009]P.1675 L.1
        for i in range(r.size):
            if(r[i]<rMin):
                bR[i]=dR[i]-g0r+1.0
            else:
                f     = pofr[i]
                bR[i] = -dR[i]*f/(1-f)
        return r, bR      
    
    def getPofR(self, r, qt):
        # Soper[2009](49)
        x         = np.multiply(r,qt)
        pofr      = lambda x: 3*(math.sin(x)-x*math.cos(x))/(x**3)
        pofrArray = np.array([pofr(i) for i in x ])
        return pofrArray
    
class PDFTophatFunction:
    def findIndex(self, inq,qt):
        #find the point near the qt
        for i,q in enumerate(inq):
            if(q>qt):
                return i-1
        return -1

    def interVolume(self, l,m,n,factv):
        # Soper[2009](A.2)
        # (2 - delta)(12l^2 + 1)/(2n + 1)^3
        value = (12*(l**2)+1)*factv
        if(l==0):
            return value
        else:
            return 2*value
        pass
    
    def outerVolume(self, l,m,n, factv, m2, factm):
        # Soper[2009](A.2)
        # (12l^2 + 1 - 3l*[4l^2 + 4m^2 - (2n + 1)^2 + 1]/2m)/(2n + 1)^3
        l2 = l**2
        return factv*(12*l2+1-3*l*(4*l2+factm)/m2)
    
    def getVolume(self, m, n, qList,factv, m2, factm):
        # Soper[2009](A.2)
        mmn = abs(m - n)
        result = {}
        for i, l in enumerate(qList):
            if(l < mmn): # 0 <= l < |m-n| from 0 rather than 1 in Python
                result[l]=self.interVolume(l,m,n,factv)
            else: # |m-n| <= l <= m+n
                result[l]=self.outerVolume(l,m,n,factv,m2,factm) 
        return result        
    
    def tophatConvolution(self, cutoffQt, inBins, inHist):
        # Soper[2009](A.3)
        n  = self.findIndex(inBins, cutoffQt)
        n2 = 2*n+1
        factv=1/(n2**3)
        origSize = inHist.size
        outHist = np.array([0.0 for i in range(origSize)], dtype=np.float64)
        
        for m in range(origSize):
            m2 = 2*m
            #factt = factv/m2
            factm = m2**2 - n2**2 + 1
            
            # the range [minQ+m, maxQ+m] of Q' with convolution
            minQ  = max(-n,-m)
            maxQ  = n
            minL  = minQ + m
            maxL  = maxQ + m
            #enumerate(xout)
            qList = [q for q in range(minL, maxL)]
            vList = self.getVolume(m,n,qList,factv, m2, factm)
            for k in range(minQ,maxQ):
                index = m+k
                if(index<origSize):
                    outHist[m] = outHist[m] + inHist[index]*vList[index]
                else: 
                    outHist[m] = outHist[m] + inHist[m]*vList[index]
        return inBins, outHist        

class PDFWorkspace:
    def __init__(self, default='01-027/i15-1-18935_tth_det2_0.'):
        self.default = default
        
    def loadData(self, filename=None, suffix=None):
        if((filename is None)and(suffix is not None)):
            filename = self.default+suffix
        print('Open file name: ' + filename)
        
        f=open(filename)
        l=f.readlines()
        x_array = []
        y_array = []
        for line in l:
            if(line[0].startswith("#")):
                continue
            i = line.split()
            if(len(i)<3):
                continue
            x_array.append(float(i[0]))
            y_array.append(float(i[1]))
        return np.array(x_array), np.array(y_array)
    
    def zeroGrFromQ(self, q):
        rstep = math.pi/q.max()
        nstep = len(q)
        r     = np.array([i for i in range(0,nstep)])
        r     = np.multiply(rstep, r)
        gR    = np.zeros(nstep, dtype=np.float64)
        return r, gR
    
    
    def iQExtend2lowQ(self, inx, iny):
        ex, ey = self.lowQEstimation(inx, iny)
        return np.append(ex, inx), np.append(ey, iny)
        
                 
    def lowQEstimation(self, inx,iny):
        step  = abs(inx[-1]-inx[0])/inx.size
        nstep = int(inx[0]//step)
        dist  = nstep*step   
        delt  = inx[0]-dist
        fCnt  = iny[0]
        
        outx=np.array([i*step+delt for i in range(nstep)])
        outy=np.array([fCnt for i in range(nstep)])
        
        return outx, outy
    
    def data2hist(self, inx, iny):
        outx,outy=self.dataOffset(inx,iny, add=False)
        outx[0]=0.0
        return outx,outy
    
    def hist2data(self, inx, iny):
        outx,outy=self.dataOffset(inx,iny, add=True)
        if(outx[0]<=0.0):
            outx[0]=abs(outx[1]-out[0])/2.0
            if(outx[0]<=0.0):
                print('Error! hist2data, the first x is '+str(outx[0]))
        return outx,outy
    
    def dataOffset(self, inx, iny, add=True):
        delt = abs(inx[-1]-inx[0])/(2*inx.size)
        if(add==False):
            delt = -delt
        outx=np.array([x+delt for x in inx])
        return outx, iny
    
class PDFFourierTransform:
    def __init__(self, density=1.0):
        self.rho=density
        
    def dst(self, inx, inf, outX):              # Error Now, cannot be used
        outF = np.zeros(outX.size)              # F(X)
        for i, X in enumerate(outX):
            core    = inf*np.sin(X*inx)         # f(x)*sin(X*x)
            outF[i] = np.trapz(core,x=inx)      # F(X)=sum[f(x)*sin(X*x)]dx
        return outF
        
    def s2d(self, q, iQ, r):
        QiQ = np.multiply(q, iQ)
        dR  = np.array(fftpack.dst(QiQ,type=2,norm='ortho')[:r.size])
        #dR  = self.dst(q,iQ,r)
        return dR
     
    def lorchFunction(self, q,iQ):
        factor = math.pi/q.max()
        lorch  = np.sinc(np.multiply(q,factor))
        return lorch
    
    def s2dLorch(self, q, iQ, r):
        lorch       = self.lorchFunction(q,iQ)
        iQWithLorch = np.multiply(iQ, lorch)
        dR          = self.s2d(q, iQWithLorch, r)
        return dR
        
    def g2f(self, r, gR, q):
        factor = 4*math.pi*self.rho
        RgR    = np.multiply(r,gR)
        fQ     = np.array(fftpack.dst(RgR,type=2,norm='ortho'))
        #fQ     = self.dst(r,gR,q)
        fQ     = np.multiply(fQ, factor)
        fQ     = np.divide(fQ,q)
        return fQ
    

    def f2g(self, q, fQ, r):
        factor = 0.5/((math.pi**2)*self.rho)
        QfQ    = np.multiply(q,fQ)
        gR     = np.array(fftpack.dst(QfQ,type=2,norm='ortho'))
        #gR     = self.dst(q,QfQ,r)
        gR     = np.multiply(gR,factor)
        gR     = np.divide(gR,r)    
        return gR
            
class PDFTest:

    def __init__(self):
        self.ws=PDFWorkspace()
        self.th=PDFTophatFunction()
        self.ft=PDFFourierTransform(density=5.9)  
        self.bg=PDFSelfScatteringBKG()
    
    def int2dofr(self):
        x,y   = self.ws.loadData(suffix='int01')
        q,iQ  = self.ws.iQExtend2lowQ(x,y)
        hr,dR = self.ws.zeroGrFromQ(q)
        pr,dR = self.ws.hist2data(hr,dR)
        dR    = self.ft.s2dLorch(q,iQ, pr)
        return r,dR
        
    
    def soq2qsmooth(self):
        x,y   = self.ws.loadData(suffix='soq')
        q,iQ  = self.ws.iQExtend2lowQ(x,y)
        hq,iQ = self.ws.data2hist(q,iQ)
        hq,sQ = self.th.tophatConvolution(2.5,hq,iQ)
        return hq,sQ
    
    def soq2int(self):
        x,y   = self.ws.loadData(suffix='soq')
        q,iQ  = self.ws.iQExtend2lowQ(x,y)
        hq,iQ = self.ws.data2hist(q,iQ)
        hq,sQ = self.soq2qsmooth()
        dQ    = np.subtract(iQ,sQ)
        hr,dR = self.ws.zeroGrFromQ(q)
        pr,dR = self.ws.hist2data(hr, dR)
        dR    = self.ft.f2g(hq, dQ, pr)
        pr,bR = self.bg.getBkg(pr, dR, 1.8, 2.5)
        bQ    = self.ft.g2f(hr, bR, q)
        bQ    = np.divide(bQ, 4*math.pi) # why?
        IQ    = iQ-sQ-bQ
        return q, IQ
    
    def getGself(self):
        x,y   = self.ws.loadData(suffix='soq')
        q,iQ  = self.ws.iQExtend2lowQ(x,y)
        const = 2.0
        Iself = np.array([const for i in range(q.size)])
        r,dR  = self.ws.zeroGrFromQ(q)
        Gself = self.ft.f2g(q,Iself,r)
        return r, Gself
    
    