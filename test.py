import PyS2Glib
from matplotlib import pyplot as plt

class PDFTest:

    def __init__(self,default='GEM_TEST/GEM79347.',density=4.8):
        # 'GEM_TEST/GEM79347.'  4.8
        # '01-027/i15-1-18935_tth_det2_0.'  5.9
        default_name    = default
        default_density = density
        self.ws=PyS2Glib.PDFWorkspace(default=default_name)
        self.th=PyS2Glib.PDFTophatFunction()
        self.ft=PyS2Glib.PDFFourierTransform(density=default_density) 
        self.bg=PyS2Glib.PDFSelfScatteringBKG()
    
    
    def dcs2dor(self):
        x,y   = self.ws.loadData(suffix='mdcs01')
        pq,iQ = self.ws.iQExtend2lowQ(x,y)
        hq,iQ = self.ws.data2hist(pq,iQ)
        
        self.ws.saveData((hq,iQ), suffix='pyhist')
        
        hq,sQ = self.th.tophatConvolution(2.0,hq,iQ)
        
        self.ws.saveData((hq,sQ), suffix='pyqsmooth')
        
        dQ    = iQ-sQ
        
        self.ws.saveData((hq,dQ), suffix='pyqsub1')

        hQ,bQ = self.bg.getFlatQBkg(hq, dQ, 17.0, 37.0) 
        
        self.ws.saveData((hq,bQ), suffix='pyqsub2')
        
        fQ    = iQ-sQ-bQ
        
        self.ws.saveData((hq,fQ), suffix='pymint01')
        
        hr,dR = self.ws.zeroGrFromQ(pq)
        pr,dR = self.ws.hist2data(hr,dR)
        dR    = self.ft.f2dLorch(pq,fQ, pr)
        
        self.ws.saveData((pr,dR), suffix='pymdor01')
        
        gR    = self.ft.f2gLorch(pq,fQ, pr)
        
        self.ws.saveData((pr,gR), suffix='pygor01')
        
        return pr,dR
        
        
        
    def int2dofr(self):
        x,y   = self.ws.loadData(suffix='int01')
        q,iQ  = self.ws.iQExtend2lowQ(x,y)
        hr,dR = self.ws.zeroGrFromQ(q)
        pr,dR = self.ws.hist2data(hr,dR)
        dR    = self.ft.f2dLorch(q,iQ, pr)
        return pr,dR
        
    
    def soq2qsmooth(self):
        x,y   = self.ws.loadData(suffix='soq')
        q,iQ  = self.ws.iQExtend2lowQ(x,y)
        hq,iQ = self.ws.data2hist(q,iQ)
        hq,sQ = self.th.tophatConvolution(2.5,hq,iQ)
        return hq,sQ
    
    def soq2dofr(self,suffix='soq'):
        default_suffix = suffix
        rMin  = 2.0
        qT    = 2.5
        x,y   = self.ws.loadData(suffix=default_suffix)
        pq,iQ = self.ws.iQExtend2lowQ(x,y)
        hq,iQ = self.ws.data2hist(pq,iQ)
        
        self.ws.saveData((hq,iQ), suffix='pyhist')
        
        hq,sQ = self.th.tophatConvolution(qT,hq,iQ)    # Soper[2009](EQ.42 EQ.43)
        
        self.ws.saveData((hq,sQ), suffix='pyqsmooth')
        
        dQ    = iQ-sQ                                  # Soper[2009](EQ.44)
        
        self.ws.saveData((hq,dQ), suffix='pyqsub1')
                                  
        hr,dR = self.ws.zeroGrFromQ(pq)
        pr,dR = self.ws.hist2data(hr, dR)
        dR    = self.ft.f2g(hq, dQ, pr)                 # Soper[2009](EQ.45)
        pr,bR = self.bg.getBkg(pr, dR, rMin, qT)        # Soper[2009](EQ.53 EQ.54)
        bQ    = self.ft.g2f(hr, bR, pq)                 # Soper[2009](EQ.55)
        bQ    = bQ/(4*math.pi)                          # why?
        
        self.ws.saveData((pq,bQ), suffix='pyqsub2')
                
        fQ    = iQ-sQ-bQ                                # Soper[2009](EQ.56)
        #fQ    = iQ-sQ
        
        self.ws.saveData((pq,fQ), suffix='pyint01')
        
        hr,dR = self.ws.zeroGrFromQ(pq)
        pr,dR = self.ws.hist2data(hr,dR)
        dR    = self.ft.f2dLorch(pq,fQ, pr)
        
        self.ws.saveData((pr,dR), suffix='pydofr')
        
        gR    = self.ft.f2gLorch(pq,fQ, pr)
        
        self.ws.saveData((pr,gR), suffix='pygofr')
        
        return pr,dR
        
    def soq2int(self):
        x,y   = self.ws.loadData(suffix='soq')
        q,iQ  = self.ws.iQExtend2lowQ(x,y)
        hq,iQ = self.ws.data2hist(q,iQ)
        hq,sQ = self.th.tophatConvolution(2.5,hq,iQ)    # Soper[2009](EQ.42 EQ.43)
        dQ    = iQ-sQ                                   # Soper[2009](EQ.44)
        hr,dR = self.ws.zeroGrFromQ(q)
        pr,dR = self.ws.hist2data(hr, dR)
        dR    = self.ft.f2g(hq, dQ, pr)                 # Soper[2009](EQ.45)
        pr,bR = self.bg.getBkg(pr, dR, 1.8, 2.5)        # Soper[2009](EQ.53 EQ.54)
        bQ    = self.ft.g2f(hr, bR, q)                  # Soper[2009](EQ.55)
        bQ    = bQ/(4*math.pi)                          # why?
        IQ    = iQ-sQ-bQ                                # Soper[2009](EQ.56)
        return q, IQ
    
    def getGself(self):
        x,y   = self.ws.loadData(suffix='soq')
        q,iQ  = self.ws.iQExtend2lowQ(x,y)
        const = 2.0
        Iself = np.array([const for i in range(q.size)])
        r,dR  = self.ws.zeroGrFromQ(q)
        Gself = self.ft.f2g(q,Iself,r)
        return r, Gself
                                                                                   
    
