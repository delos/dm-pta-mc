"""
    Evaluating phase shift integrals using an interpolation table
"""

import numpy as np
from scipy.integrate import cumtrapz
from scipy.interpolate import interp1d
from scipy.special import hyp2f1
from scipy.ndimage import map_coordinates

class dphi_interpolation_table(object):
    """
    Parameters:
      r, M: enclosed mass table for given density profile (in units of r_s, rho_s*r_s^3)
      
      Mpower: power law index of the mass -- M ~ r^Mpower at small radii
    """
    
    def prepare_mass(self, r, M, Mpower):
        
        sort = np.argsort(r)
        self.r_ = r[sort]
        self.M_ = M[sort]
        self.Mpower = Mpower
        self.M_interp = interp1d(self.r_,self.M_,copy=False)
        
        self.M = self.M_[-1]
        self.Mcoef = self.M_[0] / self.r_[0]**self.Mpower
        
        self.rmin = self.r_[0]
        self.rmax = self.r_[-1]
    
    def prepare_grid(self, Nx, Ny, xmin, xmax, ymin, ymax):
        
        # x=vt/b  grid
        self.Nx = Nx
        self.xmin = xmin
        self.xmax = xmax
        self.x_ = np.geomspace(xmin,xmax,Nx)
        self.lnxmin = np.log(self.xmin)
        self.lnxmax = np.log(self.xmax)
        
        # y=b/rs grid
        self.Ny = Ny
        self.ymin = ymin
        self.ymax = ymax
        self.y_ = np.geomspace(ymin,ymax,Ny)
        self.lnymin = np.log(self.ymin)
        self.lnymax = np.log(self.ymax)
        
        # other limits
        self.deep_r2lim = (self.xmin*self.ymin)**0.5
        self.far_ylim = self.ymax**0.5
        self.headon_ylim = self.ymin**0.5
        
        # l=xy grid (for head-on integral)
        self.Nl = int(np.round(np.sqrt(self.Nx*self.Ny)))
        self.lmin = self.r_[0]
        self.lmax = self.r_[-1]
        self.l_ = np.geomspace(self.lmin,self.lmax,self.Nl)
        self.lnlmin = np.log(self.lmin)
        self.lnlmax = np.log(self.lmax)
    
    def check_mass(self):
        
        if self.ymax * np.sqrt(1+self.xmax**2) > self.r_[-1]:
            raise ValueError('Maximum radius in mass table must be %g or greater'%(self.ymax * np.sqrt(1+self.xmax**2)))
        if self.ymin < self.r_[0]:
            raise ValueError('Minimum radius in mass table must be %g or smaller'%(self.ymin))
          
        self.sqrt1x = np.sqrt(1+self.x_**2)
        self.Mlist = self.mass_profile(self.y_.reshape((1,self.Ny))*self.sqrt1x.reshape((self.Nx,1)))
    
    def integrate_bdterm(self):
        
        bdterm_integrand = self.Mlist * (1./self.sqrt1x**3).reshape((self.Nx,1))
        bdterm_integral = cumtrapz(bdterm_integrand,x=self.x_,axis=0,initial=0) + bdterm_integrand[np.array([0])] * self.x_[0]
        self.bdterm_ = cumtrapz(bdterm_integral,x=self.x_,axis=0,initial=0) + 0.5 * bdterm_integral[np.array([0])] * self.x_[0]
        self.bdterm_int_xmax = bdterm_integral[-1]
      
    def integrate_vdterm(self):
        
        vdterm_integrand = self.Mlist * (self.x_/self.sqrt1x**3).reshape((self.Nx,1))
        vdterm_integral = cumtrapz(vdterm_integrand,x=self.x_,axis=0,initial=0) + 0.5 * vdterm_integrand[np.array([0])] * self.x_[0]
        self.vdterm_ = cumtrapz(vdterm_integral,x=self.x_,axis=0,initial=0) + 1./3*vdterm_integral[np.array([0])] * self.x_[0]
        self.vdterm_int_xmax = vdterm_integral[-1]
    
    def integrate_headon(self):
        headon_integrand = self.mass_profile(self.l_)/self.l_**2
        headon_integral = cumtrapz(headon_integrand,x=self.l_,initial=0) + headon_integrand[0]*self.l_[0] / (self.Mpower-1)
        self.headon_ = cumtrapz(headon_integral,x=self.l_,initial=0) + headon_integral[0]*self.l_[0] / self.Mpower
        self.headon_int_lmax = headon_integral[-1]
    
    def __init__(self, r, M, Mpower, Nx=1000, Ny=1000, xmin=1e-5, xmax=1e5, ymin=1e-5, ymax = 1e5):
        
        self.prepare_mass(r, M, Mpower)
        
        self.prepare_grid(Nx, Ny, xmin, xmax, ymin, ymax)
        
        self.check_mass()
        
        self.integrate_bdterm()
        
        self.integrate_vdterm()
        
        self.integrate_headon()
    
    """
    Mass profile with extrapolation
    """
    def mass_profile(self,r):
        return np.piecewise(r,[r<self.rmin,r>self.rmax],[
            lambda r: self.M_[0]*(r/self.r_[0])**self.Mpower,
            self.M,
            self.M_interp,
            ])
    
    """
    Interpolate head-on collision table
    """
    def headon_table(self,l):
        i = (np.log(l) - self.lnlmin) / (self.lnlmax - self.lnlmin) * (self.Nl-1)
        ii = i.astype(int)
        f = i - ii
        return self.headon_[ii]*(1-f) + self.headon_[ii+1]*f
    
    """
    Interpolate main tables
    Note: map_coordinates is about 10 times faster than RectBivariateSpline here
    """
    def bd_vd_table(self,x,y):
        i = (np.log(x) - self.lnxmin) / (self.lnxmax - self.lnxmin) * (self.Nx-1)
        j = (np.log(y) - self.lnymin) / (self.lnymax - self.lnymin) * (self.Ny-1)
        return (
            map_coordinates(self.bdterm_, np.asarray([i,j]), order=1),
            map_coordinates(self.vdterm_, np.asarray([i,j]), order=1),
            )
  
    """
    y >> 1 (point object)
    """
    def bd_vd_far(self,x,y=0):
        return (
            self.M * (np.sqrt(1+x**2)-1),
            self.M * (x-np.arcsinh(x)),
            )
      
    """
    y << 1 and x << 1/y (deep inside large halo)
    """
    def bd_vd_deep(self,x,y):
        Mcoef = self.Mcoef * y**self.Mpower
        return (
            Mcoef * ((1-(1+x**2)**(0.5*(self.Mpower-1.))) / (self.Mpower-1.) + x**2*hyp2f1(0.5,0.5*(3-self.Mpower),1.5,-x**2)),
            Mcoef * x/(self.Mpower-1.) * (hyp2f1(0.5,0.5*(1-self.Mpower),1.5,-x**2) - 1.),
            )
  
    """
    x << 1 (near closest approach)
    """
    def bd_vd_near(self,x,y):
        My = self.mass_profile(y)
        return (
            My * 0.5 * x**2,
            My * 0.16666666666666666 * x**3,
            )
    
    """
    x >> 1 and y << 1 (head-on)
    """
    def bd_vd_headon(self,x,y):
        l = x*y # xy = vt/r_s
        return (
            0., # bd term is subdominant to vd term anyway
            np.piecewise(l,[l<self.lmax*.99],[
                self.headon_table,
                lambda l: self.headon_[-1] + self.headon_int_lmax*(l-self.lmax)
                ]),
            )
    
    """
    x >> 1
    """
    def bd_vd_largex(self,x,y):
        i = (np.log(y) - self.lnymin) / (self.lnymax - self.lnymin) * (self.Ny-1)
        ii = i.astype(int)
        f = i - ii
        bdterm_xmax = self.bdterm_[-1,ii]*(1-f) + self.bdterm_[-1,ii+1]*f
        vdterm_xmax = self.vdterm_[-1,ii]*(1-f) + self.vdterm_[-1,ii+1]*f
        bdterm_int_xmax = self.bdterm_int_xmax[ii]*(1-f) + self.bdterm_int_xmax[ii+1]*f
        vdterm_int_xmax = self.vdterm_int_xmax[ii]*(1-f) + self.vdterm_int_xmax[ii+1]*f
        return (
            bdterm_xmax + (x-self.xmax) * bdterm_int_xmax,
            vdterm_xmax + (x-self.xmax) * vdterm_int_xmax,
            )
    
    """
    List of x,y conditions in which to use different functions
    """
    
    def condlist(self,x,y):
        
        cond_table = (x>=self.xmin)&(x<=self.xmax)&(y>=self.ymin)&(y<=self.ymax)
        cond_deep = (~cond_table) & (y**2*(1+x**2) < self.deep_r2lim)
        cond_far = (~cond_table) & (y > self.far_ylim)
        _cond3 = (~cond_table) & (~cond_deep) & (~cond_far)
        cond_near = _cond3 & (x < self.xmin)
        cond_headon = _cond3 & (x > 1.) & (y < self.headon_ylim)
        cond_largex = _cond3 & (~cond_near) & (~cond_headon)
        
        return [
            cond_table,
            cond_deep,
            cond_far,
            cond_near,
            cond_headon,
            cond_largex,
          ]
    
    """
    main function
    """
    def bd_vd_terms(self, x, y): # x = vt/b, y = b/rs
        
        negative_x = x < 0
        x[negative_x] *= -1.
        
        condlist = self.condlist(x,y)
      
        funclist = [
            self.bd_vd_table,
            self.bd_vd_deep,
            self.bd_vd_far,
            self.bd_vd_near,
            self.bd_vd_headon,
            self.bd_vd_largex,
            ]
        
        # the following construction avoids extraneous function evaluations
        bd_out = np.zeros_like(x)
        vd_out = np.zeros_like(x)
        for k in range(len(condlist)):
            if np.sum(condlist[k]) > 0:
                bd_out[condlist[k]], vd_out[condlist[k]] = funclist[k](x[condlist[k]],y[condlist[k]])
        
        vd_out[negative_x] *= -1. # bd term is even in x, while vd term is odd
        
        return bd_out, vd_out
