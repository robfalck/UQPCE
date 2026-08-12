import numpy as np
import openmdao.api as om
from fixed import parameters

class AeroComp(om.ExplicitComponent):
    """
    Component for "AeroComp" box containing analytical derivatives
    """
    def initialize(self):
        self.options.declare('vec_size', default=1, types=int)

    def setup(self):
        n = self.options['vec_size']

        #proposed design variables
        self.add_input('S', units="m**2")
        self.add_input('V_cruise', units="m/s")
        self.add_input('AR', units="unitless")
       
        #model variable (output from other component)
        self.add_input('m_total',units="kg",shape=(n,))
        
        #uncertain parameters
        self.add_input('delta_CD0',val=np.ones(n),units="unitless",shape=(n,))
        self.add_input('delta_ks',val=np.ones(n),units="unitless",shape=(n,))
        self.add_input('delta_e',val=np.ones(n),units="unitless",shape=(n,))
        
        #tuning parameters
        self.add_input('ks_base', units="1/m**2")
        self.add_input('e_base', units="unitless")
        self.add_input('C_D0_base', units="unitless")

        #constant parameters
        self.add_input('g', val=parameters['g'], units="m/s**2" )
        self.add_input('rho', val=parameters['rho'], units="kg/m**3")
        self.add_input('S_0', val=parameters['S_naught'], units="m**2" )
    
        #outputs
        # res_ref scales each residual to O(1) for the AeroStruct Newton solve.
        # Unscaled, WL (~5.6e3) and LD (~15) dominated the residual norm and forced
        # Newton to chase ~12 orders of magnitude down to atol=1e-8.
        self.add_output('CL',units="unitless",shape=(n,))
        self.add_output('CD',units="unitless",shape=(n,))
        self.add_output('LD',units="unitless",shape=(n,), res_ref=10.0)
        self.add_output('WL',units="N/m**2",shape=(n,), res_ref=5.0e3)

    
    def setup_partials(self):
        n = self.options['vec_size']
        arange = np.arange(n)
    
    #Sensitivities-start~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.declare_partials(of="CL",wrt='V_cruise',method="exact")
        self.declare_partials(of="CL",wrt="S",method="exact")
        #self.declare_partials(of="CL",wrt="AR",method="exact")
        #des variables^
        self.declare_partials(of="CL",wrt="m_total",method="exact", rows=arange, cols=arange)
        self.declare_partials(of="CL",wrt="rho",method="exact")
        self.declare_partials(of="CL",wrt="g",method="exact")
        #all the rest of CL wrt other inputs are zero by default, so not needed
        #other partials just in case^

        self.declare_partials(of="CD",wrt='V_cruise',method="exact")
        self.declare_partials(of="CD",wrt="S",method="exact")
        self.declare_partials(of="CD",wrt="AR",method="exact")
        self.declare_partials(of="CD",wrt="m_total",method="exact", rows=arange, cols=arange)
        self.declare_partials(of="CD",wrt="rho",method="exact")
        self.declare_partials(of="CD",wrt="g",method="exact")
        self.declare_partials(of="CD",wrt="C_D0_base",method="exact")
        self.declare_partials(of="CD",wrt="S_0",method="exact")
        self.declare_partials(of="CD",wrt="e_base",method="exact")
        self.declare_partials(of="CD",wrt="delta_CD0",method="exact", rows=arange, cols=arange)
        self.declare_partials(of="CD",wrt="delta_e",method="exact", rows=arange, cols=arange)
        self.declare_partials(of="CD",wrt="ks_base",method="exact")
        self.declare_partials(of="CD",wrt="delta_ks",method="exact", rows=arange, cols=arange)


        self.declare_partials(of="LD",wrt='V_cruise',method="exact")
        self.declare_partials(of="LD",wrt="S",method="exact")
        self.declare_partials(of="LD",wrt="AR",method="exact")
        self.declare_partials(of="LD",wrt="m_total",method="exact", rows=arange, cols=arange)
        self.declare_partials(of="LD",wrt="rho",method="exact")
        self.declare_partials(of="LD",wrt="g",method="exact")
        self.declare_partials(of="LD",wrt="C_D0_base",method="exact")
        self.declare_partials(of="LD",wrt="S_0",method="exact")
        self.declare_partials(of="LD",wrt="e_base",method="exact")
        self.declare_partials(of="LD",wrt="delta_CD0",method="exact", rows=arange, cols=arange)
        self.declare_partials(of="LD",wrt="delta_e",method="exact", rows=arange, cols=arange)
        self.declare_partials(of="LD",wrt="ks_base",method="exact")
        self.declare_partials(of="LD",wrt="delta_ks",method="exact", rows=arange, cols=arange)

        self.declare_partials(of="WL",wrt="S",method="exact")
        self.declare_partials(of="WL",wrt="m_total",method="exact", rows=arange, cols=arange)
        self.declare_partials(of="WL",wrt="g",method="exact")

    #passes input member inherited from om.Exp for reading and
    #outputs memeber struct/map thing whatever python calls it for writing
    def compute(self,inputs,outputs):
        g = inputs['g']
        rho = inputs['rho']
        C_D0_base = inputs['C_D0_base']
        S_0 = inputs['S_0']
        ks_base = inputs['ks_base']
        e_base = inputs['e_base']
        delta_CD0 = inputs['delta_CD0']
        delta_ks = inputs['delta_ks']
        delta_e = inputs['delta_e']
        m_total = inputs['m_total']
        V_cruise = inputs['V_cruise']
        S = inputs['S']
        AR = inputs['AR']

        #do this \/ double equal thingy to reuse output when needed, this synatx pattern might be useful
        #in compute partials function for chain rule stuff
        outputs['CL'] = CL = (m_total*g) / ((1.0/2.0)*rho*(V_cruise**2)*S)
        C_D0 = C_D0_base*delta_CD0 + ks_base*delta_ks*(S-S_0)     
        outputs['CD'] = CD = C_D0 + (CL**2) / (np.pi*AR*e_base*delta_e)
        outputs['LD'] = CL/CD
        outputs['WL'] = (m_total*g) / S

    def compute_partials(self, inputs, partials): #I presume inputs and partials are inherited memebers of
        g = inputs['g']
        rho = inputs['rho']
        C_D0_base = inputs['C_D0_base']
        S_0 = inputs['S_0']
        ks_base = inputs['ks_base']
        e_base = inputs['e_base']
        delta_CD0 = inputs['delta_CD0']
        delta_ks = inputs['delta_ks']
        delta_e = inputs['delta_e']
        m_total = inputs['m_total']
        V_cruise = inputs['V_cruise']
        S = inputs['S']
        AR = inputs['AR']

        CL = (m_total*g) / ((1.0/2.0)*rho*(V_cruise**2)*S)         
        C_D0 = C_D0_base*delta_CD0 + ks_base*delta_ks*(S-S_0) 
        CD = C_D0 + (CL**2) / (np.pi*AR*e_base*delta_e)
                                                  
        partials['CL','V_cruise'] = dCLdV = -2*CL*(1.0/V_cruise)
        partials['CL','S'] = dCLdS = -1*CL*(1.0/S)
        #partials['CL','AR'] = dCLdAR = 0 #fixed to assume S and AR as independent. span is always 
        #computed from these inputs
        dCLdAR = 0
        partials['CL','m_total'] = dCLdm = CL/m_total
        partials['CL','rho'] = dCLdrho = -CL/rho
        partials['CL','g'] = dCLdg = CL/g
    
        #ugliness helpers
        dCD_0dV = 0.0
        dCD_0dS = ks_base*delta_ks
        #b_squared = inputs['AR']*inputs['S']
        dSdAR = 0.0 #fixed to assume S and AR as independent. span is always 
        #computed from these inputs
        dCD_0dAR = dCD_0dS*dSdAR
        dARdS = 0.0 #fixed to assume S and AR as independent. span is always 
        #computed from these inputs
        dCD_0dm = 0.0
        dCD_0drho = 0.0 
        dCD_0dg = 0.0
        #dARdg = 0.0
        dCD_0dCDbase = delta_CD0
        dCD_0dS0 = -ks_base*delta_ks
        dCD_0debase = 0.0
        dCD_0ddeltaCD0 = C_D0_base
        dCD_0ddeltae = 0


        #product rule/quotient rule or whatever u wanna call it helpers
        product_rule_V = 2*CL*dCLdV*(1/AR) #+ (CL**2)*(0)
        product_rule_S = 2*CL*dCLdS*(1/AR) - (CL**2)*(1.0/(AR**2))*dARdS
        product_rule_AR = 2*CL*dCLdAR*(1/AR) - (CL**2)*(1.0/(AR**2))*(1.0)
        product_rule_m = 2*CL*dCLdm*(1/AR)
        product_rule_rho = 2*CL*dCLdrho*(1/AR)
        product_rule_g = 2*CL*dCLdg*(1/AR)
        
        partials['CD','V_cruise'] = dCDdV = dCD_0dV + (1/(np.pi*e_base*delta_e))*(product_rule_V)
        partials['CD','S'] = dCDdS =  dCD_0dS + (1/(np.pi*e_base*delta_e))*(product_rule_S)
        partials['CD','AR'] = dCDdAR = dCD_0dAR +  (1/(np.pi*e_base*delta_e))*(product_rule_AR)
        partials['CD','m_total'] = dCDdm =  dCD_0dm + (1/(np.pi*e_base*delta_e))*(product_rule_m)
        partials['CD','rho'] = dCDdrho =  dCD_0drho + (1/(np.pi*e_base*delta_e))*(product_rule_rho)
        partials['CD','g'] = dCDdg =  dCD_0dg + (1/(np.pi*e_base*delta_e))*(product_rule_g)

        partials['CD','C_D0_base'] = dCDdCD0base =  dCD_0dCDbase 
        partials['CD','S_0'] = dCDdS0 =  dCD_0dS0 
        partials['CD','e_base'] = dCDdebase =  dCD_0debase - ((CL**2)/(np.pi*e_base*e_base*delta_e*AR))
        partials['CD','delta_CD0'] = dCDddeltaCD0 =  dCD_0ddeltaCD0 
        partials['CD','delta_e'] = dCDddeltae =  dCD_0ddeltae - ((CL**2)/(np.pi*e_base*delta_e*delta_e*AR))
        partials['CD','ks_base'] = dCDdks_base = delta_ks*(S-S_0)
        partials['CD','delta_ks'] = dCDddelta_ks =  ks_base*(S-S_0)

        partials['LD','V_cruise'] = (CD*dCLdV - CL*dCDdV)/(CD**2) 
        partials['LD','S'] = (CD*dCLdS - CL*dCDdS)/(CD**2)
        partials['LD','AR'] = (CD*dCLdAR - CL*dCDdAR)/(CD**2)
        partials['LD','m_total'] = (CD*dCLdm - CL*dCDdm)/(CD**2)
        partials['LD','rho'] = (CD*dCLdrho - CL*dCDdrho)/(CD**2)
        partials['LD','g'] = (CD*dCLdg - CL*dCDdg)/(CD**2)

        partials['LD','C_D0_base'] =  (0 - CL*dCDdCD0base)/(CD**2) 
        partials['LD','S_0'] = (0 - CL*dCDdS0)/(CD**2) 
        partials['LD','e_base'] = (0 - CL*dCDdebase)/(CD**2) 
        partials['LD','delta_CD0'] = (0 - CL*dCDddeltaCD0)/(CD**2) 
        partials['LD','delta_e'] = (0 - CL*dCDddeltae)/(CD**2) 
        partials['LD','ks_base'] = -(CL*dCDdks_base)/(CD**2)
        partials['LD','delta_ks'] = -(CL*dCDddelta_ks)/(CD**2)

        partials['WL','S'] = -(inputs['m_total']*g) / (inputs['S']**2)
        partials['WL','m_total'] = (g) / (inputs['S'])
        partials['WL','g'] = (inputs['m_total']) / (inputs['S'])

#JAX component working fine :)
import jax
import jax.numpy as jnp

import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64",True)

class AeroCompJax(om.JaxExplicitComponent):

    def initialize(self):
        self.options.declare('vec_size', types=int)

    def setup(self):
        n = self.options['vec_size']

        #proposed design variables
        self.add_input('S',  units="m**2")
        self.add_input('V_cruise', units="m/s")
        self.add_input('AR', units="unitless")
       
        #model variable (output from other component)
        self.add_input('m_total',units="kg",shape=(n,))
        
        #uncertain parameters
        self.add_input('delta_CD0',val=jnp.ones(n),units=None,shape=(n,))
        self.add_input('delta_ks',val=jnp.ones(n),units=None,shape=(n,))
        self.add_input('delta_e',val=jnp.ones(n),units=None,shape=(n,))
        
        #tuning parameters
        self.add_input('ks_base', units="1/m**2")
        self.add_input('e_base', units=None)
        self.add_input('C_D0_base', units=None)

        #constant parameters
        self.add_input('g', val=parameters['g'], units="m/s**2" )
        self.add_input('rho', val=parameters['rho'], units="kg/m**3")
        self.add_input('S_0', val=parameters['S_naught'], units="m**2" )
    
        #outputs
        self.add_output('CL',units=None,shape=(n,))
        self.add_output('CD',units=None,shape=(n,))
        self.add_output('LD',units=None,shape=(n,))
        self.add_output('WL',units="N/m**2",shape=(n,))

    #jax assigns inputs to each of the follwing var names in args
    #in the order they appear in setup
    #as a result its best to just keep the names the same I guess
    def compute_primal(self,
                       S,V_cruise,AR,
                       m_total,
                       delta_CD0, delta_ks, delta_e,
                       ks_base, e_base, C_D0_base,
                       g, rho, S_0):

        CL = (m_total*g)/((1.0/2.0)*rho*V_cruise*V_cruise*S)
        CD0 = C_D0_base*delta_CD0 + ks_base*delta_ks*(S-S_0)
        CD = CD0 + (CL**2)/(jnp.pi*AR*e_base*delta_e)
        LD = CL/CD
        WL = (m_total*g)/S

        return CL , CD, LD, WL

    #This function let's Jax know to recompile if a non input 
    #static variable changes and requires recompilation
    #def get_self_statics(self):
    #    return (self.options["vec_size"],)






def main():
    pass

if __name__ == "__main__":
    main()