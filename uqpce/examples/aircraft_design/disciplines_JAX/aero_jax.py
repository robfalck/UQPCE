import jax
import jax.numpy as jnp
import openmdao.api as om

from fixed import parameters

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
        self.add_input('delta_CD0',val=jnp.ones(n),units="unitless",shape=(n,))
        self.add_input('delta_ks',val=jnp.ones(n),units="unitless",shape=(n,))
        self.add_input('delta_e',val=jnp.ones(n),units="unitless",shape=(n,))
        
        #tuning parameters
        self.add_input('ks_base', units="1/m**2")
        self.add_input('e_base', units="unitless")
        self.add_input('C_D0_base', units="unitless")

        #constant parameters
        self.add_input('g', val=parameters['g'], units="m/s**2" )
        self.add_input('rho', val=parameters['rho'], units="kg/m**3")
        self.add_input('S_0', val=parameters['S_naught'], units="m**2" )
    
        #outputs
        # res_ref mirrors disciplines/aero.py: scales residuals to O(1) for the
        # AeroStruct Newton solve.
        self.add_output('CL',units="unitless",shape=(n,))
        self.add_output('CD',units="unitless",shape=(n,))
        self.add_output('LD',units="unitless",shape=(n,), res_ref=10.0)
        self.add_output('WL',units="N/m**2",shape=(n,), res_ref=5.0e3)

    #jax assigns inputs to each of the follwing var names in args
    #in the order they appear in setup
    #as a result its best to just keep the names the same I guess
    def compute_primal(self,
                       S, V_cruise, AR,
                       m_total,
                       delta_CD0, delta_ks, delta_e,
                       ks_base, e_base, C_D0_base,
                       g, rho, S_0):

        CL = (m_total*g)/((1.0/2.0)*rho*V_cruise*V_cruise*S)
        CD0 = C_D0_base*delta_CD0 + ks_base*delta_ks*(S-S_0)
        CD = CD0 + (CL**2)/(jnp.pi*AR*e_base*delta_e)
        LD = CL/CD
        WL = (m_total*g)/S

        return CL, CD, LD, WL

    # #This function let's Jax know to recompile if a non input 
    # #static variable changes and requires recompilation
    # def get_self_statics(self):
    #     return (self.option["vec_size"],)

def main():
    pass

if __name__ == "__main__":
    main()