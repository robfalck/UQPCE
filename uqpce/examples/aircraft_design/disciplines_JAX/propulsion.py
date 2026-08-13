import openmdao.api as om
import jax.numpy as jnp
import numpy as np
from fixed import parameters

class PropulsionComp(om.JaxExplicitComponent):
    """
    Component for "PropulsionComp" box containing JAX
    """
    def initialize(self):
        self.options.declare('vec_size', default=1, types=int)

    def setup(self):
        n = self.options['vec_size']

        #Parameters
        
        #proposed design variables
        self.add_input('SFC_tech', units="unitless")
        self.add_input('V_cruise', units='m/s')

        #model variable (output from other component)

        #uncertain parameters
        self.add_input('delta_eta', val=jnp.ones(n), units="unitless", shape=(n,))
        self.add_input('delta_kv', val=jnp.ones(n), units="unitless", shape=(n,))
        
        #tuning parameters
        self.add_input('eta_base', units="unitless")
        self.add_input('kv_base', units="unitless")

        #constant parameters
        self.add_input('SFC_ref', val=parameters['SFC_ref'], units='1/s')
        self.add_input('V_ref', val=parameters['V_ref'], units="m/s")

        #outputs
        self.add_output('SFC', units="1/s", shape=(n,))

    def setup_partials(self):
        n = self.options['vec_size']
        arange = np.arange(n)
        # Vector inputs are elementwise across the sample dimension, so these
        # subjacs are diagonal. Without this OpenMDAO assumes dense (n x n),
        # which made the AeroStruct sparse LU ~82x more expensive.
        self.declare_partials(of='SFC', wrt=['delta_eta', 'delta_kv'], rows=arange, cols=arange)
        self.declare_partials(of='SFC', wrt=['SFC_tech', 'V_cruise', 'eta_base', 'kv_base', 'SFC_ref', 'V_ref'])

    def compute_primal(self, SFC_tech, V_cruise, delta_eta, delta_kv, eta_base, kv_base, SFC_ref, V_ref):
        """
        SFC = SFC_ref * (1 - eta_base * delta_eta * SFC_tech) * (1 + kv_base * delta_kv * (V/V_ref - 1)^2)
        """

        return SFC_ref * (1 - eta_base * delta_eta * SFC_tech) * (1 + kv_base * delta_kv * (V_cruise/V_ref - 1)**2)