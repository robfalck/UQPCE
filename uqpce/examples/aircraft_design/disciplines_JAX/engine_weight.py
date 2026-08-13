import openmdao.api as om
import jax.numpy as jnp
import numpy as np
from fixed import parameters

class EngineWeightComp(om.JaxExplicitComponent):
    """
    Component for "EngineWeightComp" box containing JAX
    """
    def initialize(self):
        self.options.declare('vec_size', default=1, types=int)

    def setup(self):
        n = self.options['vec_size']

        #proposed design variables
        self.add_input('SFC_tech', units="unitless")

        #model variable (output from other component)
        #n/a

        #uncertain parameters
        self.add_input('delta_alpha', val=1.0, units='unitless', shape=(n,))
        
        #tuning parameters
        self.add_input('alpha_base', units='unitless')

        #constant parameters
        self.add_input('m_eng_ref', val=parameters['m_eng_ref'], units='kg')

        #outputs
        self.add_output('m_engine', units='kg', desc='Engine mass', shape=(n,))

        n = self.options['vec_size']
        arange = np.arange(n)
        # Vector inputs are elementwise across the sample dimension, so these
        # subjacs are diagonal. Without this OpenMDAO assumes dense (n x n),
        # which made the AeroStruct sparse LU ~82x more expensive.
        self.declare_partials(of='m_engine', wrt=['delta_alpha'], rows=arange, cols=arange)
        self.declare_partials(of='m_engine', wrt=['SFC_tech', 'alpha_base', 'm_eng_ref'])

    def compute_primal(self, SFC_tech, delta_alpha, alpha_base, m_eng_ref):
        """
        m_engine = m_eng_ref * (1 + alpha_base * delta_alpha * SFC_tech)
        """
        
        return m_eng_ref * (1 + alpha_base * delta_alpha * SFC_tech)