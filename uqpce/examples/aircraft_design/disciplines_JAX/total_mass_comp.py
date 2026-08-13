import openmdao.api as om
import jax.numpy as jnp
import numpy as np
from fixed import parameters

class TotalMassComp(om.JaxExplicitComponent):
    """
    Component for "TotalMassComp" box containing JAX
    """
    def initialize(self):
        self.options.declare('vec_size', default=1, types=int)
    
    def setup(self):
        n = self.options['vec_size']

        #proposed design variables
        #n/a

        #model variable (output from other component)
        self.add_input('m_empty', units='kg', shape=(n,))
        self.add_input('m_fuel', units='kg', shape=(n,))

        #uncertain parameters
        #n/a

        #tuning parameters
        #n/a

        #constant parameters
        self.add_input('m_payload', val=parameters['m_payload_design'], units='kg')

        #outputs
        self.add_output('m_total', units='kg', shape=(n,))

    def setup_partials(self):
        n = self.options['vec_size']
        arange = np.arange(n)
        # Vector inputs are elementwise across the sample dimension, so these
        # subjacs are diagonal. Without this OpenMDAO assumes dense (n x n),
        # which made the AeroStruct sparse LU ~82x more expensive.
        self.declare_partials(of='m_total', wrt=['m_empty', 'm_fuel'], rows=arange, cols=arange)
        self.declare_partials(of='m_total', wrt=['m_payload'])

    def compute_primal(self, m_empty, m_fuel, m_payload):
        """
        m_total = m_empty + m_fuel + m_payload
        """

        return m_empty + m_fuel + m_payload