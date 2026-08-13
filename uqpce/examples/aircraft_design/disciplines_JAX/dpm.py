import openmdao.api as om
import jax.numpy as jnp
import numpy as np
from fixed import parameters

class Dpm(om.JaxExplicitComponent):
    """
    Component for normalized DOC containing JAX
    """
    def initialize(self):
        self.options.declare('vec_size', default=1, types=int)

    def setup(self):
        n = self.options['vec_size']

        #proposed design variables
        #n/a

        #model variable (output from other component)
        self.add_input('DOC', units='USD', shape=(n,))
        self.add_input('R', units='km', shape=(n,))

        #uncertain parameters
        #n/a

        #tuning parameters
        #n/a

        #constant parameters
        self.add_input('N_pax', val=parameters['N_pax'])

        #outputs
        self.add_output('Dpm', shape=(n,))

    def setup_partials(self):
        n = self.options['vec_size']
        arange = np.arange(n)
        # Vector inputs are elementwise across the sample dimension, so these
        # subjacs are diagonal. Without this OpenMDAO assumes dense (n x n),
        # which made the AeroStruct sparse LU ~82x more expensive.
        self.declare_partials(of='Dpm', wrt=['DOC', 'R'], rows=arange, cols=arange)
        self.declare_partials(of='Dpm', wrt=['N_pax'])

    def compute_primal(self, DOC, R, N_pax):
        """
        Dpm = DOC / (pax * km)
        """

        return DOC / (N_pax * R)