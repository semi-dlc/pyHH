import numpy as np
from scipy.integrate import solve_ivp


class HHModel:
    """The HHModel tracks conductances of 3 channels to calculate Vm"""
    
    class Gate:
        """The Gate object manages a channel's kinetics and open state"""
        alpha, beta, state = 0, 0, 0
        
        def temperatureFactor(T):  # Celsius
            return np.float_power(3, 0.1 * (T - 6.3))
        
        PHI_T = temperatureFactor(29)

        def setInfiniteState(self):
            self.state = self.alpha / (self.alpha + self.beta)

    ENa, EK, EKleak = 115, -12, 10.6
    gNa, gK, gKleak = 120, 36, 0.3
    m, n, h = Gate(), Gate(), Gate()
    Cm = 0.1

    def __init__(self, startingVoltage=0):
        self.Vm = startingVoltage
        self._UpdateGateTimeConstants(startingVoltage)
        self.m.setInfiniteState()
        self.n.setInfiniteState()
        self.h.setInfiniteState()

        print("n, m, h : ", self.n.state, self.m.state, self.h.state)

    def _UpdateGateTimeConstants(self, Vm):
        """Update time constants of all gates based on the given Vm"""
        self.n.alpha = 0.01 * ((10 - Vm) / (np.exp((10 - Vm) / 10) - 1))
        self.n.beta = 0.125 * np.exp(-Vm / 80)
        self.m.alpha = 0.1 * ((25 - Vm) / (np.exp((25 - Vm) / 10) - 1))
        self.m.beta = 4 * np.exp(-Vm / 18)
        self.h.alpha = 0.07 * np.exp(-Vm / 20)
        self.h.beta = 1 / (np.exp((30 - Vm) / 10) + 1)

    def _ode_system(self, t, y, stimulusCurrent):
        """Defines the system of ODEs for Vm, m, n, and h"""
        Vm, m, n, h = y

        # Update time constants based on Vm
        self._UpdateGateTimeConstants(Vm)

        # Calculate currents
        INa = (m ** 3) * self.gNa * h * (Vm - self.ENa)
        IK = (n ** 4) * self.gK * (Vm - self.EK)
        IKleak = self.gKleak * (Vm - self.EKleak)
        Isum = stimulusCurrent - INa - IK - IKleak

        # Calculate derivatives
        dVm_dt = Isum / self.Cm
        dm_dt = self.m.PHI_T * (self.m.alpha * (1 - m) - self.m.beta * m)
        dn_dt = self.n.PHI_T * (self.n.alpha * (1 - n) - self.n.beta * n)
        dh_dt = self.h.PHI_T * (self.h.alpha * (1 - h) - self.h.beta * h)

        return [dVm_dt, dm_dt, dn_dt, dh_dt]

    def iterate(self, stimulusCurrent, deltaTms):
        """Integrates the system for a single timestep using solve_ivp"""
        # Initial conditions: current Vm, m, n, and h states
        y0 = [self.Vm, self.m.state, self.n.state, self.h.state]

        # Run solve_ivp to integrate over the interval [0, deltaTms]
        sol = solve_ivp(
            fun=self._ode_system,
            t_span=[0, deltaTms],
            y0=y0,
            args=(stimulusCurrent,),
            method="BDF"  # You could also try "BDF" for better stability in stiff systems
        )

        # Update the states with the final values from solve_ivp
        self.Vm, self.m.state, self.n.state, self.h.state = sol.y[:, -1]

        # Calculate and store currents for potential later use
        self.INa = (self.m.state ** 3) * self.gNa * self.h.state * (self.Vm - self.ENa)
        self.IK = (self.n.state ** 4) * self.gK * (self.Vm - self.EK)
        self.IKleak = self.gKleak * (self.Vm - self.EKleak)
    
    def get_currents(self):
        """Returns the ionic currents for potential plotting or analysis"""
        return self.INa, self.IK, self.IKleak
