import sys
import os
if not os.path.exists("./src/pyhh/__init__.py"):
    raise Exception("script must be run in project root folder")
sys.path.append("./src")

import numpy as np
import matplotlib.pyplot as plt
import pyhh

V_r = 70

if __name__ == "__main__":

    # customize a neuron model if desired
    model = pyhh.HHModel()
    model.gNa = 120  # typically 120
    model.gK = 36  # typically 36
    model.gKleak = 0.3  # typically 0.3

    model.EK = -12  # typically -12
    model.ENa = 115 # typically 115
    model.EKleak = 10.6  # typically 10.6
    
    model.Cm = 1  # typically 0.1

    # customize a stimulus waveform
    stim = np.zeros(10)
    stim[0:43000] = 40 # add a square pulse

    # simulate the model cell using the custom waveform
    sim = pyhh.Simulation(model)
    sim.Run(stimulusWaveform=stim, stepSizeMs=0.1)

    # plot the results with MatPlotLib
    plt.figure(figsize=(10, 8))

    ax1 = plt.subplot(411)
    ax1.plot(sim.times, sim.Vm, color='b')
    ax1.set_ylabel("Potential (mV)")
    ax1.set_title("Hodgkin-Huxley Neuron Model" )

    ax2 = plt.subplot(412)
    ax2.plot(sim.times, stim, color='r')
    ax2.set_ylabel("Stimulation (µA/cm²)")

    ax3 = plt.subplot(413, sharex=ax1)
    ax3.plot(sim.times, sim.StateH, label='h')
    ax3.plot(sim.times, sim.StateM, label='m')
    ax3.plot(sim.times, sim.StateN, label='n')
    ax3.set_ylabel("Activation (frac)")
    ax3.legend()

    ax4 = plt.subplot(414, sharex=ax1)
    ax4.plot(sim.times, sim.INa, label='VGSC')
    ax4.plot(sim.times, sim.IK, label='VGKC')
    ax4.plot(sim.times, sim.IKleak, label='KLeak')
    ax4.plot(sim.times, sim.INa + sim.IK + sim.IKleak, label='Isum')
    ax4.set_ylabel("Current (µA/cm²)")
    ax4.set_xlabel("Simulation Time (milliseconds)")
    ax4.legend()

    plt.tight_layout()
    plt.savefig("tests/demo.png")
    plt.show()
