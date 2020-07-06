import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pysingfel as ps
from pysingfel.particlePlacement import max_radius, distribute_particles, position_in_3d
import os

numOpen = 50
numClosed = 10
pwd = os.path.dirname(__file__)

# Create a particle object
particleOp = ps.Particle()
particleOp.read_pdb(os.path.join(pwd,'../input/pdb/3iyf.pdb'), ff='WK')


particleCl = ps.Particle()
particleCl.read_pdb(os.path.join(pwd,'../input/pdb/3j03.pdb'), ff='WK')

# Load beam
beam = ps.Beam(os.path.join(pwd,'../input/beam/amo86615.beam'))

geom = os.path.join(pwd,'../input/lcls/amo86615/PNCCD::CalibV1/Camp.0:pnCCD.1/geometry/0-end.data')

# Load and initialize the detector
det = ps.PnccdDetector(geom=geom, beam=beam)

x, y, z = position_in_3d(particles={particleOp:numOpen,particleCl:numClosed}, beam_focus_radius=beam._focus_xFWHM/2, jet_radius=1e-4)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x*1e9, y*1e9, z*1e9)
ax.set_xlabel('X [nm]')
ax.set_ylabel('Y [nm]')
ax.set_zlabel('Z [nm]')
ax.set_title('Chaperones Distribution in 3D Real Space')
ax.set_xlim3d(-beam._focus_xFWHM/2*1e9,beam._focus_xFWHM/2*1e9)
ax.set_ylim3d(-beam._focus_xFWHM/2*1e9,beam._focus_xFWHM/2*1e9)
ax.set_zlim3d(-1e5,1e5)
#ax.auto_scale_xyz([-beam._focus_xFWHM/2*1e9,beam._focus_xFWHM/2*1e9], [-beam._focus_xFWHM/2*1e9,beam._focus_xFWHM/2*1e9], [-1e5,1e5])
ax.set_aspect('equal')
plt.show()

