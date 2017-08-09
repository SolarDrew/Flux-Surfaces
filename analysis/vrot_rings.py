from __future__ import print_function
import os
import sys
import glob

import numpy as np
from tvtk.api import tvtk
import vtk

try:
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    mpi = True
    mpi = mpi and (size != 1)
except ImportError:
    mpi = False
    rank = 0
    size = 1

import pysac.yt
import pysac.analysis.tube3D.process_utils as util
from pysac.analysis.tube3D import tvtk_tube_functions as ttf

# Import this repos config
sys.path.append("..")
from scripts.sacconfig import SACConfig
from scripts import plotting_helpers as ph

cube_slice = np.s_[:, :, :-5]

cfg = SACConfig()
cfg.print_config()

gdf_files = glob.glob(os.path.join(cfg.gdf_dir, cfg.get_identifier()+'_0*.gdf'))
gdf_files.sort()

n_lines = 100  # Number of lines in theta
thetas = np.linspace(0, 2*np.pi, n_lines)

#for tube_r in cfg.tube_radii:
for r in range(3, 66, 3):
#for r in range(54, 66, 3):
    tube_r = 'r{:02}'.format(r)
    print("Starting Tube {}".format(tube_r))
    surface_files = np.array(ph.glob_files(cfg, tube_r, "Fieldline*"))
    #rank_indices = np.array_split(np.arange(len(surface_files)), size)
    rank_indices = np.array_split(np.arange(290), size)
    if not mpi:
        rank_indices = rank_indices[0]

    if mpi:
        rank_indices = rank_indices[rank]

    for ring_height in range(5, 125, 10):  # in grid points
        print("Starting height {}".format(ring_height))

        theta_pos = np.zeros([len(surface_files), n_lines, 3])
        vphi = np.zeros([len(surface_files), n_lines])
        #print(vphipos.shape, theta_pos.shape)

        # Make variables for line intersection
        t = vtk.mutable(0)
        pcoords = [0.0, 0.0, 0.0]
        subId = vtk.mutable(0)
        cellId = vtk.mutable(0)
        cell = tvtk.GenericCell()

        for n in rank_indices:
            vtp = surface_files[n]
            #print(rank, n, vtp)
            ds = pysac.yt.SACGDFDataset(gdf_files[n])
            bfield, vfield, density, valf, cs, beta, r_pos = util.get_yt_mlab(ds, cube_slice, flux=True)
            ds.close()
            del ds, bfield, density, valf, cs, beta, r_pos

            # Build CellLocator for this time step
            surface = ttf.read_step(vtp)
            loc = tvtk.CellLocator()
            loc.data_set = surface
            loc.build_locator()
            #import pdb; pdb.set_trace()
            #vphi = ttf.get_data(surface, 'vphi')
            #print(type(vphi), dir(vphi))
            #print(vphi.shape, '\n')

            circlepoints = vtk.vtkPoints()
            for k, theta in enumerate(thetas):
                #print(n, k, end=' ')
                # Calculate end points of line
                p1 = [64, 64, ring_height]
                p2 = [64 + (64 * np.sin(theta)),
                      64 + (64 * np.cos(theta)), ring_height]

                pos = [0.0, 0.0, 0.0]

                # Intersect with line, output is in the pos variable
                loc.intersect_with_line(p1, p2, 0.00001, vtk.mutable(0), pos,
                                        pcoords, subId, cellId, cell)
                #id2 = cell.get_point_id(cellId)
                #id3 = cell.get_point_id(subId)
                #print(n, k, subId, cellId, pos, cellId > len(vphi), id2, id3)
                #if subId != 0:
                #    print(n, k, subId, id2, id3)
                #    print(n, k, subId, cellId)
                #if int(subId) > len(vphi):
                #import pdb; pdb.set_trace()
                theta_pos[n, k] = pos
                circlepoints.InsertPoint(k, pos)
                #try:
                #vphipos[n, k] = vphi[cell.get_point_id(subId)]#subId]#cellId/2]
                #except IndexError:
                #    pass
            circlepoly = vtk.vtkPolyData()
            circlepoly.SetPoints(circlepoints)
            poly_norms = ttf.make_poly_norms(circlepoly)
            vfilter, vel = ttf.interpolate_vectors(vfield.outputs[0], poly_norms.output)
            """bfilter, bfield = ttf.interpolate_vectors(bfield.outputs[0], poly_norms.output)
            surface_bfield = ttf.update_interpolated_vectors(False, bfilter)
            normals, torsionals, parallels = ttf.get_surface_vectors(poly_norms, bfield)
            vperp, vpar, vphi = ttf.get_surface_velocity_comp(vel, normals, torsionals, parallels)"""
            vphi[n] = (vel[:, 1] * np.cos(thetas)) - (vel[:, 0] * np.sin(thetas))
            #import pdb; pdb.set_trace()
            del vel, vfilter
        #import pdb; pdb.set_trace()

        if mpi:
            theta_pos_r0 = comm.gather(theta_pos, root=0)
        else:
            theta_pos_r0 = theta_pos[None]

        if rank == 0:
            theta_pos = np.concatenate(theta_pos_r0)

            theta_pos2 = theta_pos - [64, 64, 0]
            dis = np.sqrt(theta_pos2[:, :, 0]**2 + theta_pos2[:, :, 1]**2)

            np.save(os.path.join(cfg.data_dir, tube_r, "{}_h{}_theta_pos.npy".format(cfg.get_identifier(), ring_height)),
                    theta_pos2)
            np.save(os.path.join(cfg.data_dir, tube_r, "{}_h{}_distance.npy".format(cfg.get_identifier(), ring_height)),
                    dis)
            np.save(os.path.join(cfg.data_dir, tube_r, "{}_h{}_vphi.npy".format(cfg.get_identifier(), ring_height)),
                    vphi)
