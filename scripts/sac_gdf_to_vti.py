import os
import yt
from pysac.analysis.tube3D import process_utils as pu
import pysac.yt
import numpy as np
from tvtk.api import tvtk
import tvtk.common as tvtk_common


def parse_name(filename):
    fprefix = os.path.splitext(os.path.basename(filename))[0]
    return fprefix + '.vti'


globpattern = "./m-1_*_00???.gdf"

ts = yt.load(globpattern)

cube_slice = np.s_[2:-2, 2:-2]

field, vfield = pu.get_yt_mlab(ts[0], cube_slice, flux=False)

wr = tvtk.XMLImageDataWriter()
tvtk_common.configure_input(wr, vfield.image_data)

for ds in ts[1:]:
    pu.update_yt_to_mlab_vector(vfield, ds, 'velocity_x', 'velocity_y',
                                'velocity_z', cube_slice=cube_slice)

    wr.file_name = parse_name(ds.filename)
    print(wr.file_name)
    wr.write()
