import unittest
from os.path import isfile, join


def create_synthetic_dataset(directory):
    import numpy as np
    import xarray as xr
    import pandas as pd
    # Create synthetic datasets
    nx, ny, nz, t = 100, 100, 100, 10
    lons = np.linspace(-180, 180, nx)
    lats = np.linspace(-90, 90, ny)
    lon, lat = np.meshgrid(lons, lats)
    levels = np.array(range(nz))
    for dimension in [1, 2, 3]:
        if dimension == 1:
            data_size = (nx, t)
            var_dimensions = ["x", "time"]
            coord_dimensions = ["x"]
            coord_lon = lons
            coord_lat = lats
        elif dimension == 2:
            data_size = (nx, ny, t)
            var_dimensions = ["x", "y", "time"]
            coord_dimensions = ["x", "y"]
            coord_lon = lon
            coord_lat = lat
        elif dimension == 3:
            data_size = (nx, ny, nz, t)
            var_dimensions = ["x", "y", "z", "time"]
            coord_dimensions = ["x", "y"]
            coord_lon = lon
            coord_lat = lat

        temp = 15 + 8 * np.random.randn(*data_size)
        precip = 10 * np.random.rand(*data_size)

        ds = xr.Dataset(
            {
                "temperature": (var_dimensions, temp),
                "precipitation": (var_dimensions, precip),
            },

            coords={
                "lon": (coord_dimensions, coord_lon),
                "lat": (coord_dimensions, coord_lat),
                "level": ("z", levels),
                "time": pd.date_range("2014-09-06", periods=t),
                "reference_time": pd.Timestamp("2014-09-05"),
            },
        )
        ds_name = "dataset_%iD.nc" % dimension
        ds.to_netcdf(join(directory, ds_name))


def setup():
    from enstools.core.tempdir import TempDir
    # Create temporary directory in which we'll put some synthetic datasets
    temp_input_dir = TempDir()
    temp_output_dir = TempDir()
    create_synthetic_dataset(temp_input_dir.getpath())
    return temp_input_dir, temp_output_dir


def launch_bash_command(command):
    from subprocess import Popen, PIPE
    split_command = command.split(" ")
    p = Popen(split_command, stdout=PIPE)
    p.communicate()
    p.wait()
    return p.returncode


class MyTestCase(unittest.TestCase):
    def test_dataset_exists(self):
        tempdir_path = input_tempdir.getpath()

        datasets = ["dataset_%iD.nc" % dimension for dimension in range(1, 4)]
        for ds in datasets:
            self.assertTrue(isfile(join(tempdir_path, ds)))

    def test_bash(self):
        tempdir_path = input_tempdir.getpath()
        command = "ls %s" % tempdir_path
        return_code = launch_bash_command(command)
        self.assertFalse(return_code)

    def test_command_is_available(self):
        command = "enstools-compressor -h"
        return_code = launch_bash_command(command)
        self.assertFalse(return_code)

    def test_compress_vanilla(self):
        # Check that the compression without specifying compressio parameters works
        datasets = ["dataset_%iD.nc" % dimension for dimension in range(1, 4)]
        for ds in datasets:
            input_path = join(input_tempdir.getpath(), ds)
            output_path = output_tempdir.getpath()
            command = f"enstools-compressor compress {input_path} -o {output_path}"
            return_code = launch_bash_command(command)
            self.assertFalse(return_code)

    def test_compress_lossless(self):
        # Check that compression works when specifying compression = lossless
        datasets = ["dataset_%iD.nc" % dimension for dimension in range(1, 4)]
        compression = "lossless"
        for ds in datasets:
            input_path = join(input_tempdir.getpath(), ds)
            output_path = output_tempdir.getpath()
            command = f"enstools-compressor compress {input_path} -o {output_path} --compression {compression}"
            return_code = launch_bash_command(command)
            self.assertFalse(return_code)

    def test_compress_lossy(self):
        # Check that compression works when specifying compression = lossless
        datasets = ["dataset_%iD.nc" % dimension for dimension in range(1, 4)]
        compression = "lossy"
        for ds in datasets:
            input_path = join(input_tempdir.getpath(), ds)
            output_path = output_tempdir.getpath()
            command = f"enstools-compressor compress {input_path} -o {output_path} --compression {compression}"
            return_code = launch_bash_command(command)
            self.assertFalse(return_code)

    def test_compress_sz(self):
        # Check that compression works when specifying compression = lossy:sz
        datasets = ["dataset_%iD.nc" % dimension for dimension in range(1, 4)]
        compression = "lossy:sz"
        for ds in datasets:
            input_path = join(input_tempdir.getpath(), ds)
            output_path = output_tempdir.getpath()
            command = f"enstools-compressor compress {input_path} -o {output_path} --compression {compression}"
            return_code = launch_bash_command(command)
            self.assertFalse(return_code)

    def test_compress_sz_pw_rel(self):
        # Check that compression works when specifying compression = lossy:sz
        datasets = ["dataset_%iD.nc" % dimension for dimension in range(1, 4)]
        compression = "lossy:sz:pw_rel:0.1"
        for ds in datasets:
            input_path = join(input_tempdir.getpath(), ds)
            output_path = output_tempdir.getpath()
            command = f"enstools-compressor compress {input_path} -o {output_path} --compression {compression}"
            return_code = launch_bash_command(command)
            self.assertFalse(return_code)


if __name__ == '__main__':
    input_tempdir, output_tempdir = setup()
    unittest.main()
    input_tempdir.cleanup()
    output_tempdir.cleanup()
