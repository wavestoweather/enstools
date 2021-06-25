from os.path import isfile, join
import pytest
from enstools.io.encoding import check_sz_availability

def create_synthetic_dataset(directory):
    """
    Creates three synthetic netcdf datasets (1d,2d,3d) into the provided directory.
    :param directory:
    :return: None
    """
    import numpy as np
    import xarray as xr
    import pandas as pd
    from scipy.ndimage import gaussian_filter
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
        else:
            raise NotImplementedError()

        temp = 15 + 8 * np.random.randn(*data_size)
        temp = gaussian_filter(temp, sigma=5)
        precip = 10 * np.random.rand(*data_size)
        precip = gaussian_filter(precip, sigma=5)

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


def file_size(file_path):
    from pathlib import Path
    return Path(file_path).stat().st_size

folders = None

def wrapper(cls, compression=None):
        from enstools.io import compress
        input_tempdir = cls.input_tempdir
        output_tempdir = cls.output_tempdir
        # Check that the compression without specifying compression parameters works
        datasets = ["dataset_%iD.nc" % dimension for dimension in range(1, 4)]
        for ds in datasets:
            input_path = join(input_tempdir.getpath(), ds)
            output_path = output_tempdir.getpath()
            # Import and launch compress function
            compress(output_path, [input_path], compression=compression, nodes=0)


class TestClass:
    @classmethod
    def setup_class(cls):
        """
        This code will be executed at the beginning of the tests.
        We will be launching the
        :return:
        """
        """
        Creates two temporary directories:
        - Input directory: Will store the synthetic data created for the test
        - Output directory: Will store the compressed synthetic data
        :return: Tempdir, Tempdir
        """
        from enstools.core.tempdir import TempDir
        # Create temporary directory in which we'll put some synthetic datasets
        cls.input_tempdir = TempDir(check_free_space=False)
        cls.output_tempdir = TempDir(check_free_space=False)
        create_synthetic_dataset(cls.input_tempdir.getpath())
    @classmethod
    def teardown_class(cls):
        # release resources
        cls.input_tempdir.cleanup()
        cls.output_tempdir.cleanup()
    
    def test_dataset_exists(self):
        input_tempdir = self.input_tempdir
        output_tempdir = self.output_tempdir
        tempdir_path = input_tempdir.getpath()

        datasets = ["dataset_%iD.nc" % dimension for dimension in range(1, 4)]
        for ds in datasets:
            assert isfile(join(tempdir_path, ds))



    def test_compress_vanilla(self):
        wrapper(self)

    def test_compress_lossless(self):
        compression = "lossless"
        wrapper(self, compression=compression)
        
    def test_compress_lossy(self):
        compression = "lossy"
        wrapper(self, compression=compression)

    @pytest.mark.skipif( not check_sz_availability(), reason="Requires SZ")
    def test_compress_sz_pw_rel(self):
        compression = "lossy:sz:pw_rel:0.1"
        wrapper(self, compression=compression)

    def test_compress_zfp_vanilla(self):
        compression = "lossy:zfp"
        wrapper(self, compression=compression)

    def test_compress_zfp_rate_1(self):
        compression = "lossy:zfp:rate:1"
        wrapper(self, compression=compression)
        
    def test_compress_json_parameters(self):
        input_tempdir = self.input_tempdir
        import json
        
        datasets = ["dataset_%iD.nc" % dimension for dimension in range(1, 4)]
        compression_parameters = {"default":"lossless",
                                    "temperature": "lossy:zfp:rate:3",
                                    "precipitation": "lossless",
                                    }
        json_file_path = input_tempdir.getpath()+"/compression.json"
        with open(json_file_path, "w") as out_file:
            json.dump(compression_parameters, out_file)
        compression = json_file_path
        wrapper(self, compression=compression)
        
    def test_compress_yaml_parameters(self):
        input_tempdir = self.input_tempdir
        import yaml
        
        datasets = ["dataset_%iD.nc" % dimension for dimension in range(1, 4)]
        compression_parameters = {"default":"lossless",
                                    "temperature": "lossy:zfp:rate:3",
                                    "precipitation": "lossless",
                                    }
        yaml_file_path = input_tempdir.getpath()+"/compression.yaml"
        with open(yaml_file_path, "w") as out_file:
            yaml.dump(compression_parameters, out_file)
        compression = yaml_file_path
        wrapper(self, compression=compression)
        


    def test_compress_auto(self):
        compression="auto"
        wrapper(self, compression=compression)

    def test_compress_ratios_lossy(self):
        from enstools.io import compress
        input_tempdir = self.input_tempdir
        output_tempdir = self.output_tempdir
        # Check that compression works when specifying compression = lossy:sz
        datasets = ["dataset_%iD.nc" % dimension for dimension in range(1, 4)]
        compression = "lossy"
        for ds in datasets:
            input_path = join(input_tempdir.getpath(), ds)
            output_path = output_tempdir.getpath()
            output_file_path = join(output_path, ds)
            compress(output_path, [input_path], compression=compression, nodes=0)    
            initial_size = file_size(input_path)
            final_size = file_size(output_file_path)
            assert initial_size > final_size

    def test_compress_ratios_lossless(self):
        from enstools.io import compress
        # Check that compression works when specifying compression = lossy:sz
        input_tempdir = self.input_tempdir
        output_tempdir = self.output_tempdir

        datasets = ["dataset_%iD.nc" % dimension for dimension in range(2, 4)]
        compression = "lossless"
        for ds in datasets:
            input_path = join(input_tempdir.getpath(), ds)
            output_path = output_tempdir.getpath()
            output_file_path = join(output_path, ds)
            compress(output_path, [input_path], compression=compression, nodes=0)    
            initial_size = file_size(input_path)
            final_size = file_size(output_file_path)
            assert initial_size > final_size
        
    #def test_compress_multinode(self):
    #    # Check that compression works when specifying compression = lossy:sz
    #    input_tempdir = self.input_tempdir
    #    output_tempdir = self.output_tempdir

    #    datasets = ["dataset_%iD.nc" % dimension for dimension in range(1, 4)]
    #    compression = "lossless"
    #    for ds in datasets:
    #        input_path = join(input_tempdir.getpath(), ds)
    #        output_path = output_tempdir.getpath()
    #        output_file_path = join(output_path, ds)
    #        command = f"enstools-compressor compress {input_path} -o {output_path} --compression {compression} -N 2"
    #        #return_code = launch_bash_command(command)
    #        self.assertFalse(False)

    @pytest.mark.skipif( not check_sz_availability(), reason="Requires SZ")
    def test_filters_availability(self):
        from enstools.io.encoding import check_all_filters_availability
        assert check_all_filters_availability()

    def test_blosc_filter_availability(self):
        from enstools.io.encoding import check_blosc_availability
        assert check_blosc_availability

    def test_zfp_filter_availability(self):
        from enstools.io.encoding import check_zfp_availability
        assert check_zfp_availability

    """
    # For now we keep this test disabled since we are not able to easily provide the sz package

    def test_sz_filter_availability(self):
        from enstools.io.encoding import check_sz_availability
        assert check_sz_availability
    """

