from os.path import isfile, isdir
import numpy as np
from scipy.stats.stats import pearsonr
from typing import Union
from enstools.io import read
from xarray import Dataset, DataArray
from pathlib import Path

def inter_quartile_range(array: np.ndarray) -> float:
        return np.quantile(array, 0.75) - np.quantile(array, 0.25)


def value_range(array: np.ndarray) -> float:
        return np.max(array) - np.min(array)


class DataArrayMetrics:
    """
    First object oriented approach to avoid recomputation of metrics. (i.e. don't compute mse several times)
    """
    available_metrics = [
        "acumulated_difference",
            "mse",
            "rmse",
            "nrmse",
            "nrmse_I",
            "correlation",
            "correlation_I",
            "psnr",
            "ssim",
            "ssim_I",
            "gradient",
    ]

    def __init__(self, reference:Union[DataArray,np.ndarray], target:Union[DataArray,np.ndarray]) -> None:
        if isinstance(reference, DataArray):
            self.reference = reference
            self.reference_values = reference.values
        elif isinstance(reference, np.ndarray):
            self.reference = None
            self.reference_values = reference
        if isinstance(target, DataArray):
            self.target = target
            self.target_values = target.values
        elif isinstance(target, np.ndarray):
            self.target = None
            self.target_values = target
        
        self.difference = self.target_values - self.reference_values

        self.metric_values = {}
    
    def __getitem__(self, name: str) -> float:
        try:
            return self.metric_values[name]
        except KeyError:
            self.metric_values[name] = self.set_metric(name)
            return self.metric_values[name]

    def set_metric(self, method: str)-> None:
        """
        Inputs:
            target: numpy.ndarray
            reference: numpy.ndarray
        """
        if method == "acumulated_difference":
            return self.acumulated_difference()
        elif method == "range_I":
            return self.range_index()
        elif method == "mse":
            return self.mean_square_error()
        elif method == "rmse":
            return self.root_mean_square_error()
        elif method == "nrmse":
            return self.normalized_root_mean_square_error()
        elif method == "nrmse_I":
            nrmse = self["nrmse"]
            if nrmse > 0:
                return - np.log10(nrmse)
            else:
                return np.inf
        elif method == "correlation":
            return self.pearson_correlation()
        elif method == "correlation_I":
            corr = self["correlation"]
            # We expect the correlation values close to 1, in order to have a meaningful value
            # I suggest returning the logarithm of 1-corr
            if corr == 1.0:
                return np.inf
            else:
                return - np.log10(1-corr)
        elif method == "psnr":
            return self.calculate_data_psnr()
        elif method == "ssim":
            return self.compute_ssim()
        elif method == "ssim_I":
            ssim = self["ssim"]
            # We expect the ssim values close to 1, in order to have a meaningful value
            # I suggest returning the logarithm of 1-ssim
            if ssim >= 1.0:
                return np.inf
            else:
                return - np.log10(1-ssim)
        elif method == "gradient":
            return self.compute_gradient_error()
        else:
            raise NotImplementedError(f"The method'{method}' has not been implemented.\
            Available methods are: acumulated_difference, mse, rmse, nrmse, correlation, psnr and ssim.")
    def acumulated_difference(self):
        return np.sum(self.difference)

    def range_index(self):
        """
        It returns the range of the error (max - min) divided by the Inter Quartile Range.
        """
        difference_range = np.max(self.difference) - np.min(self.difference)
        iqr = inter_quartile_range(self.reference_values)
        if iqr == 0.:
            return np.inf
        else:
            return - np.log10(difference_range / iqr)
    ################################################################
    # MSE, RMSE and NRMSE
    #
    def mean_square_error(self):
        """
        It returns the mean square error.
        """
        return np.mean(self.difference**2)

    def root_mean_square_error(self):
        """
        It returns the root mean square error.
        """
        return np.sqrt(self["mse"])

    def normalized_root_mean_square_error(self, method:str="iqr"):
        """
        Normalized RMSE with two normalization methods, absolute range and inter quartile range (less sensitive to outliers)
        """
        import warnings

        reference = self.reference_values
        if method == "iqr":
            normalization_range = inter_quartile_range(reference)
        elif method == "range":
            normalization_range = value_range(reference)
        else:
            raise NotImplementedError
        
        if normalization_range != 0.:
            return self["rmse"] / normalization_range
        else:
            return self["rmse"]

    def pearson_correlation(self):
        """
        It returns the Pearson's correlation. Uses implementation from scipy.
        """
        corr, pval = pearsonr(self.reference_values.ravel(), self.target_values.ravel())
        return corr
        

    def calculate_data_psnr(self ):
        """
        It returns the Peak Signa to Noise Ratio.
        With images, it is usually defined as:
        20*log10(max_value)-10*log10(mse)
        with all the values being between 0 and max_value.
        However, for real values I found this definition:
        20*log10(range)-10*log10(mse)
        """
        # Compute range from the reference file
        ref_min, ref_max = np.min(self.reference_values), np.max(self.reference_values)
        ref_range = ref_max - ref_min
        
        return 20*log10(ref_range)-10*log10(self["mse"])

    @staticmethod
    def compute_ssim_slice(target:np.ndarray, reference:np.ndarray):
        """
        Returns the SSIM of a data slice. It uses the structural_similarity function from skimage.metrics.
        """
        from skimage.metrics import structural_similarity
        import warnings
        ref_min, ref_max = np.min(reference), np.max(reference)
        target_min, target_max = np.min(target), np.max(target)

        true_min = min(ref_min, target_min)
        true_max = min(ref_max, target_max)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ssim = structural_similarity(target, reference, data_range=true_max-true_min)
        return ssim
        
    def compute_ssim(self):
        """
        Returns the SSIM of the full DataArray. The computation its done slice by slice.
        """
        target = self.target_values
        reference = self.reference_values
        if len(target.shape) == 4:
            t,z,x,y = target.shape
            slices = [(_t, _z, slice(None), slice(None)) for _t in range(t) for _z in range(z)]
        elif len(target.shape) == 3:
            t, x,y = target.shape
            slices = [(_t, slice(None), slice(None)) for _t in range(t)]
        elif len(target.shape) == 2:
            slices = [(slice(None), slice(None))]
        elif len(target.shape) == 1:
            slices = [(slice(None))]
        else:
            raise NotImplementedError
        ssim = [self.compute_ssim_slice(target[sl], reference[sl]) for sl in slices]

        mean_ssim = np.mean(ssim)
        return mean_ssim
        


    def plot_summary(self, output_folder:str="report"):
        import matplotlib.pyplot as plt
        import matplotlib as mpl

        from os.path import isdir, join
        from os import makedirs


        # Get dimensions  
        shape = self.reference.shape
        # Get variable name from DataArray object
        variable_name = self.reference.name

        if len(shape) == 4:
            y,z,x,y = shape
            sl = (0,int(z/2),slice(None),slice(None))
        elif len(shape) == 3:
            sl = 0, slice(None),slice(None)
        elif len(shape) == 2:
            sl = slice(None),slice(None)
        elif len(shape) == 1:
            return 
            sl = slice(None)
        else:
            pass

        plot_num = 4
        reference_data = self.reference[sl]
        target_data = self.target[sl]

        iqr = inter_quartile_range(reference_data)

        # Prepare a plot of an intermediate level
        plt.figure(figsize=(9,9))

        # Plot reference
        plt.subplot(int(f"{plot_num}11"))
        plt.imshow(reference_data)
        plt.colorbar()
        # Plot comparison target
        plt.subplot(int(f"{plot_num}12"))
        plt.imshow(target_data)
        plt.colorbar()

        # Plot differences
        plt.subplot(int(f"{plot_num}13"))
        color_levels = 7
        cmap = plt.cm.seismic  # define the colormap
        # extract all colors from the colormap
        cmaplist = [cmap(i) for i in range(cmap.N)]
        # Generate new colormap with only few levels
        cmap = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', cmaplist, color_levels)
        difference = self.difference[sl]
        _min = np.min(difference)
        _max = np.max(difference)
        __ = max(abs(_min), _max)
        factor = iqr/__
        vmin = -__
        vmax = __
        #vmin =-iqr/10
        #vmax = iqr/10
        plt.imshow(difference, vmin=vmin, vmax=vmax, cmap=cmap)
        plt.title(f"The difference range is {factor:.1f} smaller than the InterQuartileRange")
        cbar = plt.colorbar()
        cbar.set_ticks(np.linspace(vmin, vmax, color_levels+1))

        # Put something related with the metrics
        fig = plt.gcf()
        ax = fig.add_subplot(414, polar=True)

        
        selected_metrics = ["correlation_I", "ssim_I", "nrmse_I"]
        selected_values = np.array([ self[m] for m in selected_metrics ])
        make_radar_chart(variable_name, selected_metrics, selected_values, ax)
        if not isdir(output_folder):
            makedirs(output_folder)
        plt.savefig(join(output_folder, f"report_{variable_name}.png"))
        plt.close("all")


class DatasetMetrics:
    def __init__(self, reference_path:str, target_path:str) -> None:
        # Check that files exist and save them
        assert isfile(reference_path), f"Path {reference_path} its not a valid file."
        assert isfile(target_path), f"Path {target_path} its not a valid file."
        
        self.reference_path = reference_path
        self.target_path = target_path

        # Read files
        self.reference = read(self.reference_path)
        self.target = read(self.target_path)

        # Get variables
        self.coords = [v for v in self.reference.coords]
        self.variables = [v for v in self.reference.variables if v not in self.coords]
        

        # Consistency check
        self.check_consistency()

        # Initialize metrics
        self.metrics = {}
        self.initialize_metrics()


    def check_consistency(self) -> None:
        # Check that files are meant to represent the same things
        assert set(self.reference.variables) == set(self.target.variables)
        assert set(self.reference.coords) == set(self.target.coords)

    def initialize_metrics(self) -> None:
        for variable in self.variables:
            self.metrics[variable] = DataArrayMetrics(self.reference[variable], self.target[variable])
        pass
    
    def __getitem__(self, name: str) -> DataArrayMetrics:
        assert name in self.variables, f"The provided variable name {name} does not exist in this dataset."
        return self.metrics[name]

    def make_plots(self):
        print("Producing plots:")
        for index, variable in enumerate(self.variables):
            print(f"\r{index+1}/{len(self.variables)} {variable:30} ",end="")
            self[variable].plot_summary()
        print("\rPlots done!"+30*" ")
    
    def create_gradients(self):
        """
        Create first order gradients
        """
        for variable in self.variables:
            ref_grad = array_gradient(self[variable].reference)
            target_grad = array_gradient(self[variable].target)
            if ref_grad is not None:
                self.reference[ref_grad.name] = ref_grad
                self.target[target_grad.name] = target_grad
        
        # Update list of variables to include the gradients
        self.variables = [v for v in self.reference.variables if v not in self.reference.coords]
        self.initialize_metrics()

    def create_second_order_gradients(self):
        """
        Create first order gradients
        """
        for variable in [v for v in self.variables if v.count("_gradient")]:
            ref_grad = array_gradient(self[variable].reference)
            target_grad = array_gradient(self[variable].target)
            ref_grad.name = ref_grad.name.replace("_gradient"*2, "_gradient_O2")
            target_grad.name = target_grad.name.replace("_gradient"*2, "_gradient_O2")
            if ref_grad is not None:
                self.reference[ref_grad.name] = ref_grad
                self.target[target_grad.name] = target_grad
        
        # Update list of variables to include the gradients
        self.variables = [v for v in self.reference.variables if v not in self.reference.coords]
        self.initialize_metrics()
    


def make_radar_chart(name, attribute_labels, stats, ax):
    import matplotlib.pyplot as plt
    markers = [0,3,6,9,12]
    str_markers = ["0", "3", "6", "9", "12"]

    labels = np.array(attribute_labels)

    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
    stats = np.concatenate((stats,[stats[0]]))
    angles = np.concatenate((angles,[angles[0]]))
    
    ax.plot(angles, stats, 'o-', linewidth=.5,alpha=1,markersize=5)
    ax.fill(angles, stats, alpha=0.25)
    ax.set_thetagrids(angles[:-1] * 180/np.pi, labels)
  
    ax.plot(angles,[5,3, 2, 5], ":", color="r", alpha=.5)
    plt.yticks(markers)
    ax.grid(True)


def array_gradient(data_array: DataArray) -> DataArray:
    """
    Takes one Data Array and returns the gradient.
    In case of multidimensional data, it returns the magnitude of the gradient.

    Maybe it shouldn't be in this file.
    """
    dimensions = len(data_array.shape)
    if dimensions == 1:
        return None
    

    gradient_axis = tuple(range(1,dimensions))
    
    new_array = data_array.copy()
    data_values = data_array.values
    try:
        gradient_vect = np.gradient(data_values,axis=gradient_axis)
    except Exception as err:
        print(err)
        return None
    
    gradient_norm = np.linalg.norm(gradient_vect, axis=0)

    new_array.values = gradient_norm

    new_array.name = f"{new_array.name}_gradient"
    return new_array

###############################################################
# Functions to calculate compression ratio from files
# 
def file_size(file_path):
    return Path(file_path).stat().st_size


def convert_size(size_bytes):
   if size_bytes == 0:
       return "0B"
   size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
   i = int(math.floor(math.log(size_bytes, 1024)))
   p = math.pow(1024, i)
   s = round(size_bytes / p, 2)
   return "%s %s" % (s, size_name[i])

def readable_size(file_path):
    return convert_size(file_size(file_path))

def compression_ratio(path_to_original, path_to_compressed):
    original_size = file_size(path_to_original)
    compressed_size = file_size(path_to_compressed)
    return  original_size / compressed_size
