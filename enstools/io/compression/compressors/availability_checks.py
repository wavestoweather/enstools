#############################################
# Availability checks
# Few functions to check whether certain plugins or libraries are available

def check_all_filters_availability(try_load_hdf5plugin=False):
    """
    Function to check that all the filters of interest are available.
    :return:  bool
    """
    blosc_available = check_blosc_availability()
    zfp_available = check_zfp_availability()
    sz_available = check_sz_availability()
    all_available = blosc_available and zfp_available and sz_available
    if all_available:
        return True
    else:
        if try_load_hdf5plugin:
            import hdf5plugin
            return check_all_filters_availability(try_load_hdf5plugin=False)
        else:
            return False


def check_blosc_availability():
    """
    Function to check that the BLOSC hdf5 filter is available.
    :return: bool
    """
    blosc_filter_id = 32001
    return check_filter_availability(blosc_filter_id)


def check_zfp_availability():
    """
    Function to check that the ZFP hdf5 filter is available.
    :return: bool
    """
    zfp_filter_id = 32013
    return check_filter_availability(zfp_filter_id)


def check_sz_availability():
    """
    Function to check that the SZ hdf5 filter is available.
    :return: bool
    """
    sz_filter_id = 32017
    return check_filter_availability(sz_filter_id)


def check_filter_availability(filter_id):
    """
    Use h5py to check if a filter is available
    :param filter_id: Id of the target filter
    :return: bool
    """
    import h5py
    return h5py.h5z.filter_avail(filter_id)


def check_libpressio_availability():
    try:
        from libpressio import PressioCompressor
        return True
    except ModuleNotFoundError as err:
        return False


def filter_availability_report():
    if check_blosc_availability():
        print("Filter BLOSC is available")
    else:
        print("Filter BLOSC is NOT available")

    if check_zfp_availability():
        print("Filter ZFP is available")
    else:
        print("Filter ZFP is NOT available")

    if check_sz_availability():
        print("Filter SZ is available")
    else:
        print("Filter SZ is NOT available")


def check_compression_filters_availability(dataset):
    # Check filter availability
    import re
    filters_in_dataset = []
    # Check if the variables have a compression attribute
    for var in dataset.variables:
        if "compression" in dataset[var].attrs.keys():
            compression_spec = dataset[var].attrs["compression"]
            m = re.search("id:([0-9]+)", compression_spec)
            if m:
                comp_id = int(m.group(1))
                filters_in_dataset.append(comp_id)

    # Set of filters
    filters_in_dataset = set(filters_in_dataset)

    # Check availability filter by filter
    for filter_id in filters_in_dataset:
        if filter_id == 32017:
            if not check_sz_availability():
                return False
        elif filter_id == 32013:
            if not check_zfp_availability():
                return False
        elif filter_id == 32001:
            if not check_blosc_availability():
                return False
    return True
