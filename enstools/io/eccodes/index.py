try:
    import eccodes
except ImportError:
    pass
import os
import threading
import atexit


# cache for already available indices
_cache = {}
_index_files_to_delete = set()

# the grib_api is not thread-safe, ensure that iterators are not interrupted
_locks = {}


class GribIndexHelper:
    """
    create and cache an index for a grib file
    """
    def __init__(self, filename):
        """
        create a grib index for the file *filename*

        Parameters
        ----------
        filename : string
                name of the grib file to index
        """
        # create grib-index of the input file, only the name is used
        self.filename = os.path.abspath(filename)
        self.index_keys = ["shortName", "typeOfLevel", "level", "validityDate", "validityTime"]
        if (self.filename, os.getpid()) in _cache:
            self.iid, self.index_vals = _cache[(self.filename, os.getpid())]
        else:
            # try to load the index from a index file
            self.iid = self.load_from_file()
            # not successful? create a new index!
            if self.iid is None:
                self.iid = eccodes.codes_index_new_from_file(self.filename, self.index_keys)
                # write the index to a file
                self.store_to_file()

            # get all possible index values
            self.index_vals = self.__get_index_values()
            _cache[(self.filename, os.getpid())] = (self.iid, self.index_vals)

    def store_to_file(self):
        """
        store the index in the default cache location
        """
        # construct a name for the index file
        cache_file_name = self.__create_cache_file_name()

        # write the index
        eccodes.codes_index_write(self.iid, cache_file_name)

    def load_from_file(self):
        """
        load an grib index created before by store_to_file
        """
        # construct a name for the index file
        cache_file_name = self.__create_cache_file_name()

        # load if possible
        if os.path.exists(cache_file_name):
            return eccodes.codes_index_read(cache_file_name)
        else:
            return None

    def __create_cache_file_name(self):
        cache_dir = os.path.join(os.getenv("HOME"), ".enstools/grib_index")
        if not os.path.isabs(cache_dir):
            raise IOError("the cache dir for the grib index is not an absolute path!")
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        # checksum for keys and file properties
        chk_sum = hash(tuple(self.index_keys) + (os.path.getsize(self.filename),
                                                 os.path.getmtime(self.filename),
                                                 eccodes.codes_get_api_version(),
                                                 os.getenv("ECCODES_DEFINITION_PATH", os.getenv("GRIBAPI_DEFINITION_PATH"))))

        # construct a name for the index file
        index_file = os.path.join(cache_dir, "%s_%d.idx" % (self.filename.replace("/", "_"), chk_sum))

        # remove old index files from cache
        index_files = os.listdir(cache_dir)
        max_cache_files = 100
        if len(index_files) > max_cache_files:
            # sort by date an d delete the oldest
            try:
                index_with_date = list(map(lambda x: (os.path.getmtime(os.path.join(cache_dir, x)), os.path.join(cache_dir, x)), index_files))
                index_with_date.sort(key=lambda x: x[0], reverse=True)
                for ifile in range(len(index_files) - max_cache_files):
                    if index_with_date[ifile][1] != index_file:
                        _index_files_to_delete.add(index_with_date[ifile][1])
            except (IOError, OSError):
                pass

        return index_file

    def __get_index_values(self):
        # get all possible values for the index keys
        index_vals = []
        for key in self.index_keys:
            key_vals = eccodes.codes_index_get(self.iid, key)
            index_vals.append(key_vals)
        return index_vals


def get_lock():
    if os.getpid() in _locks:
        return _locks[os.getpid()]
    else:
        _locks[os.getpid()] = threading.Lock()
        return _locks[os.getpid()]


def clean_index_files():
    """
    remove old index files
    """
    for one_file in _index_files_to_delete:
        os.remove(one_file)

# cleanup at exit of the script
atexit.register(clean_index_files)
