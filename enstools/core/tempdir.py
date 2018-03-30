import tempfile
import shutil
import atexit
import os


class TempDir(object):
    """
    This class represents a temporal directory which is automatically deleted on exit
    """

    def __init__(self, parentdir=None, check_free_space=True, cleanup=True):
        """
        Parameters
        ----------
        parentdir : str
                the directory is created in the parent directory. If the argument is none, then the directory is created in the default temp-Directory

        check_free_space : bool
                if true, at least 10GB free space in the temporal directory if neccessary

        cleanup: bool
                if True, the folder will be deleted at exit of python.
        """
        # create the directory
        if parentdir is not None and not os.path.exists(parentdir):
            try:
                os.makedirs(parentdir)
            except:
                raise IOError("creating temporal folder in %s failed!" % parentdir)
        self.__path = tempfile.mkdtemp(dir=parentdir)

        # check the free disk space
        if check_free_space:
            stat = os.statvfs(self.__path)
            if stat.f_frsize * stat.f_bavail < 1e10:
                raise IOError("""less than 10GB disk space for temporal directory!
                                 set one of the environment variables TMPDIR, TEMP, or TMP to specify a location with more space available!""")

        # register the cleanup-method
        if cleanup:
            atexit.register(self.cleanup)

    def cleanup(self):
        """
        Remove the temporal directory and its content.
        This method is automatically called on exit
        """
        # check is the directory is still there
        if not os.path.exists(self.__path):
            return

        # delete the directory
        shutil.rmtree(self.__path)

    def getpath(self):
        """
        get the path of the temporal directory
        """
        return self.__path
