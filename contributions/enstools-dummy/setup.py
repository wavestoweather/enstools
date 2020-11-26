from setuptools import setup
import re
import sys
import shlex


def get_settings():
    result = {}
    with open("package.conf") as f:
        tokens = list(shlex.shlex(f.read()))
        for i, t in enumerate(tokens):
            if t == "=" and 0 < i < len(tokens) - 1:
                result[tokens[i-1]] = tokens[i+1]
    return result


def get_version(settings):
    """
    read version string from enstools package without importing it

    Returns
    -------
    str:
            version string
    """
    with open(f"enstools/{settings['PACKAGE_NAME']}/__init__.py") as f:
        for line in f:
            match = re.search('__version__\s*=\s*"([a-zA-Z0-9_.]+)"', line)
            if match is not None:
                return match.group(1)


# read settings for this package
settings = get_settings()

# only print the version and exit?
if len(sys.argv) == 2 and sys.argv[1] == "--get-version":
    print(get_version(settings))
    exit()

# perform the actual install operation
setup(name=f"enstools-{settings['PACKAGE_NAME']}",
      version=get_version(settings),
      author="Your Name",
      author_email="your.name@your.institution.org",
      packages=[f"enstools.{settings['PACKAGE_NAME']}"],
      namespace_packages=['enstools'],
)
