class EnstoolsError(Exception):
    """Base class for Enstools Errors"""


class WrongCompressionSpecificationError(EnstoolsError):
    """Raised when the compression specification is not valid"""


class WrongCompressionModeError(WrongCompressionSpecificationError):
    """Raised when the compression mode is not valid"""
