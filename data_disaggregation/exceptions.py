class BaseException(Exception):
    def __init__(self, message):
        # some of our messages are long, so we write
        # them in multi line strings
        # so we want to remove the indented spaces
        message = "\n".join(s.strip() for s in message.splitlines())
        super().__init__(message)


class AggregationError(BaseException):
    pass


class DuplicateNameError(BaseException):
    pass


class DimensionStructureError(BaseException):
    pass
