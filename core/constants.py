from enum import Enum, IntEnum

ASYNC_REQUEST_PAGE_SIZE = 1000

DATE_FORMAT = "%Y-%m-%d"
TIME_FORMAT = "%H:%M:%S"
HOUR_FORMAT = "%H:%M"
DATETIME_FORMAT = DATE_FORMAT + " " + TIME_FORMAT
DATETIME_HOUR_FORMAT = DATE_FORMAT + " " + HOUR_FORMAT

MIN_QUERY_SIZE = 0
MAX_QUERY_SIZE = 2000


class MethodCode(str, Enum):
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"


class ResponseCode(IntEnum):
    SUCCESS = 200
    INVALID_LOGIN_TOKEN = 203
    DATA_NOT_EXIST = 205
    OBJECT_NOT_FOUND = 404
    OBJECT_WAITING_APPROVAL = 405
    INTERNAL_SERVER_ERROR = 500


class Result(str, Enum):
    SUCCESS = "success"
    ERROR_PARAMS = "error_params"
    ERROR_HEADER = "error_header"
    ERROR_SERVER = "error_server"
    ERROR_EXIST = "error_exist"
    ERROR_FILE_NONE = "error_file_none"
    ERROR_NOT_FOUND = "error_not_found"
    ERROR_FORBIDDEN = "error_forbidden"
    ERROR_ACCESS_TOKEN = "error_access_token"
    ERROR_AUTH = "error_auth"
    ERROR_PASSWORD_FORMAT_WRONG = "error_password_format_wrong"
