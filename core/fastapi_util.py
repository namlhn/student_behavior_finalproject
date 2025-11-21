from fastapi import Request, Response, status, HTTPException, APIRouter, params
from fastapi.datastructures import Default
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute, routing, BaseRoute, ASGIApp
from fastapi.utils import (
    generate_unique_id,
)
import sys
import time
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Type,
    Union,
)

from core.config import settings
from core.fastapi_logger import log_data


def get_request_ip(request):
    ip = str(request.client.host)
    # Note: HTTP_X_FORWARDED_FOR might be a list of addresses! NOTE: We only trust the LAST address because it's the
    # proxy address. All IPs before the last one can be mocked by sender.
    if ',' in ip:
        ip = [x for x in [x.strip() for x in ip.split(',')] if x][-1]
    return ip


class LogRequestRoute(APIRoute):
    def get_route_handler(self) -> Callable:

        original_route_handler = super().get_route_handler()

        async def custom_route_handler(request: Request) -> Response:
            start = time.time()
            ex = None
            try:
                response: Response = await original_route_handler(request)
            except RequestValidationError as exc:
                # ex = exc
                ex_type, ex_value, ex_traceback = sys.exc_info()
                log_data.exception('%s_exception_error validation', self.name)
                log_data.warning('view_params_error|format=,url=%s,error=%s',
                                 request.url, ex_value)
                response: Response = api_response_error_params()
            except Exception as e:
                ex = e
                ex_type, ex_value, ex_traceback = sys.exc_info()
                log_data.exception('%s_exception', self.name)

            end = time.time()
            elapsed = int((end - start) * 1000)
            content_type_header = request.headers.get("Content-Type")
            if content_type_header is None:
                content_type_header = "content-type-empty"
            if settings.log_request_body:
                if hasattr(request, "_body"):
                    body = await request.body()
                    # text: str = bytes.decode(body)
                    request_body = request.method + ": " + \
                        content_type_header + "|" + str(body)
                elif hasattr(request, '_form'):
                    # form_data = await request.form()
                    request_body = request.method + ": " + content_type_header
                else:
                    request_body = request.method + ": " + content_type_header
                if settings.max_request_body_length and len(request_body) > settings.max_request_body_length:
                    request_body = request_body[:settings.max_request_body_length] + '...'
            else:
                request_body = ''

            if ex is None:
                status_code = response.status_code
                response_body: str = ""
                if settings.log_response:
                    response_body = response.body.decode()
                    if settings.max_response_length and len(response_body) > settings.max_response_length:
                        response_body = response_body[:settings.max_response_length] + '...'
                else:
                    response_body = ''
            else:
                if isinstance(ex, HTTPException):
                    status_code = ex.status_code
                    response_body = 'exception:%s' % str(ex.detail)
                else:
                    status_code = 500
                    response_body = 'exception:%s' % str(ex_value)

            request_id = getattr(request.state, 'request_id', '-')
            log_data.data('http_request|request_id=%s,ip=%s,elapsed=%d,method=%s,url=%s,body=%s,status_code=%d,response=%s',
                          request_id, get_request_ip(
                              request), elapsed, request.method, request.url.path,
                          request_body, status_code, response_body)

            if ex is not None:
                if isinstance(ex, HTTPException):
                    raise HTTPException(
                        status_code=ex.status_code, detail=ex.detail)
                else:
                    raise HTTPException(
                        status_code=status_code, detail="bad request")
            return response

        return custom_route_handler


def api_response_data(result_code: str, reply: Optional[Any] = None, message: str = None):
    response_data = {"result": result_code, "reply": jsonable_encoder(reply)}
    if message:
        response_data["message"] = message
    response = JSONResponse(response_data, status_code=status.HTTP_200_OK)
    response.headers["Content-Type"] = 'application/json; charset=utf-8'
    return response


def api_simple_response(reply: Optional[Any] = None):
    response = JSONResponse(jsonable_encoder(
        reply), status_code=status.HTTP_200_OK)
    response.headers["Content-Type"] = 'application/json; charset=utf-8'
    # response = JSONResponse({"result": result_code, "reply": reply}, status_code=status.HTTP_200_OK)
    return response


def api_response_error_params():
    response = JSONResponse({"result": "error_params"})
    return response


class AppRouter(APIRouter):
    def __init__(
            self,
            *,
            prefix: str = "",
            tags: Optional[List[Union[str, Enum]]] = None,
            dependencies: Optional[Sequence[params.Depends]] = None,
            default_response_class: Type[Response] = Default(JSONResponse),
            responses: Optional[Dict[Union[int, str], Dict[str, Any]]] = None,
            callbacks: Optional[List[BaseRoute]] = None,
            routes: Optional[List[routing.BaseRoute]] = None,
            redirect_slashes: bool = True,
            default: Optional[ASGIApp] = None,
            dependency_overrides_provider: Optional[Any] = None,
            on_startup: Optional[Sequence[Callable[[], Any]]] = None,
            on_shutdown: Optional[Sequence[Callable[[], Any]]] = None,
            deprecated: Optional[bool] = None,
            include_in_schema: bool = True,
            generate_unique_id_function: Callable[[APIRoute], str] = Default(
                generate_unique_id
            ),
    ) -> None:
        super().__init__(route_class=LogRequestRoute, prefix=prefix, tags=tags, dependencies=dependencies,
                         default_response_class=default_response_class, responses=responses, callbacks=callbacks,
                         routes=routes, redirect_slashes=redirect_slashes, default=default,
                         dependency_overrides_provider=dependency_overrides_provider, on_startup=on_startup,
                         on_shutdown=on_shutdown, deprecated=deprecated, include_in_schema=include_in_schema,
                         generate_unique_id_function=generate_unique_id_function)
