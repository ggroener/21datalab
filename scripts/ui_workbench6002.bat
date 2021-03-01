setlocal
set "HTTP_PROYX="
set "HTTPS_PROXY="
set "http_proxy="
set "https_proxy="

set mydrive=%~d0
%mydrive%
set mypath=%~dp0
cd %mypath%\..
start bokeh serve bokeh_web --allow-websocket-origin="*" --port 5007 --args http://127.0.0.1:6002/ root.visualization.workbench    

endlocal