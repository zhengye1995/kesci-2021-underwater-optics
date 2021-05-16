"""Tests for the flatten observation wrapper."""
from flask import Flask, request
import json
# import ai_hub.globalvar as gl
import tools.globalvar as gl
import time
app = Flask("tccapi")


def shutdown_server():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()


@app.route('/shutdown', methods=['POST'])
def shutdown():
    shutdown_server()
    return 'Server shutting down...'

@app.route("/tccapi", methods=['GET', 'POST'])
def tccapi():
    ret = ""
    if request.method== 'POST':
        data = request.get_data()
        if data == b"exit":
            print("Server shutting down...")
            shutdown_server()
            return "Server shutting down..."

        # inferserver
        #myinserver = gl.get_value("myinserver")
        inferserver = gl.get_value("inferserver")
        #print(inferserver)

        data_pred = inferserver.pre_process(request)
        ret = inferserver.pridect(data_pred)
        ret = inferserver.post_process(ret)
        if not isinstance(ret, str):
            ret = str(ret)
        # print("return: ", ret)
    else:
        print("please use post request.such as ：curl localhost:8080/tccapi -X POST -d \'{\"img\"/:2}\'")
    return ret


class inferServer():
    def __init__(self, model1, model2):
        self.model1 = model1
        self.model2 = model2
        print("init_Server")
        gl._init()
        # gl.set_value("inferModel", model)
        gl.set_value("inferserver", self)


    def pre_process(self, request):
        data = request.get_data()
        return data

    def post_process(self, data):
        return data

    def pridect(self, data):
        data1 = self.model1(data)
        data2 = self.model2(data)
        return [data1, data2]

    def run(self, ip="127.0.0.1", port=8080, debuge=False):
        app.run(ip, port, debuge)


if __name__ == '__main__':
    myserver = inferServer("")
    myserver.run()
