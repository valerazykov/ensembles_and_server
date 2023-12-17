from waitress import serve
from ml_server import app


if __name__ == '__main__':
    serve(app, host='::', port='5000', threads=2)
