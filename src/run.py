from ml_server import app
from waitress import serve

if __name__ == '__main__':
    serve(app, host='::', port='5000', threads=2)
