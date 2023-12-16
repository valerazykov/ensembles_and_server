from ml_server import app

#TODO: waitress

if __name__ == '__main__':
    app.run(host='::', port=5000, debug=True)
