from ml_server import app

#TODO: waitress, host

if __name__ == '__main__':
    app.run(host='192.168.1.175', port=5000, debug=True)
