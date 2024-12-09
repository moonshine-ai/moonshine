from flask import Flask, send_from_directory
import os

app = Flask(__name__)

FILES_DIRECTORY = "." 


@app.route('/<path:filename>')
def serve_file(filename):
    """Serve a file from the specified directory."""
    if not os.path.exists(os.path.join(FILES_DIRECTORY, filename)):
        return "File not found", 404
    return send_from_directory(FILES_DIRECTORY, filename)

@app.route('/')
def serve_index():
    return send_from_directory(FILES_DIRECTORY, "index.html")

if __name__ == '__main__':
    app.run(host="localhost", port="5001")
