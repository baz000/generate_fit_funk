from flask import Flask, request, jsonify
import clip  # Import the module containing the CLIP functions

app = Flask(__name__)

@app.route('/generate', methods=['POST'])
def generate_outfit_route():
    data = request.json

    shirts = data.get('shirtUrls', [])
    pants = data.get('pantUrls', [])
    shoes = data.get('shoeUrls', [])
    description = data.get('prompt', '')

    # Generate outfit using the provided function
    outfit = clip.generate_outfit(shirts, pants, shoes, description)

    return jsonify(outfit)

if __name__ == '__main__':
    app.run(port=3000, debug=True)





