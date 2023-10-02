from flask import Flask, request, jsonify, render_template

import gess_the_word as game
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('formulario.html')

@app.route('/procesar', methods=['POST'])
def procesar():
    #game.init_values()
    print("Mensaje recibido:", request.form['mensaje'])
    game.init_values()
    mensaje = game.play_one(request.form['mensaje'])
    
    print("Mensaje ganador:", mensaje)

    return render_template('formulario.html', mensaje=mensaje)

if __name__ == '__main__':
  app.run(debug=True)