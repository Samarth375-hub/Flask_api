from flask import Flask, request,render_template,jsonify
import pickle
import numpy as np
import warnings
warnings.filterwarnings('ignore',category=UserWarning,module='sklearn')

with open('iris_model.pkl','rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

# @app.route('/',methods = ['GET','POST'])
# def home():
#     prediction = None

#     if request.method == 'POST':
#         # Get the input data from the form
#         sepal_length = float(request.form['sepal_length'])
#         sepal_width = float(request.form['sepal_width'])
#         petal_length = float(request.form['petal_length'])
#         petal_width = float(request.form['petal_width'])

#         # Prepare the data for prediction
#         features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

#         # Make the prediction
#         prediction = model.predict(features)[0]  # get the first prediction result

#         # Map prediction to actual species (Optional)
#         species = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}
#         prediction = species.get(prediction, 'Unknown')

#     return render_template('index.html', prediction=prediction)
    

@app.route('/predict',methods=['POST'])
def predict():
    try:
        data = request.json
        features = data['features']
        
        features = np.array(features).reshape(1,-1)
        
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features).tolist()[0]
        
        class_names = ['setosa','versicolor','virginica']
        predicted_class = class_names[prediction]
        
        return jsonify({
            'prediction': predicted_class,
            'probabilities':dict(zip(class_names,probabilities))
        })
    
    except Exception as e:
        return jsonify({'error':str(e)})

if __name__ == '__main__':
    app.run(debug = True)
