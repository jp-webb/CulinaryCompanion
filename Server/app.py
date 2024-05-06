import os
import json
from flask import Flask, render_template, jsonify
from watchdog.observers import Observer
import pandas as pd
from watchdog.events import FileSystemEventHandler
import recipeRecommender as rr

# Initialize Flask app
app = Flask(__name__)

# Path to JSON file containing inventory data
JSON_FILE = 'Server/inventory.json'
food_data = {}

def get_recipe():
    df_recipes = pd.read_csv("Datasets/RAW_recipes.csv")
    return rr.recommend_by_highest_ingredient_match(df_recipes, list(food_data), topk=5)

def load_json():
    global food_data, df
    with open(JSON_FILE, 'r') as file:
        food_data = json.load(file)
    df = get_recipe()
    

load_json()

# Class to handle changes to the JSON file
class JSONFileHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if event.src_path.endswith('.json'):
            print("JSON file changed. Reloading data...")
            load_json()

observer = Observer()
observer.schedule(JSONFileHandler(), path='.')
observer.start()

# Route for the index page
@app.route('/')
def index():
    return render_template('index.html')

# Route to get current food inventory
@app.route('/foods')
def get_foods():
    return jsonify(food_data)

# Route to execute and return recipe recommendations
@app.route('/execute', methods=['POST'])
def execute():
    global df
    df_json = df.to_json(orient='records')
    return jsonify(df_json)

# Main entry point
if __name__ == '__main__':
    print("app.run")
    app.run(debug=True)
