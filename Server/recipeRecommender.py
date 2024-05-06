import pandas as pd
import numpy as np

RECIPE_CSV_PATH = 'Datasets/RAW_recipes.csv'
df_recipes = pd.read_csv(RECIPE_CSV_PATH)

'''
Retrieve top k recipes with the most ingredient matches.
This function takes three parameters:
df_recipes: DataFrame containing recipe data, including columns 'id' and 'ingredients'.
ingredients_list: List of ingredients provided by the user.
topk (optional): Number of top recommended recipes to return. Default is 5.
'''
def recommend_by_highest_ingredient_match(df_recipes, ingredients_list, topk=5):
  # Store recipe id: num matching ingredients
  recipe_match_counts = {}

  # Iterate through each recipe in the DF
  for index, recipe_row in df_recipes.iterrows():
      recipe_id = recipe_row['id']
      recipe_ingredients = recipe_row['ingredients']
      # Count the number of matching ingredients between user's list and recipe's ingredients
      ingredient_matches = sum(1 for ing in ingredients_list if ing in recipe_ingredients)
      recipe_match_counts[recipe_id] = ingredient_matches

  # Sort the recipes based on the number of matchingingredients
  sorted_matches = sorted(recipe_match_counts.items(), key=lambda x: x[1], reverse=True)
  topk_recipes = sorted_matches[:topk]
  # Filter the DF to include only the recommended recipes
  recommended_recipes = df_recipes[df_recipes['id'].isin([recipe[0] for recipe in topk_recipes])]

  return recommended_recipes[['name','tags', 'ingredients']]

# Test ingredient matching recs
# INVENTORY = ['chicken', 'spinach']
# print(recommend_by_highest_ingredient_match(df_recipes, INVENTORY, topk=5))