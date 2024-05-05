import pandas as pd
import numpy as np

df_recipes = pd.read_csv('RAW_recipes.csv')

# Define Feature Engineering Function
def fe_tags(food_tags):
    values = []

    for tag in INGREDIENT_TAGS:
        values.append(True) if tag in food_tags else values.append(False)

    return values

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

  for index, recipe_row in df_recipes.iterrows():
      recipe_id = recipe_row['id']
      recipe_ingredients = recipe_row['ingredients']

      ingredient_matches = sum(1 for ing in ingredients_list if ing in recipe_ingredients)
      recipe_match_counts[recipe_id] = ingredient_matches

  sorted_matches = sorted(recipe_match_counts.items(), key=lambda x: x[1], reverse=True)
  topk_recipes = sorted_matches[:topk]
  recommended_recipes = df_recipes[df_recipes['id'].isin([recipe[0] for recipe in topk_recipes])]

  return recommended_recipes[['name','tags', 'ingredients']]

# Test ingredient matching recs
# INVENTORY = ['chicken', 'spinach']
# print(recommend_by_highest_ingredient_match(df_recipes, INVENTORY, topk=5))