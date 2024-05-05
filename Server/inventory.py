import json

INVENTORY_DB = 'Server/inventory.json'

def update_inventory(item_name, inserted, filename=INVENTORY_DB):
    item_dict = {}
    item_dict['name'] = item_name
    if inserted:
        item_dict['count'] = 1
    else:
        item_dict['count'] = -1
    
    update_db(item_dict, filename)

def update_db(item_dict, filename):
    """
    Update inventory JSON file with current item.

    Parameters:
        item_dict (list): Dictionary containing item name and count.
        filename (str): Name of the JSON file to write to.
    """
    # Load existing inventory if file exists
    try:
        with open(filename, 'r') as file:
            existing_inventory = json.load(file)
            print("Read in existing inventory. Adding new items.")
    except FileNotFoundError:
        existing_inventory = {}
        print("No existing inventory file. Creating new inventory json.")

    # Update existing inventory with new item and count
    name = item_dict['name']
    count = item_dict['count']
    if name in existing_inventory:
        existing_inventory[name] += count
        if existing_inventory[name] == 0: del existing_inventory[name] # remove from inventory db if count is 0
    else:
        if count != -1:
            existing_inventory[name] = count

    # Write inventory to JSON file
    with open(filename, 'w') as file:
        json.dump(existing_inventory, file, indent=4)

    print("Done updating inventory json db.")
    
    

def get_inventory(filename=INVENTORY_DB):
    """
    Read inventory from a JSON file into a list containing item names.

    Parameters:
        filename (str): Name of the JSON file to read from.

    Returns:
        inventory (list): List of dictionaries containing item names and counts.
    """
    try:
        with open(filename, 'r') as file:
            inventory = json.load(file)
            print("Found inventory file. Processing...")
    except FileNotFoundError:
        inventory = {}
        print("No inventory file. Creating empty inventory")

    # Convert inventory dictionary to list of dictionaries
    item_counts = [{'name': name, 'count': count} for name, count in inventory.items()]
    print(item_counts)

    # Extract item names from the inventory dictionary
    items = list(inventory.keys())

    return items


# Testing
# update_inventory("apple")
# get_inventory()
# update_inventory("banana")
# get_inventory()
# update_inventory("apple")
# get_inventory()
# update_inventory("apple", False)
# get_inventory()
# update_inventory("apple", False)
# get_inventory()
