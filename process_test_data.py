import json
import os

def main():
    # Assuming this script is run from aeroduo directory, or we can use absolute path logic like in train script
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(base_dir, 'data', 'test_unseen_new.json')
    output_path = os.path.join(base_dir, 'data', 'test_data_unsee.json')
    
    print(f"Loading data from {input_path}...")
    with open(input_path, 'r') as f:
        data = json.load(f)
        
    # User stated that only this town data is unzipped in aeroduo/data/Hal-13k_test/
    allowed_towns = ['Carla_Town05']
    
    new_data = []
    
    print("Processing test items...")
    # The test JSON is a list of folder paths (strings)
    for item_path in data:
        # Check if the folder path contains Carla_Town05
        matched_town = None
        for town in allowed_towns:
            if town in item_path:
                matched_town = town
                break
        
        if matched_town:
            # Replace 'data/HaL-13k/' with 'data/Hal-13k_test/'
            # The original paths are like "data/HaL-13k/Carla_Town05/..."
            if "data/HaL-13k/" in item_path:
                new_path = item_path.replace("data/HaL-13k/", "data/Hal-13k_test/")
            elif "data/Hal-13k/" in item_path:
                 new_path = item_path.replace("data/Hal-13k/", "data/Hal-13k_test/")
            else:
                # Handle cases where it might just be the town name
                new_path = item_path # Fallback

            # Check if critical files exist in this new location
            # (Similar to process_train_data.py sanity checks)
            desc_file = os.path.join(base_dir, new_path, "object_description_with_help.json")
            waypoints_file = os.path.join(base_dir, new_path, "gt_waypoints.json")
            
            is_valid = True
            for file_path in [desc_file, waypoints_file]:
                if not os.path.exists(file_path):
                    # print(f"Warning: Missing metadata {file_path}")
                    is_valid = False
                    break
                if os.path.getsize(file_path) == 0:
                    # print(f"Warning: Empty metadata {file_path}")
                    is_valid = False
                    break
            
            if is_valid:
                new_data.append(new_path)
            
    print(f"Filtered down to {len(new_data)} items from {len(data)} items originally.")
    
    print(f"Saving to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(new_data, f, indent=4)
        
    print("Done!")

if __name__ == "__main__":
    main()
