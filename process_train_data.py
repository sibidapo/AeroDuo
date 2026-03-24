import json
import os

def main():
    # Use absolute or relative paths from the script execution directory
    # Assuming this script is run from aeroduo directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(base_dir, 'data', 'train_data.json')
    output_path = os.path.join(base_dir, 'data', 'train_data_sample.json')
    
    print(f"Loading data from {input_path}...")
    with open(input_path, 'r') as f:
        data = json.load(f)
        
    allowed_towns = ['Carla_Town01', 'Carla_Town04', 'Carla_Town06','Carla_Town07', 'Carla_Town10HD']
    
    new_data = []
    
    print("Processing items...")
    for item in data:
        img_path = item.get('image_path', '')
        
        # Check if the path contains any of the allowed towns
        matched_town = None
        for town in allowed_towns:
            if town in img_path:
                matched_town = town
                break
                
        if matched_town:
            # Replace the part before the town name with '../data/Hal-13k/'
            # Using str.split(..., 1) ensures we only split at the first occurrence
            if 'image_path' in item:
                parts = item['image_path'].split(matched_town, 1)
                item['image_path'] = f"data/Hal-13k/{matched_town}{parts[1]}"
                
            if 'traj_folder_path' in item:
                parts = item['traj_folder_path'].split(matched_town, 1)
                item['traj_folder_path'] = f"data/Hal-13k/{matched_town}{parts[1]}"

            # Add sanity check to avoid empty JSON issues
            traj_dir = item.get('traj_folder_path', '')
            # Need to get path relative to current WD for existence checks
            # Since traj_dir starts with "data/...", it should be correct if run from project root
            desc_file = os.path.join(traj_dir, "object_description_with_help.json")
            waypoints_file = os.path.join(traj_dir, "gt_waypoints.json")
            
            # Check if critical files exist and are not empty
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
                new_data.append(item)
            
    print(f"Filtered down to {len(new_data)} items from {len(data)} items originally.")
    
    print(f"Saving to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(new_data, f, indent=4)
        
    print("Done!")

if __name__ == "__main__":
    main()
