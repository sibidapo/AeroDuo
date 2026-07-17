## Preprocessing Steps After Downloading and Extracting Hal-13K


1. Scenes downloaded are: ```(Carla_Town01, Carla_Town02, Carla_Town03, ModularPark) -> scratch, (Carla_Town04, Carla_Town05, Carla_Town06, Carla_Town07, Carla_Town10HD, ModernCityMap, NewYorkCity, NYCEnvironmentMegapa, TropicalIsland) -> r-cj124-0, (Carla_Town15) -> r-lgan31```

2. Total number of supposed trajectories (number of trajectory folders) = 13,640. Total number of trajectories (probably) used in training Aeroduo  (from the original ``train_data.json``) = 6,225

3. First preprocessing step: Pre-populate ```train_data.json``` to include missing scenes, exclude the scenes in ```test_unseen_new.json```, filtering off trajectories with ```_supp``` entries and filtering off trajectories without gt_waypoints. Result is a ```train_data_new.json``` with a total of 9082 trajectories. 
    Command used for this is:
    ```bash
    python generate_train_data.py   --output data/train_data_new.json   --report data/train_data_new.audit.json
    ```

4. Note that in both ```train_data.json``` and ```train_data_new.json```, the test maps ```Carla_Town05, ModularPark and NewYorkCity``` are excluded from the training dataset.

5. Detailed data analysis in ```aeroduo/pilot_llm/data_analysis.ipynb```

6. Upon analysis, a significant amount of trajectories with discrepancies in the number of front and bev cameras. A conservative discrepancy filter threshold of 70 is chosen to filter off episodes where frontcamera exceeds bevcamera (positive discrepancy) by 70. Negative discrepancy is not filtered off because it the training objective is focused on learning high_uav representation. The dataeset class: ```aeroduo/pilot_llm/high_uav/dataset.py``` handles this negative discrepancy.

7. Updated ```train_data_new.json``` by running:
    ```bash
    python filter_discrepant_episodes.py --max-discrepancy 70
    ```
    Total number of trajectory = 8901