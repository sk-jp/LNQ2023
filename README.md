# LNQ2023

## Training
1. Go to 'train' directory.
2. Make train and validation datalists.
   Samples are under 'datalist' directory.
3. Edit a yaml file.
   The sample yaml file is 'flexunet_b7_2.5d_random_noresize.yaml'.
   You must change the following items.
     Data.dataset.top_dir
     Data.dataset.train_datalist
     Data.dataset.valid_datalist
     Data.dataset.cache_dir
4. Run the main script.
     $ python main.py --config <yaml file>
   The result files are created under 'results' directory.

## Prediction
1. Go to 'predict' directory.
2. Make predict datalist.
   A sample is under 'datalist' directory.
3. Edit a yaml file.
   The sample yaml file is 'flexunet_b7_2.5d_random_predict_noresize.yaml'.
   You must change the following items.
     Data.dataset.top_dir
     Data.dataset.predict_datalist
     Data.dataset.cache_dir
     Model.pretrained
5. Run the main script.
     $ python predict.py --config <yaml file>
   The result files are created under 'results' directory.
