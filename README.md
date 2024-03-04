## CUB Finetuning using EfficientNet

### To download the dataset and prepare it for the task please execute the below commands
```
chmod +x ./dataset_download.sh
./dataset_download.sh
```

### To make a environment for the same, execute the following commands
```
conda env create -f environment.yml
conda activate doshi
```

### To start training
```
python train.py --data_dir CUB_200_2011/images/ --output_dir [output_dir] --batch_size [batch_size]  --num_epochs [num_epochs]
```

### To test the model
```
python test.py --model_weights [model_weights] --data_dir CUB_200_2011/images/
```
