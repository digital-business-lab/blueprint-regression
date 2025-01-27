# Explanation of variables
The current config.yaml holds standard values that are set for a overall good and quick run. All the specified columns in numColumns, etc. must have the same name as they are defined in your dataset.
### Dataset
---
**name** : The name of your dataset

**target_column** : The column name you want to predict

**train_size** : The amount of data used for training (0.7 -> 70% of your dataset)

### Preprocessing
---
**dropColumns** : Columns in your dataset you want to delete / not use for model

**numColumns** : All the numerical columns in your dataset you want to preprocess. Columns that are numeric and not specified here will be skipped but also used for training.

**numColsProcessor** : Method how you want to fill missing values in your numerical columns. Available values are: mean, median, constant, most_frequent.

**numColsScale** : Boolean if you want to scale your data or not

**catColumns**: All the categorical columns in your dataset you want to preprocess. Columns that are categorical and not specified here will be skipped but used for training, which leads to an error when running the model. Add **ALL** your **CATEGORICAL** column names here.

**catColsProcessor** : Method how you want to fill missing values in your categorical columns. Available values are: most_frequent, constant.

### model
---
**modelName** : Name of the model you want to load. If None, a new one will be created and saved.

**modelMode** : Mode in which you want to execute the model. Available values are: train, evaluate, predict, train_predict

### modelParams
---
**input_size** : Input size for the model (Number of Features)

**hidden_size** : Size of hidden layers in your model

**output_size** : Output size of the model. Usually 1 for regression tasks.

**epochs** : Number of epochs the model should run

**batch_size** : Number of rows that will be loaded into the model per step

**lr** : Learning rate

**dropout_rate** : Number of Neurons that will randomly be deactivated on the hidden layers. Is usefull against overfitting. (0.3 -> 30% of Neurons in a hidden layer)

