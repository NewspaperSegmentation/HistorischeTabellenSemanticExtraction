# HistorischeTabellenSemanticExtraction

using python 3.10



# Get started
The dataset currently can not be downloaded by a script. So it needs to copied in data folder by hand. It needs to have
the following structure, with _annotations_ containing all the Transkribus annotation -xml-files and _images_ containing all
the images as .jpg-files.

```
.
+-- data
|   +--BonnData
|   |  +--annoations
|   |  |  +--.xml-files
|   |  |        ...
|   |
|   |  +--images
|      |  +--.jpg-files
|      |        ...
```

### GloSAT dataset
The GloSAT dataset can be downloaded using the `download.py` script: 
```python
    python -m src.download
```


## Preprocess the data
For preprocessing the data the `preprocess.py` can be used. It creates a new folder in data/Tables 
with all preprocessed images and annotations. Creating a new folder for very image, c

```python
    python -m src.TableExtraction.preprocess
```

## Create Training, Validation and Test split
To create a split on the data the `split.py` can be used:

```python
    python -m src.TableExtraction.split
```

## Train a model

## Evaluated a model

