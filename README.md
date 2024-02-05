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
    python -m src.TableExtraction.download
```


## Preprocess the data
For preprocessing the data the `preprocess.py` can be used. It creates a new folder in data/BonnData
or data/GloSAT called preprocessed with all the preprocessed data. Use `--BonnData` or `--GloSAT` to
preprocess the specific dataset.

```python
    python -m src.TableExtraction.preprocess --BonnData
```
```python
    python -m src.TableExtraction.preprocess --GloSAT
```

## Create Training, Validation and Test split
To create a split on the data the `split.py` can be used:

```python
    python -m src.TableExtraction.split --BonnData
```
```python
    python -m src.TableExtraction.split --GloSAT
```

## Train a model

## Evaluated a model

