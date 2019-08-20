<h1>Semantic data augmentation</h1>

* Based on numpy, OpenCV,  picking the best from each of them.
* Simple, flexible API that allows the library to be used in any computer vision pipeline.
* Easy to extend the library to wrap around other libraries.
* Easy to extend to other tasks.
* Supports python 2.7-3.7
* Easy integration with PyTorch.
* Supports extraction of people on segmented images.

## Table of contents
- [How to use](#how-to-use)
- [Installation](#installation)
  - [People extract](#extract-people)
  - [Overlay road with people](#road-overlay)

<a name="features"/>

## How to use

- clone the repository
```bash
git clone https://github.com/nemodrive/semantic-data-augmentation.git
```
- download Cityscapes Dataset from [Cityscape Dataset](https://www.cityscapes-dataset.com).
- create a two columns CSV file with original image path and coresponding segmented image path (one example is in resources/good_train_fine.txt)
- create your own dataset (similar to Cityscape Dataset), having the original image and segmented road of that
- have fun :)

<a name="#installation"/>

## Installation

1. Install pip
```bash
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python get-pip.py
```
<a name="#extract-people" />

## Extract people

2. Clone the repository

```bash
git clone https://github.com/nemodrive/semantic-data-augmentation.git
```

3. Go to extract_people.py

```bash
cd scripts
```

4. Run extract_people.py

```bash
python extract_people.py <path_to_CSV_file>
```

<a name="#road-overlay" />

## Overlay road with people

5. You can install roadpackage using pip command

```bash
pip install git+https://github.com/nemodrive/semantic-data-augmentation.git
```

6. Import in your file 

```bash 
from roadpackage.road import overlay_people_on_road
```

