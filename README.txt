________________________________________________________________________
NOTE

THIS PROJECT USES DATA FILES THAT ARE TOO LARGE TO BE CONTAINED IN GIT.
IF YOU WISH TO REPRODUCE THESE RESULTS, CONTACT ME AT mam084@ucsd.edu
________________________________________________________________________


README.txt
ASL Interpretation Project (WLASL Dataset)

Overview
This project builds and evaluates machine learning models for recognizing American Sign Language (ASL) signs using the WLASL video dataset. The notebook implements a full workflow including data loading, preprocessing, bounding-box baseline utilities, model training, and evaluation using PyTorch.

The primary file is:

Final_Report.ipynb

This notebook contains the entire pipeline for preparing the dataset and training the ASL recognition model.

------------------------------------------------------------
System Requirements
------------------------------------------------------------

Python 3.9+ recommended

Required Python packages:

torch
torchvision
numpy
pandas
matplotlib
opencv-python (recommended for video handling)
jupyter

You can install the dependencies using pip:

pip install torch torchvision numpy pandas matplotlib opencv-python jupyter

------------------------------------------------------------
Dataset Requirements
------------------------------------------------------------

This project uses the WLASL dataset.

Required files:

WLASL_v0.3.json
missing.txt
video directory containing all sign videos

Example folder structure:

project_folder/
│
├── Final_Report.ipynb
├── dataset/
│   ├── WLASL_v0.3.json
│   ├── missing.txt
│   └── videos/
│       ├── video1.mp4
│       ├── video2.mp4
│       └── ...
│
└── clean videos/
    └── wlasl-complete/
        ├── WLASL_v0.3.json
        ├── missing.txt
        └── videos/

The notebook searches for videos inside directories such as:

clean videos/wlasl-complete/videos

If your videos are stored elsewhere, update the VIDEO_DIR_CANDIDATES path in the notebook.

------------------------------------------------------------
Running the Project
------------------------------------------------------------

Step 1 — Open the notebook

Launch Jupyter:

jupyter notebook

Then open:

Final_Report.ipynb


Step 2 — Run the Setup Section

The first section imports required libraries and initializes the environment.

Run all cells in order.


Step 3 — Data Loading and Preprocessing

The notebook will:

1. Load WLASL_v0.3.json
2. Convert the JSON structure into a clean tabular dataframe
3. Expand video instances into individual rows
4. Remove videos listed in missing.txt
5. Validate that video files exist locally


Step 4 — Dataset Construction

The notebook defines PyTorch Dataset and DataLoader objects for:

• Loading video frames
• Applying transforms
• Feeding batches into the model


Step 5 — Model Training

The modeling section trains an ASL recognition model using PyTorch.

Typical steps include:

• Building the neural network architecture
• Creating training and validation loaders
• Running training epochs
• Monitoring loss and accuracy


Step 6 — Evaluation

After training, the notebook evaluates model performance and visualizes results using matplotlib.


------------------------------------------------------------
Notes
------------------------------------------------------------

1. GPU acceleration is recommended but not required.

If using CUDA:

torch.cuda.is_available()

should return True.


2. Large dataset warning

The WLASL dataset contains many video files and may require significant storage and memory.


3. Path configuration

If the notebook cannot find the dataset, update the path variables near the top of the notebook:

JSON_PATH
VIDEO_DIR_CANDIDATES


------------------------------------------------------------
Expected Output
------------------------------------------------------------

The notebook will produce:

• Processed dataset dataframe
• Training progress logs
• Evaluation metrics
• Visualizations of model performance


------------------------------------------------------------
Authors
------------------------------------------------------------

Team 13
Franky Liu
Jaron Zhu
Matthew Mitchell

University of California, San Diego
HDSI Masters of Data Science Program
