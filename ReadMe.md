# NeuraTrace
View the internals of a network, compute forward pass activations and visualise relationship between layers.

# Setup
```
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
python server.py
```

Access site on `http://localhost:8765`

# Features

## Dynamic inputs & data ingestion
Model loaded on the RHS panel, you can also change inputs, either uploading .npy files / images or inputing data directly for small input layers, example RT-DETR pass which uses an input image + a [N,2] input for image resolution.

---
<img width="1859" height="916" alt="image" src="https://github.com/user-attachments/assets/75af10a0-41ec-4d7f-ab99-9329e18e173e" />

## Interactive layers
3D view is fully interactive and allows you select individual layers, previewing connections and best effort visualisation.

---
<img width="1347" height="884" alt="image" src="https://github.com/user-attachments/assets/5198dff3-bb28-4a05-8faf-53f411c26c30" />

## Drill down visualisations
Drill down is also supported for complex channel layers, also includes PCA filtering and scalar activation output so you can see which channels are contributing most
<\br>
All visualisations exportable as .png

---
<img width="1033" height="903" alt="image" src="https://github.com/user-attachments/assets/2a353ed7-54cc-4c6c-b793-5e8536e8c795" />

## Layer Filtering
Filtering is available on the LHS where you can hide neurons that aren't firing, you can also filter by discovered layers (naming convention)\

---
<img width="1389" height="912" alt="image" src="https://github.com/user-attachments/assets/03cb4efd-a605-4ec5-ae64-be019ca6e041" />

## Playthrough visualisations
Playthrough is also supported for previewing activations through the layers

---
<img width="680" height="383" alt="output" src="https://github.com/user-attachments/assets/9fa7c64a-2ac9-4bb3-af5e-0b82eeeb299d" />


# Usage help
Most items on the UI have tooltips, just hover mouse over for an explanation on what they do.
