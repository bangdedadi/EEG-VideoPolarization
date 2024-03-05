# Code Documentation

We have provided the code in two formats, **Python** and **Jupyter Notebook**, for demonstration purposes.

## Preprocessing Code

- **Purpose**: Aligns EEG data with the absolute time collected from the web (timestamps) and extracts EEG data corresponding to each event (video playback time).

## DE_Feature Code

- **Purpose**: Extracts Differential Entropy (DE) features.
- **Frequency Bands**: The extraction is performed across five frequency bands:
  - `delta`
  - `theta`
  - `alpha`
  - `beta`
  - `gamma`

## Draw_Topomap Code

- **Purpose**: Calculates the correlation between exposure to Opinion Polarization (OP) and EEG signals.
- **Visualization**: Plots an electroencephalogram to visualize this correlation information.

## OP_Detection_Model Code

- **Purpose**: Uses EEG data, behavior signals, and questionnaire data to train a model.
- **Application**: Predicting exposure to opinion polarization .

