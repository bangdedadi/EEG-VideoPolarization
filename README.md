# Project Overview

This project encompasses a comprehensive dataset and associated code, aiming to analyze and predict exposure to opinion polarization using EEG data, behavior signals, and questionnaire responses.

## Data Folder Overview

This folder contains three key datasets: `behavio_signals`, `EEG_data`, and `questionnaire`. 

### Behavio_Signals Dataset Overview

- **Content**: Data on likes and viewing duration for each video by participants during both pre-study and post-study phases.
- **Privacy**: Uses `"member_id"` to represent users for privacy protection.
- **Data Columns**:
  - **Like Column**: A checkmark (✓) for a like, and a blank space for no like.
  - **Viewing Time Column**: Time format is `hours:minutes:seconds`.
- **Video Title Information**: Includes `"LAB2"` for post-study, `"LAB1"` for pre-study, and terms like `"neutral," "fear," "happy,"` and `"sad"` for video types from SEED-IV dataset.

### Questionnaire Data Overview

- **Content**: Responses from participants in a Likert scale format.
- **Structure**:
  - **20 Questions**: On sentiments and familiarity with 10 different personages.
  - **Distractor Questions**: Unrelated to the personages.
- **Consistency and Aim**: Reflects changes in sentiments towards personages after the field study.
- **Bias Elimination**: Three different randomized questionnaire setups (A, B, and C).

### EEG Data Overview

- **Content**: EEG data of 23 participants from both pre-study and post-study phases.
- **Data Size and Accessibility**: Total size nearly 40GB; one participant's data provided as an example.
- **Future Access**: Post-review, a cloud drive link with the full dataset will be added to GitHub.

## Code Documentation

The code is provided in Python and Jupyter Notebook formats.

### Preprocessing Code

- **Purpose**: Aligns EEG data with web-collected timestamps and extracts data for events.

### DE_Feature Code

- **Purpose**: Extracts Differential Entropy (DE) features.
- **Frequency Bands**: Includes `delta`, `theta`, `alpha`, `beta`, and `gamma`.

### Draw_Topomap Code

- **Purpose**: Calculates correlation between exposure to Opinion Polarization (OP) and EEG signals.
- **Visualization**: Plots an electroencephalogram to visualize correlation.

### OP_Detection_Model Code

- **Purpose**: Uses EEG data, behavior signals, and questionnaire data for training a model.
- **Application**: Predicting exposure to opinion polarization.
