# Data Folder Overview

This folder contains three key datasets: `behavio_signals`, `EEG_data`, and `questionnaire`. Below is a detailed overview of each dataset.

## Behavio_Signals Dataset Overview

The **"behavio_signals"** dataset includes data on likes and viewing duration for each video by participants during both pre-study and post-study phases.

### Privacy Considerations
To protect the privacy of the subjects, we use `"member_id"` to represent users.

### Data Columns

- **Like Column**: 
  - A checkmark (✓) indicates a like.
  - A blank space signifies no like was given.

- **Viewing Time Column**:
  - The format for representing time is `hours:minutes:seconds`.

### Video Title Information

The "video title" contains information about the type of video:
- `"LAB2"` represents post-study.
- `"LAB1"` indicates pre-study.
- The terms `"neutral," "fear," "happy,"` and `"sad"` correspond to the four types of videos from the SEED-IV dataset.
- The names of personages are associated with the type of personage videos.
- The terms `"positive"` and `"negative"` refer to the polarity of the video, which denotes the attitude towards the personage.

## Questionnaire Data Overview

The questionnaire data comprises responses from participants during the pre-study and post-study phases, presented in the form of a Likert-scale.

### Structure of the Questionnaire

- Questions about participants' sentiments and familiarity with **10 different personages**, totaling **20 questions**.
- **Distractor questions**, unrelated to the personages.

### Consistency and Aim

- Consistent questions in pre-study and post-study.
- Reflect changes in sentiments towards the personages after the field study.

### Eliminating Biases

- **Three different questionnaire setups - A, B, and C**.
- Randomized order of questions to eliminate biases.

## EEG Data Overview

The EEG dataset includes data of **23 participants** from both pre-study and post-study phases.

### Data Size and Accessibility

- Total size: nearly **40GB**.
- Provided EEG data of **one participant as an example**.

### Future Access to Full Dataset

- Post-review, a link to the cloud drive with the full dataset will be added to **GitHub**.
