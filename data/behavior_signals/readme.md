# Behavio_Signals Dataset Overview

The **"behavio_signals"** dataset includes data on likes and viewing duration for each video by participants during both pre-study and post-study phases.

## Privacy Considerations
To protect the privacy of the subjects, we use `"member_id"` to represent users.

## Data Columns

- **Like Column**: 
  - A checkmark (✓) indicates a like.
  - A blank space signifies no like was given.

- **Viewing Time Column**:
  - The format for representing time is `hours:minutes:seconds`.

## Video Title Information

The "video title" contains information about the type of video:
- `"LAB2"` represents post-study.
- `"LAB1"` indicates pre-study.
- The terms `"neutral," "fear," "happy,"` and `"sad"` correspond to the four types of videos from the SEED-IV dataset.
- The names of personages are associated with the type of personage videos.
- The terms `"positive"` and `"negative"` refer to the polarity of the video, which denotes the attitude towards the personage.
