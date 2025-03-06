import pyedflib
import pandas as pd
import numpy as np
import mne

all_data = []
study_prefix = "S"
file_prefix = "S00"
recordings = ["06", "10", "14"]
target_electrodes = ["Fpz", "F3", "F4", "C3", "C4", "P3", "Pz", "P4", "O1", "O2"]

# Iterate through all study files
for study in range(1, 109):  # Assuming the study number can go up to 999
    study_str = f"{study:03}"
    for recording in recordings:
        edf_file_path = f'{study_prefix}{study_str}/{study_prefix}{study_str}R{recording}.edf'
        try:
            print(f"Processing file: {edf_file_path}")
            edf = mne.io.read_raw_edf(edf_file_path, preload=True)
            header = ','.join(edf.ch_names)
            data = edf.get_data().T

            f = pyedflib.EdfReader(edf_file_path)
            annotations = f.readAnnotations()

            start_times = annotations[0]
            durations = annotations[1]
            descriptions = annotations[2]

            annotations_df = pd.DataFrame({
                'Start': start_times,
                'Duration': durations,
                'Description': descriptions
            })

            # Create dataframe for EEG data
            data_df = pd.DataFrame(data, columns=edf.ch_names)
            data_df.columns = data_df.columns.str.replace('.', '', regex=False)
            data_df.columns = data_df.columns.str.replace('# ', '', regex=False)
            target_data = data_df[target_electrodes]

            # Add annotations to the target data
            target_data['Annotations'] = ''
            for i in range(len(descriptions)):
                start_index = int(round(start_times[i] * edf.info['sfreq']))
                end_index = int(round(start_times[i] * edf.info['sfreq']) + round(durations[i] * edf.info['sfreq']))
                target_data.iloc[start_index:end_index, target_data.columns.get_loc('Annotations')] = descriptions[i].replace('T', '')

            # Append data to the list if not empty
            if not target_data.empty:
                all_data.append(target_data)

            f.close()
        except (FileNotFoundError, OSError) as e:
            print(f"Skipping file: {edf_file_path}, Error: {e}")
            continue

# Concatenate all dataframes and save to a CSV file
if all_data:
    final_data = pd.concat(all_data, ignore_index=True)
    final_data.to_csv('all_target_data.csv', index=False)
    print("All data saved to all_target_data.csv")
else:
    print("No valid data found to concatenate.")