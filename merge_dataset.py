import pandas as pd

base_dir = '/home/a/model_analysis/2024_competition_dataset/dataset_merge/'
# Load the datasets
avante_cn7_info_path_new = base_dir + 'avante_cn7_info.csv'
wheel_speed_data_path = base_dir + 'cmd_accel.csv'

# Read the CSV files
avante_cn7_info_new = pd.read_csv(avante_cn7_info_path_new)
wheel_speed_data = pd.read_csv(wheel_speed_data_path)

# Convert time columns to seconds (assuming time is in nanoseconds)
avante_cn7_info_new['time_sec'] = avante_cn7_info_new['time'] / 1e9
wheel_speed_data['time_sec'] = wheel_speed_data['time'] / 1e9

# Merge the datasets based on the closest time value
merged_data = pd.merge_asof(
    avante_cn7_info_new.sort_values('time_sec'),
    wheel_speed_data.sort_values('time_sec'),
    on='time_sec',
    direction='nearest'  # Merge with the nearest timestamp
)

# Save the merged data to a new CSV file
merged_data.to_csv(base_dir + 'merged_data.csv', index=False)

# To preview the first few rows of the merged data
print(merged_data.head())

# Select the columns which are needed for the model
selected_columns = ['time_sec', 'field.linear.x', 'field.wheel_speed.front_left', 'field.wheel_speed.front_right', 'field.wheel_speed.rear_left', 'field.wheel_speed.rear_right']
time = merged_data['time_sec'].values
#offset the time to start from 0
time = time - time[0]
merged_data['time_sec'] = time

# Get average wheel speed
merged_data['average_wheel_speed'] = merged_data[['field.wheel_speed.front_left', 'field.wheel_speed.front_right', 'field.wheel_speed.rear_left', 'field.wheel_speed.rear_right']].mean(axis=1)

# change the column names 'time_sec' to 'time' and 'field.linear.x' to 'accleration' and 'average_wheel_speed' to 'velocity_x'
merged_data = merged_data.rename(columns={'time_sec': 'time', 'average_wheel_speed': 'velocity_x', 'field.linear.x': 'acceleration'})

selected_columns = ['time', 'velocity_x', 'acceleration']

# Save the selected columns to a new CSV file
merged_data[selected_columns].to_csv(base_dir + 'selected_data.csv', index=False)



