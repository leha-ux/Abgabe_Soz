import pandas as pd

# Load both DataFrames
aggregated_df = pd.read_csv("aggregated.csv")
processed_df = pd.read_csv("processed_dataframe.csv")

# Convert the ID-String into a list of IDs and remove potential whitespaces
aggregated_df['id'] = aggregated_df['id'].str.split(',').apply(lambda x: [item.strip() for item in x])

# Create a mapping of each ID to a name
id_to_name_mapping = {}
for _, row in aggregated_df.iterrows():
    for single_id in row['id']:
        id_to_name_mapping[single_id] = row['Name']

# Ensure that both IDs (in the dictionary and in processed_df) are of the same type
processed_df['id'] = processed_df['id'].astype(str)

# Add the name to processed_df based on the ID
processed_df['topic'] = processed_df['id'].map(id_to_name_mapping)

# Count rows with NaN in 'topic' column
nan_count = processed_df['topic'].isna().sum()
print(f"Number of rows with NaN in 'topic' column: {nan_count}")

# Print the IDs that couldn't be mapped
missing_ids = processed_df[processed_df['topic'].isna()]['id']
print("IDs that couldn't be mapped:")
print(missing_ids)

# Fill NaN values in the 'topic' column with 'unknown'
processed_df['topic'].fillna('unknown', inplace=True)

# Save the processed DataFrame to a CSV file
processed_df.to_csv("processed_dataframe_updated.csv", index=False)