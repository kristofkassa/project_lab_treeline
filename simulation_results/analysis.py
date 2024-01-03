import pandas as pd
import matplotlib.pyplot as plt
import ast 

def load_data_from_excel(file_name):
    # Load data from each sheet into a separate DataFrame
    box_df = pd.read_excel(file_name, sheet_name='box_counting', index_col=0)
    corr_df = pd.read_excel(file_name, sheet_name='correlation', index_col=0)
    ruler_df = pd.read_excel(file_name, sheet_name='ruler', index_col=0)
    avgdist_df = pd.read_excel(file_name, sheet_name='avgdist', index_col=0)

    return box_df, corr_df, ruler_df, avgdist_df

file_name = 'simulation_results_20240103_174543.xlsx'  
box_df, corr_df, ruler_df, avgdist_df = load_data_from_excel(file_name)


def parse_and_average(df):
    averages = []
    for index, row in df.iterrows():
        # Parse each cell in the row as a list and calculate its average
        row_averages = []
        for cell in row:
            # Safely evaluate the string as a list
            if isinstance(cell, str):
                try:
                    values = ast.literal_eval(cell)
                    if isinstance(values, list):
                        row_averages.append(sum(values) / len(values))
                except:
                    pass  # Handle cells that cannot be parsed as lists
        if row_averages:
            averages.append(sum(row_averages) / len(row_averages))
    return averages

# Assuming box_df, corr_df, ruler_df, avgdist_df are already loaded DataFrames
box_averages = parse_and_average(box_df)
corr_averages = parse_and_average(corr_df)
ruler_averages = parse_and_average(ruler_df)
avgdist_averages = parse_and_average(avgdist_df)

# Plotting
plt.figure(figsize=(12, 8))

# Plot each set of averages
plt.plot(box_averages, label='Box Counting')
plt.plot(corr_averages, label='Correlation')
plt.plot(ruler_averages, label='Ruler')
plt.plot(avgdist_averages, label='Average Distance')

# Adding titles and labels
plt.title('Average Values per Line')
plt.xlabel('Line Number')
plt.ylabel('Average Value')
plt.legend()

# Show plot
plt.show(block=True)
