import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# Paths to your CSV files
file_paths = [
    "/Users/emmanuelephraim/Desktop/Premier League 23:24 Data/player_accurate_long_balls.csv",
    "/Users/emmanuelephraim/Desktop/Premier League 23:24 Data/player_accurate_passes.csv",
    "/Users/emmanuelephraim/Desktop/Premier League 23:24 Data/player_big_chances_created.csv",
    "/Users/emmanuelephraim/Desktop/Premier League 23:24 Data/player_big_chances_missed.csv",
    "/Users/emmanuelephraim/Desktop/Premier League 23:24 Data/player_expected_assists.csv",
    "/Users/emmanuelephraim/Desktop/Premier League 23:24 Data/player_fouls_committed.csv",
    "/Users/emmanuelephraim/Desktop/Premier League 23:24 Data/player_interceptions.csv",
    "/Users/emmanuelephraim/Desktop/Premier League 23:24 Data/player_tackles_won.csv",
    "/Users/emmanuelephraim/Desktop/Premier League 23:24 Data/player_top_assists.csv",
    "/Users/emmanuelephraim/Desktop/Premier League 23:24 Data/player_top_scorers.csv",
    "/Users/emmanuelephraim/Desktop/Premier League 23:24 Data/player_contests_won.csv",
    "/Users/emmanuelephraim/Desktop/Premier League 23:24 Data/player_on_target_scoring_attempts.csv",
    "/Users/emmanuelephraim/Desktop/Premier League 23:24 Data/player_expected_goals.csv"
]

# Load the first file to initialize the DataFrame
data = pd.read_csv(file_paths[0])

# Loop through the remaining files and merge them, specifying suffixes to avoid column conflicts
for file_path in file_paths[1:]:
    new_data = pd.read_csv(file_path)
    data = pd.merge(data, new_data, on="Player", how="outer", suffixes=('', '_duplicate'))

# Remove any duplicate columns created during merging
# Keep the original columns and drop duplicates
data = data.loc[:, ~data.columns.str.endswith('_duplicate')]


# List of features to include in the similarity calculation
features = [
    'Accurate Long Balls per 90', 'Accurate Passes per 90', 'Big Chances Created', 
    'Big Chances Missed', 'Expected Assists (xA)', 'Dribble Success Rate (%)', 
    'Fouls Committed per 90', 'Yellow Cards','Total Interceptions', 'Tackle Success Rate (%)',
    'Assists','Secondary Assists', 'Goals', 'Penalties', 'Expected Goals (xG)', 'Shots on Target per 90'
]

# Handle any missing values (fill NaNs with 0 for simplicity)
data[features] = data[features].fillna(0)

# Standardize the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(data[features])

# Calculate cosine similarity
cosine_sim = cosine_similarity(features_scaled)
similarity_df = pd.DataFrame(cosine_sim, index=data['Player'], columns=data['Player'])

# Define the query player
query_player1 = 'Erling Haaland'
query_player2 = 'Declan Rice'
query_player3 = 'Mohamed Salah'

# Get top 10 similar players excluding the player itself
if query_player1 in similarity_df.index:
    top_10_similar = similarity_df[query_player1].nlargest(11).iloc[1:]  # Exclude itself
    print("Top 10 similar players to", query_player1)
    print(top_10_similar, "\n")
else:
    print(f"Player {query_player1} not found in the dataset.")
    
# Get top 10 similar players excluding the player itself    
if query_player2 in similarity_df.index:
    top_10_similar = similarity_df[query_player2].nlargest(11).iloc[1:]  # Exclude itself
    print("Top 10 similar players to", query_player2)
    print(top_10_similar, "\n")
else:
    print(f"Player {query_player2} not found in the dataset.")



# Get top 10 similar players excluding the player itself    
if query_player3 in similarity_df.index:
    top_10_similar = similarity_df[query_player3].nlargest(11).iloc[1:]  # Exclude itself
    print("Top 10 similar players to", query_player3)
    print(top_10_similar, "\n")
else:
    print(f"Player {query_player3} not found in the dataset.")