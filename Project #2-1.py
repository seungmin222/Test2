import pandas as pd

data_df = pd.read_csv('2019_kbo_for_kaggle_v2.csv')

# Q1 
def top_10_players():
    for column_name in ['H', 'avg', 'HR', 'OBP']:
        print(f"\nTop 10 players in {column_name} for each year:")
        for year in range(2015, 2019):
            year_data = df[df['year'] == year]
            top_10 = year_data.nlargest(10, column_name)
            print(f"\nYear {year}:")
            print(top_10[['batter_name', column_name]])

# Q2
def highest_war_by_position():
    df_2018 = df[df['year'] == 2018]
    positions = ['포수', '1루수', '2루수', '3루수', '유격수', '좌익수', '중견수', '우익수']
    
    print("\nPlayer with the highest war by position in 2018:")
    
    for position in positions:
        position_data = df_2018[df_2018['cp'] == position]
        
        if not position_data.empty:
            highest_war_player = position_data.nlargest(1, 'war')
            player_name = highest_war_player.iloc[0]['batter_name']
            war_value = round(highest_war_player.iloc[0]['war'], 2) 
            
            print(f"\nPosition: {position}")
            print(f"Player with the highest war: {player_name}")
            print(f"Highest war value: {war_value}")

# Q3
def highest_correlation_with_salary():
    selected_columns = ['R', 'H', 'HR', 'RBI', 'SB', 'war', 'avg', 'OBP', 'SLG']
    subset_df = df[selected_columns + ['salary']]
    correlations = subset_df.corr()
    highest_corr_feature = correlations['salary'].drop('salary').idxmax()
    highest_corr_value = correlations.loc[highest_corr_feature, 'salary']
    
    print(f"\n'{highest_corr_feature}' has the highest correlation with salary ({highest_corr_value:.2f})")

# Print
top_10_players()
highest_war_by_position()
highest_correlation_with_salary()