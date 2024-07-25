from google_drive_df import get_google_drive_df

def load_mlb_data(link_list):
    dfs = []
    for link in link_list:
        data = get_google_drive_df(link)
        dfs.append(data)
    
    return dfs