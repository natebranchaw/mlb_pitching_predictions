from google_drive_df import get_google_drive_df

google_drive_player_mapping_link = 'https://drive.google.com/file/d/11Wswb5fHy_BWkQnZ28SSdU6nsUDSomhu/view?usp=drive_link'
google_drive_p_statcast_link='https://drive.google.com/file/d/1LSO1jY5Tbek6ZN24PuVdfe4gdiZW-vJu/view?usp=drive_link'
google_drive_p_data_link = 'https://drive.google.com/file/d/1IezCfAMHw5UFroFonCa_ikJqij6QpLEi/view?usp=drive_link'

player_mapping_df = get_google_drive_df(google_drive_player_mapping_link)
p_statcast_df = get_google_drive_df(google_drive_p_statcast_link)
p_traditional_df = get_google_drive_df(google_drive_p_data_link)

