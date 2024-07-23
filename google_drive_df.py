from io import StringIO
import pandas as pd
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive

def get_google_drive_df(file_link):
    gauth = GoogleAuth()
    
    #Try to load saved client credentials
    gauth.LoadCredentialsFile('client_secrets.json')
    
    if gauth.credentials is None:
        #Authenticate if they're not there
        gauth.LocalWebserverAuth()
    
    elif gauth.access_token_expired:
        #Refresh them if expired
        gauth.Refresh()

    else:
        #Initialize the saved creds
        gauth.Authorize()
        
    #Save the current credentials to a file
    gauth.SaveCredentialsFile('client_secrets.json')
        
    drive = GoogleDrive(gauth)
    
    file_id = file_link.split('/')[-2]
    file = drive.CreateFile({"id": file_id})
    content_io_buffer = file.GetContentIOBuffer()
    data = pd.read_csv(StringIO(content_io_buffer.read().decode()))
    return data
