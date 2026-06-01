import os 

def ensure_directories(paths): 
        for path in  paths: 
                os.makedirs(path, exist_ok = True)