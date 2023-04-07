from urllib.request import urlopen
import os
import requests
from pathlib import Path
import time
import environ
import urllib.request
import requests
from PIL import Image
from datetime import datetime


class crawler:
    
    #BASE_DIR = Path(__file__).resolve().parent
     
    def __init__(self, query, start_page=1):

        BASE_DIR = os.getcwd()
        env = environ.Env(DEBUG=(bool, True))
        environ.Env.read_env(env_file=os.path.join(BASE_DIR, '.env'))  
        self.query = query
        self.start_page = start_page
        self.images = []
        self.is_activate = False
        self.env = env

    def activate(self):

        def get_response(url):
            res = requests.get(url)
            return res.json()
                
        response = get_response(f'https://url?client_id={self.env("ACCESS_KEY")}&query={self.query}')
        total_page = int(response['total_pages'])
        print(f"Found {total_page}")
        for element in response['results']:
                    self.images.append(element['urls']['regular'])
        
        for i in range(self.start_page+1,total_page+1):
            try:
                response = get_response(f'https://url?client_id={self.env("ACCESS_KEY")}&query={self.query}&page={i}')
                
                for element in response['results']:
                    self.images.append(element['urls']['regular'])
                print(f"{datetime.now().strftime('%Y.%m.%d - %H:%M:%S')} Accept {i} page")

            except Exception as e:
                
                print(f"{datetime.now().strftime('%Y.%m.%d - %H:%M:%S')} Reject {i} Page. Retry 1h later, start {i} Page.", e)
                self.start_page = i
                time.sleep((60*60)+1)

        print(f"{datetime.now().strftime('%Y.%m.%d - %H:%M:%S')} Image url count : {len(self.images)}.")
        self.is_activate = True
    
    
    def execute_images(self):
        if self.is_activate and self.images:
            def save_images(image_url, paths, i):
                import base64
                import io
                            
                if 'data:' in str(image_url):
                    pass
                else:
                    try:
                        t= urlopen(image_url).read()
                        t = requests.get(image_url)
                        t = Image.open(io.BytesIO(t.content))
                        #t = t.resize((64, 64))
                        t.save(os.path.join(paths, str(i)+".jpg"),format="JPEG")                        
                    except Exception as e:
                        print(e)
                        
                        
            def makedirs(path): 
                try: 
                        os.makedirs(path) 
                except OSError: 
                    if not os.path.isdir(path): 
                        raise
                        
                        
            for i, image in enumerate(self.images,1):
                base_path = os.path.dirname(os.path.abspath(os.getcwd()))
                save_path = base_path + f'/{self.query}'
                makedirs(save_path)
                save_images(image, save_path, i)
            print(f"Save path : {save_path}")
            return True
        
        
        else:
            if not self.images:
                print("No searching images. Please check URL and retry run.")
            elif not self.is_activate:
                print("No activate Crawler. Please check activate.")

                
    def deactivate(self):
        self.is_activate = False
        return True
    
    def show(self):
        print(self.images)
        return True
    
if __name__ == '__main__':
    c = crawler('crawling_imges')
    c.activate()
    c.show()
    c.execute_images()
    c.deactivate()