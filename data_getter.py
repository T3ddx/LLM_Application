from selenium import webdriver
from selenium.webdriver.common.by import By
import time
from selenium.webdriver.edge.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import os

start_value = 17947
end_value = 18418

def main():
    files_removed = 0
    for num, value in enumerate(range(start_value, end_value+1)):
        file_name = f'major_data/major_data_{num-files_removed}.txt'
        file = open(file_name, 'a')
        #_service = Service("./msedgedriver.exe")

        _options = webdriver.EdgeOptions()
        _options.add_argument("--disable-notifications")
        _options.add_argument("--disable-extensions")
        _options.add_experimental_option("detach", True)

        driver = webdriver.Edge(options=_options)
    
        driver.get(f'https://courses.syracuse.edu/preview_program.php?catoid=35&poid={value}')
        #print(driver.find_element(By.CLASS_NAME, 'block_content').text)
        
        try:
            if driver.find_element(By.ID, 'acalog-page-title').text == '2023-2024 Undergraduate Course Catalog':
                raise 'Error'
            file.write(driver.find_element(By.CLASS_NAME, 'block_content').text)
            file.write('\n\n')
            file.close()
        except:
            file.close()
            os.remove(file_name)
            files_removed += 1
        driver.close() 

main()

#resp = requests.get(f'https://courses.syracuse.edu/preview_program.php?catoid=35&poid={start_value}')
#print(resp.status_code)
#print(resp.content)