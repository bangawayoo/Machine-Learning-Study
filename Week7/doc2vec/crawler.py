import time 
from random import randint
import selenium
from selenium.webdriver import Chrome

def crawl_scholar(title_list, min_sleeping_time = 1):
	browser=Chrome()
	output = []
	browser.get('https://scholar.google.co.kr/')	
	try:
		for idx1, title in enumerate(title_list):
			browser.get('https://scholar.google.co.kr/')
			time.sleep(randint(min_sleeping_time, min_sleeping_time+3))
			search = browser.find_element_by_css_selector('#gs_hdr_tsi')
			search.clear()
			search.send_keys(title)
			#time.sleep(3)
			search_button = browser.find_element_by_css_selector('#gs_hdr_tsb > span')
			search_button.click()
			#time.sleep(3)
			try :
				abstract_table = browser.find_element_by_class_name('gs_rs')
				abstract = abstract_table.text
			except:
				print("Error at title {}: {}".format(idx1, e))
				abstract = ''
			output.append(abstract)
	except :
		return output
	return output 
                
                        
    