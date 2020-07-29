# -*- coding: utf-8 -*-
"""
Created on Thu May 14 23:49:52 2020

@author: Giats
"""

from selenium.common.exceptions import NoSuchElementException, ElementClickInterceptedException
from selenium.common.exceptions import TimeoutException
from selenium import webdriver
import time
import pandas as pd
from urllib.parse import urlparse

num_cars = 50000
path = "chromedriver"
verbose = False

'''Gathers cars as a dataframe, scraped from Gar.gr'''
'''Condision : Used, Offer type: Sale, Price: over 50€, Crashed: No'''

#Initializing the webdriver
options = webdriver.ChromeOptions()
options.add_argument("enable-automation");
options.add_argument("--headless");
options.add_argument("--no-sandbox");
options.add_argument("enable-features=NetworkServiceInProcess")
options.add_argument("--disable-extensions");
options.add_argument("--dns-prefetch-disable");
options.page_load_strategy = 'eager'
#Change the path to where chromedriver is in your home folder.
driver = webdriver.Chrome(executable_path=path, options=options)
driver.set_window_size(1200, 1000)

url = 'https://www.car.gr/classifieds/cars/?condition=used&lang=en&offer_type=sale&onlyprice=1&pg=1&price=%3E50&significant_damage=f&sort=dm'
driver.get(url)
cars = []
df = pd.read_csv("car.gr_cars.csv")
while len(cars) < num_cars:  #If true, should be still looking for new jobs.
    
    #Let the page load. Change this number based on your internet speed.
    #Or, wait until the webpage is loaded, instead of hardcoding it.
    time.sleep(2)
    
    #Number of cars i got in current page
    no_car=0;
    
    #Going through each car in this page
    car_buttons = driver.find_elements_by_class_name("clsfd_list_row_group")
    
    for car_button in car_buttons:
        
        #link of used car
        car_link = car_button.find_element_by_css_selector('a').get_attribute('href')
        
        #save main window
        main_window = driver.current_window_handle
        
        #get car id, we need it on xPaths
        urlparse(car_link)
        path = urlparse(car_link).path[1:]
        parts = path.split('-')
        
        #copy last 23 cars in new dataframe
        check_df = df[len(df)-100:]
                
        #car.gr inserts cars every second so when we change page we are getting same cars
        #check if new car is in our dataframe
        if not (str(parts[0]) in check_df['Classified_number'].values) :
            
            #open used car link in new tab
            driver.execute_script("window.open('"+car_link+"', '_blank')")
            driver.switch_to_window(driver.window_handles[1])
            
            collected_successfully = False
            while not collected_successfully:
                try:
                    table  = driver.find_element_by_css_selector(".vtable.table.table-striped.table-condensed")
                    rows = table.find_elements_by_tag_name("tr")
                    make_model = rows[0].find_elements_by_tag_name("td")[1].text
                    classified_number = rows[1].find_elements_by_tag_name("td")[1].text
                    price = rows[2].find_elements_by_tag_name("td")[1].text 
                    category = rows[3].find_elements_by_tag_name("td")[1].text 
                    registration = rows[4].find_elements_by_tag_name("td")[1].text 
                    mileage = rows[5].find_elements_by_tag_name("td")[1].text 
                    fuel_type = rows[6].find_elements_by_tag_name("td")[1].text 
                    cubic_capacity = rows[7].find_elements_by_tag_name("td")[1].text 
                    power = rows[8].find_elements_by_tag_name("td")[1].text 
                    transmission  = rows[9].find_elements_by_tag_name("td")[1].text 
                    color  = rows[10].find_elements_by_tag_name("td")[1].text 
                    div_zip = driver.find_elements_by_class_name("information-outline")
                    zip_code = div_zip[1].find_elements_by_tag_name("p")[1].text
                    
                    #next variables are not everytime in same order so we must find them
                    number_plate = -1
                    previous_owners = -1
                    drive_type = -1
                    airbags = -1
                    doors = -1
                    seats = -1
                    for row in rows[11:]:
                        td_category = row.find_elements_by_tag_name("td")[0].text
                        if (td_category == "Number plate:" ) or (td_category == "Previous owners:" ) or (td_category == "Drive type:" ) or (td_category == "Airbags:" ) or (td_category == "Doors:" ) or (td_category == "Seats:"):
                            td_category = td_category.lower()
                            td_category = td_category.replace(" ","_")
                            td_category = td_category.replace(":","")
                            exec(td_category + " = '"+row.find_elements_by_tag_name("td")[1].text+"'")
                    collected_successfully = True
                except (NoSuchElementException,TimeoutException):
                    #if something went wrong try again 
                    time.sleep(2)
                    driver.refresh()
            
            #Printing for debugging
            if verbose:
                print("Make/Model: {}".format(make_model))
                print("Classified number: {}".format(classified_number))
                print("Price: {}".format(price))
                print("Category: {}".format(category))
                print("Registration: {}".format(registration))
                print("Mileage: {}".format(mileage))
                print("Fuel type: {}".format(fuel_type))
                print("Cubic capacity: {}".format(cubic_capacity))
                print("Power: {}".format(power))
                print("Transmission: {}".format(transmission))
                print("Color: {}".format(color))
                print("Number Plate: {}".format(number_plate))
                print("Previous owners: {}".format(previous_owners))
                print("Drive type: {}".format(drive_type))
                print("Airbags: {}".format(airbags))
                print("Doors: {}".format(doors))
                print("Seats: {}".format(seats))
                print("Zip Code: {}".format(zip_code))
            
            #close new tab
            driver.close()
                
            # back to the main window
            driver.switch_to_window(main_window)
                
            #if we got values then increase no_car and append new car to our list           
            if collected_successfully:
                #add car to cars 
                cars.append({"Make_model" : make_model,
                    "Classified_number" : classified_number,
                    "Price" : price,
                    "Category" : category,
                    "Registration" : registration,
                    "Mileage" : mileage,
                    "Fuel_type" : fuel_type,
                    "Cubic_capacity" : cubic_capacity,
                    "Power" : power,
                    "Transmission" : transmission,
                    "Color" : color,
                    "Number_plate" : number_plate,
                    "Previous_owners" : previous_owners,
                    "Drive_type" : drive_type,
                    "Airbags" : airbags,
                    "Doors":doors,
                    "Seats" : seats,
                    "Zip_code" : zip_code})
                df = df.append(pd.DataFrame(cars[-1],index=[0]))
                df.to_csv("car.gr_cars.csv",  index=False)
        no_car = no_car + 1
        #check collected cars length
        print("Progress: {}".format("" + str(len(df)) + "/" + str(num_cars)))
        if len(df) >= num_cars:
            break
            
        #Clicking on the "next page" button
        if no_car == len(car_buttons):
            page_changed = False
            while not page_changed:
                try:
                    next_page = driver.find_element_by_class_name("next")
                    next_page = next_page.get_attribute("href")
                    driver.get(next_page)
                    page_changed = True
                except (NoSuchElementException,TimeoutException):
                    time.sleep(2)
                    driver.refresh();
       