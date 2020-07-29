# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 21:21:32 2020

@author: Giats
"""

import pandas as pd

df = pd.read_csv("car.gr_cars.csv")

#remove all bad rows (Some rows has columns like price on the category etc.)
df = df[df.Cubic_capacity.str.contains("cc")]
df = df[~df.Category.str.contains("€")]
df = df[df.Power.str.contains("bhp")]
df = df[~df.Power.str.contains(",")]
df = df[~df.Make_model.str.contains("Αλλο")]

#make brand name column from Make_model column
df['Brand'] = df['Make_model'].apply(lambda x: x.split(' ')[0])

#make model name column from Make_model column
df['Model'] = df['Make_model'].apply(lambda x: x.split(' ')[1])

#remove all bad rows with bad model name
df = df[~df.Model.str.contains("'")]
df = df[~df.Model.str.contains("Allroad")]
df = df[~df.Model.str.contains("Coupe")]
df = df[~df.Model.str.contains("Quattro")]
#remove models who contains no ascii 
def is_ascii(model):
    if (all(ord(c) < 128 for c in model)):
        return model
    else:
        return '-1'
df['Model'] = df.apply(lambda x: is_ascii(x.Model), axis=1)

#remove models who contains '.'
def removeModel(model):
    if '.' in model:
        return '-1'
    else:
        return model
df['Model'] = df.apply(lambda x: removeModel(x.Model), axis=1)

df = df[df.Model != '-1']

#price parse
df['Price'] = df['Price'].apply(lambda x: x.replace('.',''))
df['Price'] = df['Price'].apply(lambda x: int(x.split(' ')[0]))

#Registration parse
df['Registration'] = df['Registration'].apply(lambda x: int(x.split(' ')[-1]))

#Mileage parse
df['Mileage'] = df['Mileage'].apply(lambda x: x.replace(',',''))
df['Mileage'] = df['Mileage'].apply(lambda x: int(x.split(' ')[0]))
#if a car has <300 km and registration <2015 we should remove it
def removeMileage(km,year):
    if (km >300 or year>2015):
        return km
    else:
        return -1
df['Mileage'] = df.apply(lambda x: removeMileage(x.Mileage, x.Registration), axis=1)
df = df[df.Mileage != -1]

#Remove cars with fuel type == other
df = df[~df.Fuel_type.str.contains("Other")]

#Cubic_capacity parse
df['Cubic_capacity'] = df['Cubic_capacity'].apply(lambda x: x.replace(',',''))
df['Cubic_capacity'] = df['Cubic_capacity'].apply(lambda x: int(x.split(' ')[0]))

#if a car has cubic capacity == 1cc and its not electric we should remove it
def removeCC(cc,fuel):
    if (cc <500 and fuel != 'Electric'):
        return -1
    else:
        return cc
df['Cubic_capacity'] = df.apply(lambda x: removeCC(x.Cubic_capacity, x.Fuel_type), axis=1)
df = df[df.Cubic_capacity != -1]

#Power parse
df['Power'] = df['Power'].apply(lambda x: int(x.split(' ')[0]))

#if a car has power <45 or 650> we should remove it
df['Power'] = df['Power'].apply(lambda x: -1 if x < 45 or x >650 else x )
df = df[df.Power != -1]

#Color parse
df['Color'] = df['Color'].apply(lambda x: x.split('(')[0])

#Drive Type fill nulls
def fillDriveType(drive_type,model):
    new_drive_type = {}
    if (drive_type == '-1'):
        for i in ['FWD', '4x4', 'RWD']:
            try:
                new_drive_type[i] = group_drive_type[i,model]
            except:
                new_drive_type[i] = -1
        return max(new_drive_type, key=new_drive_type.get)
    else:
        return drive_type
#group by Drive_type and Model then size()
group_drive_type = df.groupby(["Drive_type", "Model"]).size()
df['Drive_type'] = df.apply(lambda x: fillDriveType(x.Drive_type, x.Model), axis=1)

#Airbags fill nulls
def fillAirbags(airbags,model):
    new_airbags = {}
    if (airbags == -1):
        for i in range(11):
            try:
                new_airbags[i] = group_airbags[i,model]
            except:
                new_airbags[i] = -1
        return max(new_airbags, key=new_airbags.get)
    else:
        return airbags
#group by Airbags and Model then size()
group_airbags = df.groupby(["Airbags", "Model"]).size()
df['Airbags'] = df.apply(lambda x: fillAirbags(x.Airbags, x.Model), axis=1)
df['Airbags'] = df['Airbags'].apply(lambda x: -1 if x == 0 else x)

#Seats fill nulls
def fillSeats(seats,model):
    new_seats = {}
    if (seats == -1):
        for i in range(12):
            try:
                new_seats[i] = group_seats[i,model]
            except:
                new_seats[i] = -1
        return max(new_seats, key=new_seats.get)
    else:
        return seats
#group by Seats and Model then size()
group_seats = df.groupby(["Seats", "Model"]).size()
df['Seats'] = df.apply(lambda x: fillSeats(x.Seats, x.Model), axis=1)

#Remove zip_code equals with nan
df = df[df['Zip_code'].notnull()]
#Zip_code parse
df['Zip_code'] = df['Zip_code'].apply(lambda x: int(x.split(' ')[-1]))

df.to_csv('data_cleaned.csv',index = False)