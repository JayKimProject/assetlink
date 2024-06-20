#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 17:25:14 2024
@author: jaykim
"""
#!/usr/bin/env python

import pymysql
import pandas as pd

conn = pymysql.connect(user='#########', passwd='##########', db='mysql')

cur = conn.cursor()
cur.execute("SELECT * FROM test limit 10")

print(cur.description)
print()

for i, row in enumerate(cur, start=0):
    if i ==0:
      dff = pd.DataFrame(row).T
      print(dff)
    else:
        df = pd.DataFrame(row).T
        dff = pd.concat([df, dff])
   
cur.close()
conn.close()

dff.columns = [
    'CRD',
    'NPN',
    'FirstName',
    'MiddleName',
    'LastName',
     'OtherNames'  ,
    'Team'  ,
    'BrokerDealer' ,
    'BrokerDealerCRD'  ,
    'YearsWithCurrentBD' ,
    'CurrentBDStartDate' ,
    'RIA' ,
    'RIACRD' ,
    'YearsWithCurrentRIA' ,
    'CurrentRIAStateDate' ,
    'Address' ,
    'City' ,
    'State' ,
    'Zip' ,
    'MetroArea' ,
    'LicensesExams',
    'Title' ,
    'Designations' ,
    'Phone' ,
    'PhoneType',
    'LinkedIn' ,
    'Email1',
    'Email2' ,
    'Email3' ,
    'PersonalEmail' ,
    'Bio' ,
    'YearsOfExperience' ,
    'EstAge',
    'PreviousBrokerDealer',
    'PreviousRIA' ,
    'Gender' ,
    'TeamID' ,
    'PersonTagRole',
    'PersonTagFamily',
    'PersonTagHobbies',
    'PersonTagExpertise' ,
    'PersonTagServices',
    'PersonTagInvestments',
    'PersonTagSportsTeams',
    'PersonTagSchool' ,
    'PersonTagGreekLife',
    'PersonTagMilitaryStatus',
    'PersonTagFaithBasedInvesting',
    'FirmTagPlatform' ,
    'FirmTagTechnology' ,
    'FirmTagServices' ,
    'FirmTagInvestments',
    'FirmTagCustodian' ,
    'FirmTagClients' ,
    'FirmTagCRM',
    'Notes' ,
    'Profile' ,
    'SECLink' ,
    'FINRALink' ,
    'FirmCompanyName',
    'FirmType',
    'FirmAddress',
    'FirmCity',
    'FirmState' ,
    'FirmZip' ,
    'FirmPhone' ,
    'FirmAUM',
    'FirmTotalAccounts',
    'FirmCustodians' ,
    'FirmTotalEmployees',
    'FirmRIAReps' ,
    'FirmBDReps',
    'FirmForm13F',
    'Line',
    'Carrier',
    'Company' ,
    'InitialAppointmentDate' ,
    'Captive' ,
    'NumOfCarriers' ,
    'NumOfLines',
    'NonProducing',
    'InsuranceYearsOfExperience']
