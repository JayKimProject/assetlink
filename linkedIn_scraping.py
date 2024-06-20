#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 17:38:35 2024

@author: jaykim
"""
#import pymysql
#import pandas as pd

from Assetlink_webscraping import *

import json
import requests
import logging
import csv
import uuid
import io
#from .models import ProfileResult
import threading
import numpy as np

logger = logging.getLogger(__name__)
API_KEY = "oaYzGxas1CfdOKjaRoONCg"
HEADERS = {'Authorization': f"Bearer {API_KEY}"}
#WEBHOOK_URL = 'https://your-url.ngrok-free.app/webhook/'

def get_personal_emails(linkedin_profile_url):
    api_endpoint = 'https://nubela.co/proxycurl/api/contact-api/personal-email'
    params = {'linkedin_profile_url': linkedin_profile_url, 'email_validation': 'include'}
    response = requests.get(api_endpoint, headers=HEADERS, params=params)
    return response.json() if response.status_code == 200 else None

def get_personal_numbers(linkedin_profile_url):
    api_endpoint = 'https://nubela.co/proxycurl/api/contact-api/personal-contact'
    params = {'linkedin_profile_url': linkedin_profile_url}
    response = requests.get(api_endpoint, headers=HEADERS, params=params)
    return response.json() if response.status_code == 200 else None

def lookup_work_email(linkedin_profile_url):
    api_endpoint = 'https://nubela.co/proxycurl/api/linkedin/profile/email'
    params = {'linkedin_profile_url': linkedin_profile_url} #, 'callback_url': WEBHOOK_URL}
    requests.get(api_endpoint, headers=HEADERS, params=params)

dff['Personal_EMAIL_acquire'] = ''
dff['Personal_Invalid_EMAIL_acquire'] = ''
dff['PHONE_numbers'] = ''
dff['Work_emails'] = ''

for i in range(len(dff)):
    if dff['LinkedIn'].iloc[i,] != '':
        try:
           dff['Personal_EMAIL_acquire'].iloc[i,]  =  get_personal_emails(dff['LinkedIn'].iloc[i,])['emails']
        except:
            pass
        try:
           dff['Personal_Invalid_EMAIL_acquire'].iloc[i,]  =  get_personal_emails(dff['LinkedIn'].iloc[i,])['emails']
        except:
            pass
        try:
           dff['PHONE_numbers'].iloc[i,] = get_personal_numbers(dff['LinkedIn'].iloc[i,])['numbers']
        except:
            pass
        try:
           dff['Work_emails'].iloc[i,] = lookup_work_email(dff['LinkedIn'].iloc[i,])
        except:
            pass
        


