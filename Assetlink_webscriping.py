
from Taskassetlink import *
from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.document_transformers import BeautifulSoupTransformer
"""
CRD = dff.iloc[3,0]
loader =AsyncChromiumLoader(['https://adviserinfo.sec.gov/individual/summary/' + CRD])
html = loader.load()

# Transform
bs_transformer = BeautifulSoupTransformer()
docs_transformed = bs_transformer.transform_documents(html)

urls = ['https://adviserinfo.sec.gov/individual/summary/' + CRD]

loader = AsyncHtmlLoader(urls)
docs = loader.load()
"""
from langchain_community.document_transformers import Html2TextTransformer
"""
html2text = Html2TextTransformer()
docs_transformed = html2text.transform_documents(docs)
"""

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613", openai_api_key ="###################")
from langchain.chains import create_extraction_chain

schema = {
    "properties": {
        "First Name": {"type": "string"},
        "Middle Name": {"type": "string"},   
        "Last Name": {"type": "string"},  
        "Other Name": {"type": "string"},    
        "Broker Dealer": {"type": "string"}, 
        "Broker Dealer CRD": {"type": "string"},      
        "Years With Current BD": {"type": "string"},       
        "RIA": {"type": "string"},  
        "RIA CRD": {"type": "string"},  
        "Years With Current RIA": {"type": "string"},  
        "Current RIA State Date": {"type": "string"},  
        "Address": {"type": "string"},  
        "City": {"type": "string"},  
        "State": {"type": "string"},  
        "Zip": {"type": "string"},  
        "Metro Area": {"type": "string"},  
        "Licenses Exams": {"type": "string"},  
        "Title": {"type": "string"},  
        "Designations": {"type": "string"},  
        "Phone": {"type": "string"},  
        "Phone Type": {"type": "string"},  
        "Bio": {"type": "string"},  
        "Years Of Experience": {"type": "string"},  
        "Previous Broker Dealer": {"type": "string"},    
        "Previous RIA": {"type": "string"},   
        "Gender": {"type": "string"},   
        "TeamID": {"type": "string"},   
        "Person Tag Role": {"type": "string"},   
        "Person Tag Family": {"type": "string"},   
        "Person Tag Hobbies": {"type": "string"},   
        "Person Tag Expertise": {"type": "string"},   
        "Person Tag Services": {"type": "string"}, 
        "Person Tag Investments": {"type": "string"}, 
        "Person Tag SportsTeams": {"type": "string"}, 
        "Person Tag School": {"type": "string"}, 
        "Person Tag GreekLife": {"type": "string"}, 
        "Person Tag MilitaryStatus": {"type": "string"}, 
        "Person Tag FaithBasedInvesting": {"type": "string"}, 
        "Firm Tag Platform": {"type": "string"}, 
        "Firm Tag Technology": {"type": "string"}, 
        "Firm Tag Services": {"type": "string"}, 
        "Firm Tag Investments": {"type": "string"},  
        "Firm Tag Custodian": {"type": "string"},  
        "Firm Tag Clients": {"type": "string"},  
        "Firm Tag CRM": {"type": "string"},  
        "Profile": {"type": "string"},  
        "SEC Link": {"type": "string"},  
        "Company": {"type": "string"},  
        "Carrier": {"type": "string"},
    },
    "required": ["First Name", "Middle Name", "Last Name", "Other Name", "Broker Dealer", 
                 "Broker Dealer CRD", "Years With Current BD", "RIA", 
                 "RIA CRD", "Years With Current RIA", "Current RIA State Date",
                 "Address", "City", "State", "Zip", "Metro Area", 
                 "Licenses Exams", "Title", "Designations", "Phone", 
                 "Phone Type", "Bio", "Years Of Experience", 
                 "Previous Broker Dealer", "Previous RIA", 
                 "Gender", "TeamID", "Person Tag Role", 
                 "Person Tag Family", "Person Tag Hobbies", 
                 "Person Tag Expertise", "Person Tag Services", 
                 "Person Tag Investments", "Person Tag SportsTeams", 
                 "Person Tag School", "Person Tag GreekLife", 
                 "Person Tag MilitaryStatus",
                 "Person Tag FaithBasedInvesting", 
                 "Firm Tag Platform", 
                 "Firm Tag Services", 
                 "Firm Tag Investments",
                 "Firm Tag Custodian", 
                 "Firm Tag Clients",
                 "Firm Tag CRM",
                 "Profile",
                 "SEC Link",
                 "Company",
                 "Carrier"]}


def extract(content: str, schema: dict):
    return create_extraction_chain(schema=schema, llm=llm).run(content)

import pprint

from langchain_text_splitters import RecursiveCharacterTextSplitter

def scrape_with_playwright(urls, schema):
    loader = AsyncChromiumLoader(urls)
    docs = loader.load()
    bs_transformer = BeautifulSoupTransformer()
    docs_transformed = bs_transformer.transform_documents(
        docs
    )
    print("Extracting content with LLM")
    # Grab the first 1000 tokens of the site
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000, chunk_overlap=0
    )
    splits = splitter.split_documents(docs_transformed)
    # Process the first split
    extracted_content = extract(schema=schema, content=splits[0].page_content)
    pprint.pprint(extracted_content)
    return extracted_content



extracted_content = scrape_with_playwright(urls, schema=schema)

dff = dff['CRD']
dff['n_First_Name'] = ''
dff['n_Middle_Name'] = ''
dff['n_Last_Name'] = ''
dff['n_Other_Name'] = ''
dff['Broker Dealer'] = ''
dff['Broker Dealer CRD'] = ''
dff["Years With Current BD"] = ''     
dff["RIA"] = ''
dff["RIA CRD"] = '' 
dff["Years With Current RIA"]= ''    
dff["Current RIA State Date"]= ''    
dff["Address"]= ''    
dff["City"]= ''   
dff["State"]= ''   
dff["Zip"]= ''     
dff["Metro Area"]= ''   
dff["Licenses Exams"]= ''    
dff["Title"]= ''   
dff["Designations"]= ''   
dff["Phone"]= ''   
dff["Phone Type"]= ''   
dff["Bio"]= ''   
dff["Years Of Experience"]= ''    
dff["Previous Broker Dealer"]= ''    
dff["Previous RIA"]= ''   
dff["Gender"]= ''   
dff["TeamID"]= ''   
dff["Person Tag Role"]= ''   
dff["Person Tag Family"]= ''   
dff["Person Tag Hobbies"]= ''    
dff["Person Tag Expertise"]= ''   
dff["Person Tag Services"]= ''   
dff["Person Tag Investments"]= ''   
dff["Person Tag SportsTeams"]= ''   
dff["Person Tag School"]= ''   
dff["Person Tag GreekLife"]= ''   
dff["Person Tag MilitaryStatus"]= ''   
dff["Person Tag FaithBasedInvesting"]= ''   
dff["Firm Tag Platform"]= ''   
dff["Firm Tag Technology"]= ''   
dff["Firm Tag Services"]= ''   
dff["Firm Tag Investments"]= ''   
dff["Firm Tag Custodian"]= ''   
dff["Firm Tag Clients"]= ''   
dff["Firm Tag CRM"]= ''   
dff["Profile"]= ''   
dff["SEC Link"]= ''   
dff["Company"]= ''   
dff["Carrier"]= ''   



for i in range(len(dff)):
    CRD = dff.iloc[i,0]
    urls = ['https://adviserinfo.sec.gov/individual/summary/' + CRD]
    extracted_content = scrape_with_playwright(urls, schema=schema)
    try:
         dff['n_First_Name'][dff['CRD'] == dff.iloc[i, 0]] = extracted_content['First Name']
    except:
        pass
    try:
         dff['n_Middle_Name'][dff['CRD'] == dff.iloc[i, 0]] = extracted_content['Middle Name']     
    except:
        pass
    try:
         dff['n_Last_Name'][dff['CRD'] == dff.iloc[i, 0]] = extracted_content['Last Name']     
    except:
        pass
    try:
         dff['n_Other_Name'][dff['CRD'] == dff.iloc[i, 0]] = extracted_content['Other Name']     
    except:
        pass
    try:
         dff['Broker Dealer'][dff['CRD'] == dff.iloc[i, 0]] = extracted_content['Broker Dealer']     
    except:
        pass
    try:
         dff['Broker Dealer CRD'][dff['CRD'] == dff.iloc[i, 0]] = extracted_content['Broker Dealer CRD']     
    except:
        pass
    try:
         dff['Years With Current BD'][dff['CRD'] == dff.iloc[i, 0]] = extracted_content['Years With Current BD']
    except:
        pass
    try:
         dff['RIA'][dff['CRD'] == dff.iloc[i, 0]] = extracted_content['RIA']
    except:
        pass
    try:
         dff['RIA CRD'][dff['CRD'] == dff.iloc[i, 0]] = extracted_content['RIA CRD']
    except:
        pass
    try:
         dff['Years With Current RIA'][dff['CRD'] == dff.iloc[i, 0]] = extracted_content['Years With Current RIA']
    except:
        pass
    try:
         dff['Current RIA State Date'][dff['CRD'] == dff.iloc[i, 0]] = extracted_content['Current RIA State Date']
    except:
        pass
    try:
         dff['Address'][dff['CRD'] == dff.iloc[i, 0]] = extracted_content['Address']
    except:
        pass
    try:
         dff['City'][dff['CRD'] == dff.iloc[i, 0]] = extracted_content['City']
    except:
        pass
    try:
         dff['State'][dff['CRD'] == dff.iloc[i, 0]] = extracted_content['State']
    except:
        pass
    try:
         dff['Zip'][dff['CRD'] == dff.iloc[i, 0]] = extracted_content['Zip']
    except:
        pass
    try:
         dff['Metro Area'][dff['CRD'] == dff.iloc[i, 0]] = extracted_content['Metro Area']
    except:
        pass
    try:
         dff['Licenses Exams'][dff['CRD'] == dff.iloc[i, 0]] = extracted_content['Licenses Exams']
    except:
        pass
    try:
         dff['Title'][dff['CRD'] == dff.iloc[i, 0]] = extracted_content['Title']
    except:
        pass
    try:
         dff['Designations'][dff['CRD'] == dff.iloc[i, 0]] = extracted_content['Designations']
    except:
        pass
    try:
         dff['Phone'][dff['CRD'] == dff.iloc[i, 0]] = extracted_content['Phone']
    except:
        pass
    try:
         dff['Phone Type'][dff['CRD'] == dff.iloc[i, 0]] = extracted_content['Phone Type']
    except:
        pass
    try:
         dff['Bio'][dff['CRD'] == dff.iloc[i, 0]] = extracted_content['Bio']
    except:
        pass
    try:
         dff['Years Of Experience'][dff['CRD'] == dff.iloc[i, 0]] = extracted_content['Years Of Experience']
    except:
        pass
    try:
         dff['Previous Broker Dealer'][dff['CRD'] == dff.iloc[i, 0]] = extracted_content['Previous Broker Dealer']
    except:
        pass
    try:
         dff['Previous RIA'][dff['CRD'] == dff.iloc[i, 0]] = extracted_content['Previous RIA']
    except:
        pass
    try:
         dff['Gender'][dff['CRD'] == dff.iloc[i, 0]] = extracted_content['Gender']
    except:
        pass
    try:
         dff['TeamID'][dff['CRD'] == dff.iloc[i, 0]] = extracted_content['TeamID']
    except:
        pass
    try:
         dff['Person Tag Role'][dff['CRD'] == dff.iloc[i, 0]] = extracted_content['Person Tag Role']
    except:
        pass
    try:
         dff['Person Tag Family'][dff['CRD'] == dff.iloc[i, 0]] = extracted_content['Person Tag Family']
    except:
        pass
    try:
         dff['Person Tag Hobbies'][dff['CRD'] == dff.iloc[i, 0]] = extracted_content['Person Tag Hobbies']
    except:
        pass
    try:
         dff['Person Tag Expertise'][dff['CRD'] == dff.iloc[i, 0]] = extracted_content['Person Tag Expertise']
    except:
        pass
    try:
         dff['Person Tag Services'][dff['CRD'] == dff.iloc[i, 0]] = extracted_content['Person Tag Services']
    except:
        pass
    try:
         dff['Person Tag Investments'][dff['CRD'] == dff.iloc[i, 0]] = extracted_content['Person Tag Investments']
    except:
        pass
    try:
         dff['Person Tag SportsTeams'][dff['CRD'] == dff.iloc[i, 0]] = extracted_content['Person Tag SportsTeams']
    except:
        pass
    try:
         dff['Person Tag School'][dff['CRD'] == dff.iloc[i, 0]] = extracted_content['Person Tag School']
    except:
        pass
    try:
         dff['Person Tag GreekLife'][dff['CRD'] == dff.iloc[i, 0]] = extracted_content['Person Tag GreekLife']
    except:
        pass
    try:
         dff['Person Tag MilitaryStatus'][dff['CRD'] == dff.iloc[i, 0]] = extracted_content['Person Tag MilitaryStatus']
    except:
        pass
    try:
         dff['Person Tag FaithBasedInvesting'][dff['CRD'] == dff.iloc[i, 0]] = extracted_content['Person Tag FaithBasedInvesting']
    except:
        pass
    try:
         dff['Firm Tag Platform'][dff['CRD'] == dff.iloc[i, 0]] = extracted_content['Firm Tag Platform']
    except:
        pass
    try:
         dff['Firm Tag Technology'][dff['CRD'] == dff.iloc[i, 0]] = extracted_content['Firm Tag Technology']
    except:
        pass
    try:
         dff['Firm Tag Services'][dff['CRD'] == dff.iloc[i, 0]] = extracted_content['Firm Tag Services']
    except:
        pass
    try:
         dff['Firm Tag Investments'][dff['CRD'] == dff.iloc[i, 0]] = extracted_content['Firm Tag Investments']
    except:
        pass
    try:
         dff['Firm Tag Custodian'][dff['CRD'] == dff.iloc[i, 0]] = extracted_content['Firm Tag Custodian']
    except:
        pass
    try:
         dff['Firm Tag Clients'][dff['CRD'] == dff.iloc[i, 0]] = extracted_content['Firm Tag Clients']
    except:
        pass
    try:
         dff['Firm Tag CRM'][dff['CRD'] == dff.iloc[i, 0]] = extracted_content['Firm Tag CRM']
    except:
        pass
    try:
         dff['Profile'][dff['CRD'] == dff.iloc[i, 0]] = extracted_content['Profile']
    except:
        pass
    try:
         dff['SEC Link'][dff['CRD'] == dff.iloc[i, 0]] = extracted_content['SEC Link']        
    except:
        pass
    try:
         dff['Company'][dff['CRD'] == dff.iloc[i, 0]] = extracted_content['Company']
    except:
        pass
    try:
         dff['Carrier'][dff['CRD'] == dff.iloc[i,0]] = extracted_content['Carrier']
    except:
        pass
     
