import spacy
import pycountry
from collections import defaultdict
import pandas as pd

class DatasetBias:
    def __init__(self):
        pass

    @staticmethod
    def get_location_distribution(df):

        # Load the English spaCy model
        nlp = spacy.load("en_core_web_sm")

        text_list = df[df['input'].notnull()].reset_index(drop=True)['input'].tolist()

        locations = []
        for text in text_list:
            # Process the text using spaCy
            doc = nlp(text)

            # Iterate through entities and check for "GPE" (Geopolitical Entity) labels
            for ent in doc.ents:
                if ent.label_ == "GPE":
                    locations.append(ent.text)


        # Get a list of all countries
        countries_set = set([i.name.lower() for i in list(pycountry.countries)])
        locations_set = set([i.lower() for i in locations])

        existing_countries = []
        for country in countries_set:
            if country.lower() in locations_set:
                existing_countries.append(country.lower())

        hashpmap_counter = defaultdict(int)
        for idx, exist_country in enumerate(existing_countries):
            for text in text_list:
                if exist_country in text.lower():
                    hashpmap_counter[exist_country]+=1

        res_df = pd.DataFrame(hashpmap_counter, index=[0]).T
        res_df = res_df.reset_index().rename(columns={0:'Counts', 'index':"Country"}).sort_values(by = 'Counts', ascending=False)
        return res_df

    @staticmethod
    def get_person_distribution(df):
        nlp = spacy.load("en_core_web_sm")

        text_list = df[df['input'].notnull()].reset_index(drop=True)['input'].tolist()

        person_counter = defaultdict(int)
        for text in text_list:
            # Process the text using spaCy
            doc = nlp(text)

            # Iterate through entities and check for "PERSON" labels
            for ent in doc.ents:
                if ent.label_ == "PERSON":
                    # persons.append(ent.text)
                    person_counter[ent.text.lower()]+=1

        res_df = pd.DataFrame(person_counter, index=[0]).T
        res_df = res_df.reset_index().rename(columns={'index':'Person', 0:'Counts'}).sort_values(by='Counts', ascending=False).head(10)
        return res_df
    
    @staticmethod
    def get_norp_distribution(df):
        nlp = spacy.load("en_core_web_sm")

        text_list = df[df['input'].notnull()].reset_index(drop=True)['input'].tolist()

        norp_counter = defaultdict(int)
        for text in text_list:
            # Process the text using spaCy
            doc = nlp(text)

            # Iterate through entities and check for "NORP" labels
            for ent in doc.ents:
                if ent.label_ == "NORP":
                    norp_counter[ent.text.lower()]+=1

        res_df = pd.DataFrame(norp_counter, index=[0]).T
        res_df = res_df.reset_index().rename(columns={'index':'Norp', 0:'Counts'}).sort_values(by='Counts', ascending=False).head(10)
        return res_df

    @staticmethod
    def get_event_distribution(df):
        nlp = spacy.load("en_core_web_sm")

        text_list = df[df['input'].notnull()].reset_index(drop=True)['input'].tolist()

        event_counter = defaultdict(int)
        for text in text_list:
            # Process the text using spaCy
            doc = nlp(text)

            # Iterate through entities and check for "EVENT" labels
            for ent in doc.ents:
                if ent.label_ == "EVENT":
                    event_counter[ent.text.lower()]+=1

        res_df = pd.DataFrame(event_counter, index=[0]).T
        res_df = res_df.reset_index().rename(columns={'index':'Event', 0:'Counts'}).sort_values(by='Counts', ascending=False).head(10)
        return res_df

    @staticmethod
    def get_product_distribution(df):
        nlp = spacy.load("en_core_web_sm")

        text_list = df[df['input'].notnull()].reset_index(drop=True)['input'].tolist()

        event_counter = defaultdict(int)
        for text in text_list:
            # Process the text using spaCy
            doc = nlp(text)

            # Iterate through entities and check for "PRODUCT" labels
            for ent in doc.ents:
                if ent.label_ == "PRODUCT":
                    event_counter[ent.text.lower()]+=1

        res_df = pd.DataFrame(event_counter, index=[0]).T
        res_df = res_df.reset_index().rename(columns={'index':'Product', 0:'Counts'}).sort_values(by='Counts', ascending=False).head(10)
        return res_df