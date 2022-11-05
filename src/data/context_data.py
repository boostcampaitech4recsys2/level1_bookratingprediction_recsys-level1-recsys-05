import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Dataset

def age_map(x: int) -> int:
    x = int(x)
    if x < 20:
        return 1
    elif x >= 20 and x < 30:
        return 2
    elif x >= 30 and x < 40:
        return 3
    elif x >= 40 and x < 50:
        return 4
    elif x >= 50 and x < 60:
        return 5
    else:
        return 6

def process_context_data(users, books, ratings1, ratings2):
    users['location_city'] = users['location'].apply(lambda x: x.split(',')[0])
    users['location_state'] = users['location'].apply(lambda x: x.split(',')[1])
    users['location_country'] = users['location'].apply(lambda x: x.split(',')[2])
    users = users.drop(['location'], axis=1)
    
    ######################### location 전처리
    users['location_city'] = users['location_city'].str.strip()
    users['location_state'] = users['location_state'].str.strip()
    users['location_country'] = users['location_country'].str.strip()
    users['location_city'] = users['location_city'].str.replace(r'[^a-zA-Z]', '', regex=True)
    users['location_state'] = users['location_state'].str.replace(r'[^a-zA-Z]', '', regex=True)
    users['location_country'] = users['location_country'].str.replace(r'[^a-zA-Z]', '', regex=True)
    '''
    location_country
    '''
    # null & na & universe & etc
    null_repl = [
        'universe', 'na', '', 'lava', 'petrolwarnation', 'space', 'lachineternelle',
        'faraway', 'everywhereandanywhere', 'hereandthere', 'tdzimi', 'naontheroad',
        'unknown'
    ]
    for keyword in null_repl:
        users.loc[users['location_country'] == keyword, 'location_country'] = 'null'
    users.loc[users['location_country'] == 'c', 'location_country'] = 'null'
    # australia
    australia_repl = [
        'newsouthwales', 'queensland', 'tasmania', 'victoria', 'nsw', 'southaustralia'
    ]
    for keyword in australia_repl:
        users.loc[users['location_country'].str.contains(keyword), 'location_country'] = 'australia'
    # italy
    users.loc[users['location_country'].str.contains('ital'), 'location_country'] = 'italy'
    users.loc[users['location_country'].str.contains('ferrara'), 'location_country'] = 'italy'
    users.loc[users['location_country'].str.contains('veneziagiulia'), 'location_country'] = 'italy'
    users.loc[users['location_country'].str.contains('ineurope'), 'location_country'] = 'italy'
    # germany
    users.loc[users['location_country'].str.contains('deut'), 'location_country'] = 'germany'
    users.loc[users['location_country'].str.contains('germ'), 'location_country'] = 'germany'
    users.loc[users['location_country'].str.contains('berlin'), 'location_country'] = 'germany'
    users.loc[users['location_country'].str.contains('niedersachsen'), 'location_country'] = 'germany'
    # united kingdom
    uk_repls = [
        'unitedkingdom', 'eng', 'king', 'wales', 'scotland', 'aberdeenshire', 'camden', 'unitedkindgonm',
        'middlesex', 'nottinghamshire', 'westyorkshire', 'cambridgeshire', 'sthelena', 'northyorkshire',
        'obviously'
    ]
    for keyword in uk_repls:
        users.loc[users['location_country'].str.contains(keyword), 'location_country'] = 'united kingdom'
    users.loc[users['location_country'] == 'uk', 'location_country'] = 'united kingdom'
    # ireland
    users.loc[users['location_country'].str.contains('countycork'), 'location_country'] = 'ireland'
    users.loc[users['location_country'].str.contains('cocarlow'), 'location_country'] = 'ireland'
    # france
    users.loc[users['location_country'].str.contains('fran'), 'location_country'] = 'france'
    users.loc[users['location_country'].str.contains('paris'), 'location_country'] = 'france'
    # spain
    spain_repl = [
        'esp', 'catal', 'galiza', 'euskalherria', 'lleida', 'gipuzkoa', 'orense', 'pontevedra', 'almera',
        'bergued', 'andalucia'
    ]
    for keyword in spain_repl:
        users.loc[users['location_country'].str.contains(keyword), 'location_country'] = 'spain'
    # portugal
    users.loc[users['location_country'].str.contains('oeiras'), 'location_country'] = 'portugal'
    # belgium
    users.loc[users['location_country'].str.contains('labelgique'), 'location_country'] = 'belgium'
    # austria
    users.loc[users['location_country'].str.contains('eu'), 'location_country'] = 'austria'
    # swiss
    users.loc[users['location_country'].str.contains('lasuisse'), 'location_country'] = 'switzerland'
    # finland
    users.loc[users['location_country'].str.contains('etelsuomi'), 'location_country'] = 'finland'
    # usa
    usa_repl = [
        'unitedstaes', 'america', 'usa', 'state', 'sate', 'cali', 'dc', 'oregon', 'texas', 'florida',
        'newhampshire', 'newmexico', 'newjersey', 'newyork', 'virginia', 'bermuda', 'illinois', 'michigan',
        'arizona', 'indiana', 'minnesota', 'tennessee', 'dakota', 'connecticut', 'wisconsin', 'ohio',
        'maryland', 'northcarolina', 'massachusetts', 'colorado', 'washington', 'maine', 'georgia', 'oklahoma',
        'maracopa', 'districtofcolumbia', 'saintloius', 'orangeco', 'aroostook', 'arkansas', 'montana',
        'rhodeisland', 'nevada', 'kern', 'fortbend', 'nebraska', 'usofa', 'alabama', 'csa', 'polk',
        'alachua', 'austin', 'alaska', 'hawaii', 'worcester', 'iowa', 'cherokee', 'shelby', 'stthomasi',
        'vanwert', 'kansas', 'idaho', 'tn', 'framingham', 'pender', 'ysa', 'arizona', 'morgan', 'rutherford'
    ]
    for keyword in usa_repl:
        users.loc[users['location_country'].str.contains(keyword), 'location_country'] = 'usa'
    users.loc[users['location_country'] == 'us', 'location_country'] = 'usa'
    users.loc[users['location_country'] == 'ca', 'location_country'] = 'usa'
    users.loc[users['location_country'] == 'il', 'location_country'] = 'usa'
    users.loc[users['location_country'] == 'ua', 'location_country'] = 'usa'
    # cananda
    canada_repl = [
        'cananda', 'british', 'newfoundland', 'newbrunswick', 'alberta', 'ontario', 'lkjlj', 'bc',
        'novascotia', 'kcb', 'quebec', 'maricopa', 'travelling', 'vvh', 'saskatchewan'
    ]
    for keyword in canada_repl:
        users.loc[users['location_country'].str.contains(keyword), 'location_country'] = 'canada'
    # new zealand
    users.loc[users['location_country'] == 'nz', 'location_country'] = 'newzealand'
    users.loc[users['location_country'].str.contains('otago'), 'location_country'] = 'newzealand'
    users.loc[users['location_country'].str.contains('auckland'), 'location_country'] = 'newzealand'
    # malaysia
    users.loc[users['location_country'].str.contains('kedah'), 'location_country'] = 'malaysia'
    # uae
    users.loc[users['location_country'].str.contains('uae'), 'location_country'] = 'unitedarabemirates'
    # kuwait
    users.loc[users['location_country'].str.contains('quit'), 'location_country'] = 'kuwait'
    # phillipines
    users.loc[users['location_country'].str.contains('phill'), 'location_country'] = 'philippines'
    users.loc[users['location_country'].str.contains('metromanila'), 'location_country'] = 'philippines'
    # uruguay
    users.loc[users['location_country'].str.contains('urugua'), 'location_country'] = 'uruguay'
    # panama
    users.loc[users['location_country'].str.contains('republicofpanama'), 'location_country'] = 'panama'
    # trinidadandtobago
    users.loc[users['location_country'].str.contains('westindies'), 'location_country'] = 'trinidadandtobago'
    # guernsey
    users.loc[users['location_country'].str.contains('alderney'), 'location_country'] = 'guernsey'
    # japan
    users.loc[users['location_country'].str.contains('okinawa'), 'location_country'] = 'japan'
    # korea
    users.loc[users['location_country'].str.contains('seoul'), 'location_country'] = 'southkorea'
    # brazil
    users.loc[users['location_country'].str.contains('disritofederal'), 'location_country'] = 'brazil'
    '''
    location_state
    '''
    # usa
    usa_state_repl = [
        'usa', 'texas', 'tx', 'california', 'massachusetts', 'michigan', 'carolina', 'florida', 'colorado', 'pennsylvania',
        'newyork', 'newjersey', 'virginia', 'dc', 'washington', 'iowa', 'illinois', 'georgia', 'kansas', 'missouri',
        'mississippi', 'oregon', 'arizona', 'ohio', 'tennessee', 'idaho', 'alaska', 'alabama', 'minnesota', 'utah',
        'kentucky', 'rhodeisland', 'maryland', 'louisiana', 'indiana', 'connecticut', 'wisconsin', 'newhampshire',
        'nevada', 'oklahoma', 'georgia', 'maine', 'newmexico', 'nebraska', 'wyoming', 'frenchquarter', 'fl', 'nebr', 'ct',

    ]
    for keyword in usa_state_repl:
        users.loc[(users['location_country'] == 'null') & (users['location_state'].str.contains(keyword)), 'location_country'] = 'usa'
    # canada
    canada_state_repl = [
        'britishcolumbia', 'newbrunswick', 'novascotia', 'ontario', 'alberta', 'quebec', 'saskatchewan',
        'manitoba', 
    ]
    for keyword in canada_state_repl:
        users.loc[(users['location_country'] == 'null') & (users['location_state'].str.contains(keyword)), 'location_country'] = 'canada'
    # mexico
    users.loc[(users['location_country'] == 'null') & (users['location_state'].str.contains('jalisco')), 'location_country'] = 'mexico'
    users.loc[(users['location_country'] == 'null') & (users['location_state'].str.contains('michoacan')), 'location_country'] = 'mexico'
    users.loc[(users['location_country'] == 'null') & (users['location_state'].str.contains('morelos')), 'location_country'] = 'mexico'
    # united kingdom
    uk_state_repl = [
        'newhampshire', 'nottinghamshire', 'england', 'middlesex', 'midlothian', 'scotland', 'westyorkshire',
        'canterbury', 'wiltshire', 'kent', 'london', 'cambs', 'herts', 'isleofman', 'surrey', 'cheshire',
        'gloucestershire', 'aberdeenshire'
    ]
    for keyword in uk_state_repl:
        users.loc[(users['location_country'] == 'null') & (users['location_state'].str.contains(keyword)), 'location_country'] = 'united kingdom'
    # ireland
    users.loc[(users['location_country'] == 'null') & (users['location_state'].str.contains('dublin')), 'location_country'] = 'ireland'
    users.loc[(users['location_country'] == 'null') & (users['location_state'].str.contains('wicklow')), 'location_country'] = 'ireland'
    users.loc[(users['location_country'] == 'null') & (users['location_state'].str.contains('colimerick')), 'location_country'] = 'ireland'
    # australia
    australia_state_repl = [
        'newsouthwales', 'victoria', 'australiancapitalterritory', 'southaustralia', 'nsw'
    ]
    for keyword in australia_state_repl:
        users.loc[(users['location_country'] == 'null') & (users['location_state'].str.contains(keyword)), 'location_country'] = 'australia'
    # germany
    germany_state_repl = [
        'nordrheinwestfalen', 'bayern', 'hamburg', 'badenwuerttemberg', 'badenwrttemberg', 'sachsen', 'berlin',
        'stuttgart', 'nrw', 'bavaria', 'bremen'
    ]
    for keyword in germany_state_repl:
        users.loc[(users['location_country'] == 'null') & (users['location_state'].str.contains(keyword)), 'location_country'] = 'germany'
    # switzerland
    users.loc[(users['location_country'] == 'null') & (users['location_state'].str.contains('bern')), 'location_country'] = 'switzerland'
    # austria
    users.loc[(users['location_country'] == 'null') & (users['location_state'].str.contains('niederoesterreich')), 'location_country'] = 'austria'
    users.loc[(users['location_country'] == 'null') & (users['location_state'].str.contains('vienna')), 'location_country'] = 'austria'
    users.loc[(users['location_country'] == 'null') & (users['location_state'].str.contains('wien')), 'location_country'] = 'austria'
    # slovenia
    users.loc[(users['location_country'] == 'null') & (users['location_state'].str.contains('ljubljanskaregija')), 'location_country'] = 'slovenia'
    # spain
    spain_state_repl = [
        'catalunya', 'pontevedra', 'madrid', 'bizkaia', 'asturias', 'pontevedra', 'barcelona', 'pasvasco',
        'espaa', 'badajoz', 'gipuzkoa', 'valencia', 'galicia'
    ]
    for keyword in spain_state_repl:
        users.loc[(users['location_country'] == 'null') & (users['location_state'].str.contains(keyword)), 'location_country'] = 'spain'
    # portugal
    users.loc[(users['location_country'] == 'null') & (users['location_state'].str.contains('lisboa')), 'location_country'] = 'portugal'
    users.loc[(users['location_country'] == 'null') & (users['location_state'].str.contains('coimbra')), 'location_country'] = 'portugal'
    users.loc[(users['location_country'] == 'null') & (users['location_state'].str.contains('porto')), 'location_country'] = 'portugal'
    users.loc[(users['location_country'] == 'null') & (users['location_state'].str.contains('estremadura')), 'location_country'] = 'portugal'
    # netherlands
    netherlands_state_repl = [
        'noordholland', 'utrecht', 'zuidholland', 'overijssel', 'friesland', 'northholland', 'schleswigholstein',
        'zh', 'twente', 
    ]
    for keyword in netherlands_state_repl:
        users.loc[(users['location_country'] == 'null') & (users['location_state'].str.contains(keyword)), 'location_country'] = 'netherlands'
    # belgium
    belgium_state_repl = [
        'vlaamsbrabant', 'liege'
    ]
    for keyword in belgium_state_repl:
        users.loc[(users['location_country'] == 'null') & (users['location_state'].str.contains(keyword)), 'location_country'] = 'belgium'
    # new zealand
    users.loc[(users['location_country'] == 'null') & (users['location_state'].str.contains('auckland')), 'location_country'] = 'newzealand'
    users.loc[(users['location_country'] == 'null') & (users['location_state'].str.contains('northisland')), 'location_country'] = 'newzealand'
    users.loc[(users['location_country'] == 'null') & (users['location_state'].str.contains('waikato')), 'location_country'] = 'newzealand'
    # italy
    users.loc[(users['location_country'] == 'null') & (users['location_state'].str.contains('italia')), 'location_country'] = 'italy'
    users.loc[(users['location_country'] == 'null') & (users['location_state'].str.contains('toscana')), 'location_country'] = 'italy'
    users.loc[(users['location_country'] == 'null') & (users['location_state'].str.contains('piemonte')), 'location_country'] = 'italy'
    users.loc[(users['location_country'] == 'null') & (users['location_state'].str.contains('lombardia')), 'location_country'] = 'italy'
    users.loc[(users['location_country'] == 'null') & (users['location_state'].str.contains('gorizia')), 'location_country'] = 'italy'
    users.loc[(users['location_country'] == 'null') & (users['location_state'] == 're'), 'location_country'] = 'italy'
    # greece
    users.loc[(users['location_country'] == 'null') & (users['location_state'].str.contains('townofbali')), 'location_country'] = 'greece'
    # nigeria
    users.loc[(users['location_country'] == 'null') & (users['location_state'].str.contains('imostate')), 'location_country'] = 'nigeria'
    # southafrica
    users.loc[(users['location_country'] == 'null') & (users['location_state'].str.contains('westerncape')), 'location_country'] = 'southafrica'
    # romania
    users.loc[(users['location_country'] == 'null') & (users['location_state'].str.contains('ilfov')), 'location_country'] = 'romania'
    # malaysia
    users.loc[(users['location_country'] == 'null') & (users['location_state'].str.contains('penang')), 'location_country'] = 'malaysia'
    users.loc[(users['location_country'] == 'null') & (users['location_state'].str.contains('negerisembilan')), 'location_country'] = 'malaysia'
    # indonesia
    users.loc[(users['location_country'] == 'null') & (users['location_state'].str.contains('jakarta')), 'location_country'] = 'indonesia'
    # philippines
    users.loc[(users['location_country'] == 'null') & (users['location_state'].str.contains('laguna')), 'location_country'] = 'philippines'
    # singapore
    users.loc[(users['location_country'] == 'null') & (users['location_state'].str.contains('singapore')), 'location_country'] = 'singapore'
    # pakistan
    users.loc[(users['location_country'] == 'null') & (users['location_state'].str.contains('punjab')), 'location_country'] = 'pakistan'
    # denmark
    users.loc[(users['location_country'] == 'null') & (users['location_state'].str.contains('jutland')), 'location_country'] = 'denmark'
    # france
    users.loc[(users['location_country'] == 'null') & (users['location_state'].str.contains('lorraine')), 'location_country'] = 'france'
    users.loc[(users['location_country'] == 'null') & (users['location_state'].str.contains('hautegaronne')), 'location_country'] = 'france'
    users.loc[(users['location_country'] == 'null') & (users['location_state'].str.contains('heraut')), 'location_country'] = 'france'
    users.loc[(users['location_country'] == 'null') & (users['location_state'].str.contains('rhnealpes')), 'location_country'] = 'france'
    users.loc[(users['location_country'] == 'null') & (users['location_state'].str.contains('iledefrance')), 'location_country'] = 'france'
    users.loc[(users['location_country'] == 'null') & (users['location_state'].str.contains('paca')), 'location_country'] = 'france'
    # uruguay
    users.loc[(users['location_country'] == 'null') & (users['location_state'].str.contains('montevideo')), 'location_country'] = 'uruguay'
    # argentina
    users.loc[(users['location_country'] == 'null') & (users['location_state'].str.contains('buenosaires')), 'location_country'] = 'argentina'
    # peru
    users.loc[(users['location_country'] == 'null') & (users['location_state'].str.contains('southamerica')), 'location_country'] = 'peru'
    # chile
    users.loc[(users['location_country'] == 'null') & (users['location_state'].str.contains('santiago')), 'location_country'] = 'chile'
    # japan
    users.loc[(users['location_country'] == 'null') & (users['location_state'].str.contains('iwakuni')), 'location_country'] = 'japan'
    users.loc[(users['location_country'] == 'null') & (users['location_state'].str.contains('tokyo')), 'location_country'] = 'japan'



    '''
    location_city
    '''
    # usa
    usa_city_repl = [
        'losang', 'seattle', 'sanf', 'sand', 'newyork', 'newark', 'newbedford', 'portland', 'cincinnati',
        'houston', 'albuquerque', 'chicago', 'austin', 'beaverton', 'raleigh', 'richmond', 'fairbanks',
        'minneapolis', 'stlouis', 'tucson', 'oakland', 'boston', 'kansascity', 'denver', 'springfield',
        'topeka', 'dallas', 'asheville', 'buffalo', 'fremont', 'stpaul', 'elcajon', 'miami', 'marysville',
        'baltimore', 'charleston', 'santamonica', 'knoxville', 'rochester', 'orlando', 'coloradosprings',
        'arlington', 'pensacola', 'sanjose', 'cedarrapids', 'olympia', 'lasvegas', 'mercerisland',
        'encinitas', 'omaha', 'lawrence', 'sacramento', 'norfolk', 'kirkwood', 'tallahassee', 'lexington',
        'kalamazoo', 'orleans', 'desmoines', 'aurora', 'annarbor', 'newbern', 'somerville', 'lakeland',
        'hartford', 'tigard', 'phoenix', 'irvine', 'sanantonio', 'mesa', 'brooklyn', 'philadelphia',
        'lacey', 'greenbay', 'pittsburg', 'wichita', 'elizabeth', 'murrieta', 'batonrouge', 'yuma',
        'baycity', 'lynchburg', 'santabarbara', 'statenisland', 'saintpaul', 'lakewood', 'fallschurch',
        'northhaven', 'frederick', 'milwaukie', 'cary', 'stcharles', 'lewiston', 'virginiabeach',
        'longbranch', 'indianapolis', 'portales', 'fountainvalley', 'sebastopol', 'washington', 'louisville',
        'millersburg'
    ]
    for keyword in usa_city_repl:
        users.loc[(users['location_country'] == 'null') & (users['location_city'].str.contains(keyword)), 'location_country'] = 'usa'
    # canada
    canada_city_repl = [
        'calgary', 'vancouver', 'toronto', 'ottawa', 'fredericton', 'victoria', 'hamilton', 'montreal',
        'kelowna', 'winnipeg', 'saskatoon', 'halifax', 'edmonton', 'kitchener', 'regina', 'lethbridge',

    ]
    for keyword in canada_city_repl:
        users.loc[(users['location_country'] == 'null') & (users['location_city'].str.contains(keyword)), 'location_country'] = 'canada'
    # italy
    users.loc[(users['location_country'] == 'null') & (users['location_city'].str.contains('milano')), 'location_country'] = 'italy'
    users.loc[(users['location_country'] == 'null') & (users['location_city'].str.contains('roma')), 'location_country'] = 'italy'
    users.loc[(users['location_country'] == 'null') & (users['location_city'].str.contains('rome')), 'location_country'] = 'italy'
    users.loc[(users['location_country'] == 'null') & (users['location_city'].str.contains('genova')), 'location_country'] = 'italy'
    users.loc[(users['location_country'] == 'null') & (users['location_city'].str.contains('torino')), 'location_country'] = 'italy'
    users.loc[(users['location_country'] == 'null') & (users['location_city'].str.contains('perugia')), 'location_country'] = 'italy'
    users.loc[(users['location_country'] == 'null') & (users['location_city'].str.contains('salerno')), 'location_country'] = 'italy'
    users.loc[(users['location_country'] == 'null') & (users['location_city'].str.contains('firenze')), 'location_country'] = 'italy'
    # united kingdom
    users.loc[(users['location_country'] == 'null') & (users['location_city'].str.contains('london')), 'location_country'] = 'united kingdom'
    users.loc[(users['location_country'] == 'null') & (users['location_city'].str.contains('manchester')), 'location_country'] = 'united kingdom'
    users.loc[(users['location_country'] == 'null') & (users['location_city'].str.contains('cambridge')), 'location_country'] = 'united kingdom'
    users.loc[(users['location_country'] == 'null') & (users['location_city'].str.contains('york')), 'location_country'] = 'united kingdom'
    users.loc[(users['location_country'] == 'null') & (users['location_city'].str.contains('birmingham')), 'location_country'] = 'united kingdom'
    users.loc[(users['location_country'] == 'null') & (users['location_city'].str.contains('edinburgh')), 'location_country'] = 'united kingdom'
    users.loc[(users['location_country'] == 'null') & (users['location_city'].str.contains('newcastle')), 'location_country'] = 'united kingdom'
    # germany
    users.loc[(users['location_country'] == 'null') & (users['location_city'].str.contains('hamburg')), 'location_country'] = 'germany'
    users.loc[(users['location_country'] == 'null') & (users['location_city'].str.contains('berlin')), 'location_country'] = 'germany'
    users.loc[(users['location_country'] == 'null') & (users['location_city'].str.contains('augsburg')), 'location_country'] = 'germany'
    # france
    users.loc[(users['location_country'] == 'null') & (users['location_city'].str.contains('paris')), 'location_country'] = 'france'
    # spain
    users.loc[(users['location_country'] == 'null') & (users['location_city'].str.contains('barcelona')), 'location_country'] = 'spain'
    # finland
    users.loc[(users['location_country'] == 'null') & (users['location_city'].str.contains('helsinki')), 'location_country'] = 'finland'
    # australia
    users.loc[(users['location_country'] == 'null') & (users['location_city'].str.contains('melbourne')), 'location_country'] = 'australia'
    users.loc[(users['location_country'] == 'null') & (users['location_city'].str.contains('sidney')), 'location_country'] = 'australia'
    users.loc[(users['location_country'] == 'null') & (users['location_city'].str.contains('canberra')), 'location_country'] = 'australia'
    # singapore
    users.loc[(users['location_country'] == 'null') & (users['location_city'].str.contains('singapore')), 'location_country'] = 'singapore'

    '''
    모든 null 값을 다 볼 순 없으므로,
    user_id 별 데이터 많은 순서대로 null 값 처리
    '''
    # usa
    usa_uids = [
        83671, 179718, 187065, 104278, 146230, 93565, 67663, 84795, 175100,
        273190, 51350, 19493, 226745, 57620, 125031, 113663, 178201, 91631,
        83443, 239535, 135228, 23680, 259264, 209229, 929, 168036, 50129,
        129368, 136465, 8937, 84523, 241749, 48743, 132188, 270897, 171045,
        44842, 115473, 1131, 91017, 68768, 167587, 135411, 30889, 221557,
        39195, 154346, 273110, 29497, 223816, 38718, 175529, 186238, 239449,
        141543, 77676, 258277, 240113, 172486, 34988, 112818, 129474, 46295,
        142041, 268035, 176102, 126985, 93386, 114601, 30650, 24105, 170850,
        28372, 207651, 122802, 129389, 266764, 269140, 50504, 52993, 170208,
        162264, 45641, 226556, 241214

    ]
    for uid in usa_uids:
        users.loc[users['user_id'] == uid, 'location_country'] = 'usa'    
    # uk
    users.loc[users['user_id'] == 178522, 'location_country'] = 'united kingdom'
    users.loc[users['user_id'] == 5476, 'location_country'] = 'united kingdom'
    users.loc[users['user_id'] == 237064, 'location_country'] = 'united kingdom'
    users.loc[users['user_id'] == 241537, 'location_country'] = 'united kingdom'
    # ireland
    users.loc[users['user_id'] == 26432, 'location_country'] = 'ireland'
    # canada
    canada_uids = [44089, 79188, 176100, 34087, 172962, 103160, 206693]
    for uid in canada_uids:
        users.loc[users['user_id'] == uid, 'location_country'] = 'canada'
    # france
    users.loc[users['user_id'] == 179641, 'location_country'] = 'france'
    # germany
    users.loc[users['user_id'] == 276538, 'location_country'] = 'germany'
    users.loc[users['user_id'] == 102169, 'location_country'] = 'germany'
    # austria
    users.loc[users['user_id'] == 3923, 'location_country'] = 'austria'
    users.loc[users['user_id'] == 14393, 'location_country'] = 'austria'
    # portugal
    users.loc[users['user_id'] == 164581, 'location_country'] = 'portugal'
    # australia
    users.loc[users['user_id'] == 11399, 'location_country'] = 'australia'
    # malaysia
    users.loc[users['user_id'] == 30445, 'location_country'] = 'malaysia'
    users.loc[users['user_id'] == 28543, 'location_country'] = 'malaysia'
    # philippines
    users.loc[users['user_id'] == 131023, 'location_country'] = 'philippines'
    #########################

    #########################
    users = users.drop(['location_city', 'location_state'], axis=1)
    #########################

    ratings = pd.concat([ratings1, ratings2]).reset_index(drop=True)

    #출판사
    publisher_dict=(books['publisher'].value_counts()).to_dict()
    publisher_count_df= pd.DataFrame(list(publisher_dict.items()),columns = ['publisher','count'])
    publisher_count_df = publisher_count_df.sort_values(by=['count'], ascending = False)
    modify_list = publisher_count_df[publisher_count_df['count']>1].publisher.values
    for publisher in modify_list:
        try:
            number = books[books['publisher']==publisher]['isbn'].apply(lambda x: x[:4]).value_counts().index[0]
            right_publisher = books[books['isbn'].apply(lambda x: x[:4])==number]['publisher'].value_counts().index[0]
            books.loc[books[books['isbn'].apply(lambda x: x[:4])==number].index,'publisher'] = right_publisher
        except: 
            pass
    
    #카테고리
    books.loc[books[books['category'].notnull()].index, 'category'] = books[books['category'].notnull()]['category'].apply(lambda x: re.sub('[\W_]+',' ',x).strip())
    category_df = pd.DataFrame(books['category'].value_counts()).reset_index()
    category_df.columns = ['category','count']  
    books['category_high'] = books['category'].copy()
    books.loc[books[books['category']=='biography'].index, 'category_high'] = 'biography autobiography'
    books.loc[books[books['category']=='autobiography'].index,'category_high'] = 'biography autobiography'
    books.loc[books[books['category'].str.contains('history',na=False)].index,'category_high'] = 'history'
    categories = ['garden','crafts','physics','adventure','music','fiction','nonfiction','science','science fiction','social','homicide',
    'sociology','disease','religion','christian','philosophy','psycholog','mathemat','agricult','environmental',
    'business','poetry','drama','literary','travel','motion picture','children','cook','literature','electronic',
    'humor','animal','bird','photograph','computer','house','ecology','family','architect','camp','criminal','language','india']
    for category in categories:
        books.loc[books[books['category'].str.contains(category,na=False)].index,'category_high'] = category
    category_high_df = pd.DataFrame(books['category_high'].value_counts()).reset_index()
    category_high_df.columns = ['category','count']
    # 5개 이하인 항목은 others로 묶어주도록 하겠습니다.
    others_list = category_high_df[category_high_df['count']<5]['category'].values
    books.loc[books[books['category_high'].isin(others_list)].index, 'category_high']='others'
    # # del books['category']
    # # books.rename(columns = {'category_high':'category'},inplace=True)

    

    # 인덱싱 처리된 데이터 조인
    # isbn,book_title,book_author,year_of_publication,publisher,img_url,language,category,summary,img_path
    context_df = ratings.merge(users, on='user_id', how='left').merge(books[['isbn', 'category', 'language', 'book_title']], on='isbn', how='left')
    train_df = ratings1.merge(users, on='user_id', how='left').merge(books[['isbn', 'category','language', 'book_title']], on='isbn', how='left')
    test_df = ratings2.merge(users, on='user_id', how='left').merge(books[['isbn', 'category','language', 'book_title']], on='isbn', how='left')


    ######################### age 변수 category 별 평균으로 전처리
    # train_df
    cat_age_dict = train_df.groupby('category').mean()['age'].to_dict()
    
    def fill_age(cat):
        return cat_age_dict[cat]

    train_df.loc[(train_df['age'].isna()) & (train_df['category'].notna()), 'age'] =\
        train_df.loc[(train_df['age'].isna()) & (train_df['category'].notna()), 'category'].apply(fill_age)

    # test_df
    cat_age_dict = test_df.groupby('category').mean()['age'].to_dict()
    
    def fill_age(cat):
        return cat_age_dict[cat]

    test_df.loc[(test_df['age'].isna()) & (test_df['category'].notna()), 'age'] =\
        test_df.loc[(test_df['age'].isna()) & (test_df['category'].notna()), 'category'].apply(fill_age)
    

    ######################### data 수가 n개 이하인 나라 추출
    cnty_n = 1
    cnty_val_cnt = pd.DataFrame(train_df['location_country'].value_counts())
    countries_to_empty_set = set(cnty_val_cnt.loc[cnty_val_cnt['location_country'] <= cnty_n].index)

    #########################
    # 해당 나라들 empty로 때려박음
    '''
    def one_frq_cnty_to_ety(cnty):
        if cnty in countries_to_empty_set:
            return 'empty'
        return cnty
    
    train_df['location_country'] = train_df['location_country'].apply(one_frq_cnty_to_ety)
    test_df['location_country'] = test_df['location_country'].apply(one_frq_cnty_to_ety)
    test_df.loc[test_df['location_country'].isna(), 'location_country'] = 'empty'
    '''
    # 해당 나라들 데이터에서 제거
    def sparse_frq_cnty_to_zero(cnty):
        if cnty in countries_to_empty_set:
            return 0
        return 1

    train_df['drop_zeros'] = train_df['location_country'].apply(sparse_frq_cnty_to_zero)
    train_df = train_df.loc[train_df['drop_zeros'] == 1].drop(['drop_zeros'], axis=1)
    #########################

    ######################### language 변수 user_id 별로 채우기
    # train_df
    lang_by_uid = pd.DataFrame(train_df.groupby('user_id')['language'].agg(pd.Series.mode))
    lang_by_uid = lang_by_uid.reset_index()
    lang_by_uid.columns = ['user_id', 'lang_by_uid']
    train_df = train_df.merge(lang_by_uid, on='user_id', how='left')

    def fill_lang(lang):
        if isinstance(lang, str):
            return lang
        elif not len(lang):
            return 'en'
        else:
            return lang[0]

    train_df.loc[train_df['language'].isna(), 'language'] =\
        train_df.loc[train_df['language'].isna(), 'lang_by_uid'].apply(fill_lang)
    train_df = train_df.drop(['lang_by_uid'], axis=1)

    # test_df
    lang_by_uid = pd.DataFrame(test_df.groupby('user_id')['language'].agg(pd.Series.mode))
    lang_by_uid = lang_by_uid.reset_index()
    lang_by_uid.columns = ['user_id', 'lang_by_uid']
    test_df = test_df.merge(lang_by_uid, on='user_id', how='left')

    def fill_lang(lang):
        if isinstance(lang, str):
            return lang
        elif not len(lang):
            return 'en'
        else:
            return lang[0]

    test_df.loc[test_df['language'].isna(), 'language'] =\
        test_df.loc[test_df['language'].isna(), 'lang_by_uid'].apply(fill_lang)
    test_df = test_df.drop(['lang_by_uid'], axis=1)
    #########################

    # sparse language 데이터에서 제거
    lang_cnt_dict = train_df.language.value_counts().to_dict()
    lang_n = 1

    def elim_sparse_lang(lang):
        if lang_cnt_dict[lang] <= lang_n:
            return 0
        return 1

    train_df['elim_lang'] = train_df['language'].apply(elim_sparse_lang)
    train_df = train_df.loc[train_df['elim_lang'] == 1].drop(['elim_lang'], axis=1)
    #########################

    # 인덱싱 처리
    # loc_city2idx = {v:k for k,v in enumerate(context_df['location_city'].unique())}
    # loc_state2idx = {v:k for k,v in enumerate(context_df['location_state'].unique())}
    loc_country2idx = {v:k for k,v in enumerate(context_df['location_country'].unique())}
    # train_df['location_city'] = train_df['location_city'].map(loc_city2idx)
    # train_df['location_state'] = train_df['location_state'].map(loc_state2idx)
    train_df['location_country'] = train_df['location_country'].map(loc_country2idx)
    # test_df['location_city'] = test_df['location_city'].map(loc_city2idx)
    # test_df['location_state'] = test_df['location_state'].map(loc_state2idx)
    test_df['location_country'] = test_df['location_country'].map(loc_country2idx)
    train_df['age'] = train_df['age'].fillna(int(train_df['age'].mean()))
    train_df['age'] = train_df['age'].apply(age_map)
    test_df['age'] = test_df['age'].fillna(int(test_df['age'].mean()))
    test_df['age'] = test_df['age'].apply(age_map)
    # book 파트 인덱싱
    category2idx = {v:k for k,v in enumerate(context_df['category'].unique())}
    # publisher2idx = {v:k for k,v in enumerate(context_df['publisher'].unique())}
    language2idx = {v:k for k,v in enumerate(context_df['language'].unique())}
    author2idx = {v:k for k,v in enumerate(context_df['book_title'].unique())}
    train_df['category'] = train_df['category'].map(category2idx)
    # train_df['publisher'] = train_df['publisher'].map(publisher2idx)
    train_df['language'] = train_df['language'].map(language2idx)
    train_df['book_title'] = train_df['book_title'].map(author2idx)
    test_df['category'] = test_df['category'].map(category2idx)
    # test_df['publisher'] = test_df['publisher'].map(publisher2idx)
    test_df['language'] = test_df['language'].map(language2idx)
    test_df['book_title'] = test_df['book_title'].map(author2idx)
    idx = {
        # "loc_city2idx":loc_city2idx,
        # "loc_state2idx":loc_state2idx,
        "loc_country2idx":loc_country2idx,
        "category2idx":category2idx,
        # "publisher2idx":publisher2idx,
        "language2idx":language2idx,
        "author2idx":author2idx,
    }
    return idx, train_df, test_df
	


def context_data_load(args):

    ######################## DATA LOAD
    users = pd.read_csv(args.DATA_PATH + 'users.csv')
    books = pd.read_csv(args.DATA_PATH + 'books.csv')
    train = pd.read_csv(args.DATA_PATH + 'train_ratings.csv')
    test = pd.read_csv(args.DATA_PATH + 'test_ratings.csv')
    sub = pd.read_csv(args.DATA_PATH + 'sample_submission.csv')

    ids = pd.concat([train['user_id'], sub['user_id']]).unique()
    isbns = pd.concat([train['isbn'], sub['isbn']]).unique()

    idx2user = {idx:id for idx, id in enumerate(ids)}
    idx2isbn = {idx:isbn for idx, isbn in enumerate(isbns)}

    user2idx = {id:idx for idx, id in idx2user.items()}
    isbn2idx = {isbn:idx for idx, isbn in idx2isbn.items()}

    train['user_id'] = train['user_id'].map(user2idx)
    sub['user_id'] = sub['user_id'].map(user2idx)
    test['user_id'] = test['user_id'].map(user2idx)
    users['user_id'] = users['user_id'].map(user2idx)

    train['isbn'] = train['isbn'].map(isbn2idx)
    sub['isbn'] = sub['isbn'].map(isbn2idx)
    test['isbn'] = test['isbn'].map(isbn2idx)
    books['isbn'] = books['isbn'].map(isbn2idx)

    idx, context_train, context_test = process_context_data(users, books, train, test)
    '''
    field_dims = np.array([len(user2idx), len(isbn2idx),
                            6, len(idx['loc_city2idx']), len(idx['loc_state2idx']), len(idx['loc_country2idx']),
                            len(idx['category2idx']), len(idx['publisher2idx']), len(idx['language2idx']), len(idx['author2idx'])], dtype=np.uint32)
    '''
    field_dims = np.array([len(user2idx), len(isbn2idx),
                            6, len(idx['loc_country2idx']),
                            len(idx['category2idx']), len(idx['language2idx']), len(idx['author2idx'])], dtype=np.uint32)

    data = {
            'train':context_train,
            'test':context_test.drop(['rating'], axis=1),
            'field_dims':field_dims,
            'users':users,
            'books':books,
            'sub':sub,
            'idx2user':idx2user,
            'idx2isbn':idx2isbn,
            'user2idx':user2idx,
            'isbn2idx':isbn2idx,
            }


    return data
	


def context_data_load(args):

    ######################## DATA LOAD
    users = pd.read_csv(args.DATA_PATH + 'users.csv')
    books = pd.read_csv(args.DATA_PATH + 'books.csv')
    train = pd.read_csv(args.DATA_PATH + 'train_ratings.csv')
    test = pd.read_csv(args.DATA_PATH + 'test_ratings.csv')
    sub = pd.read_csv(args.DATA_PATH + 'sample_submission.csv')

    ids = pd.concat([train['user_id'], sub['user_id']]).unique()
    isbns = pd.concat([train['isbn'], sub['isbn']]).unique()

    idx2user = {idx:id for idx, id in enumerate(ids)}
    idx2isbn = {idx:isbn for idx, isbn in enumerate(isbns)}

    user2idx = {id:idx for idx, id in idx2user.items()}
    isbn2idx = {isbn:idx for idx, isbn in idx2isbn.items()}

    train['user_id'] = train['user_id'].map(user2idx)
    sub['user_id'] = sub['user_id'].map(user2idx)
    test['user_id'] = test['user_id'].map(user2idx)
    users['user_id'] = users['user_id'].map(user2idx)

    train['isbn'] = train['isbn'].map(isbn2idx)
    sub['isbn'] = sub['isbn'].map(isbn2idx)
    test['isbn'] = test['isbn'].map(isbn2idx)
    books['isbn'] = books['isbn'].map(isbn2idx)

    idx, context_train, context_test = process_context_data(users, books, train, test)
    '''
    field_dims = np.array([len(user2idx), len(isbn2idx),
                            6, len(idx['loc_city2idx']), len(idx['loc_state2idx']), len(idx['loc_country2idx']),
                            len(idx['category2idx']), len(idx['publisher2idx']), len(idx['language2idx']), len(idx['author2idx'])], dtype=np.uint32)
    '''
    field_dims = np.array([len(user2idx), len(isbn2idx),
                            6, len(idx['loc_country2idx']),
                            len(idx['category2idx']), len(idx['language2idx']), len(idx['author2idx'])], dtype=np.uint32)

    data = {
            'train':context_train,
            'test':context_test.drop(['rating'], axis=1),
            'field_dims':field_dims,
            'users':users,
            'books':books,
            'sub':sub,
            'idx2user':idx2user,
            'idx2isbn':idx2isbn,
            'user2idx':user2idx,
            'isbn2idx':isbn2idx,
            }


    return data
    
def context_data_split(args, data):
    if args.SPLIT_OPT == 'tts':
        X_train, X_valid, y_train, y_valid = train_test_split(
                                                            data['train'].drop(['rating'], axis=1),
                                                            data['train']['rating'],
                                                            test_size=args.TEST_SIZE,
                                                            random_state=args.SEED,
                                                            shuffle=True,
                                                            ##### stratify
                                                            # stratify=data['train']['rating']
                                                            #####
                                                            )
        data['X_train'], data['X_valid'], data['y_train'], data['y_valid'] = X_train, X_valid, y_train, y_valid


    ######################### StratifiedKFold
    elif args.SPLIT_OPT == 'skf':
        xt = 'X_train'; xv = 'X_valid'
        yt = 'y_train'; yv = 'y_valid'        
        n_splits = 5

        skf = StratifiedKFold(n_splits=n_splits, random_state=args.SEED, shuffle=True)
        for i, train_idx, test_idx in enumerate(skf.split(data['train'].drop(['rating'], axis=1), data['train']['rating'])):
            X_train, X_valid =\
                data['train'].drop(['rating'], axis=1)[train_idx], data['train'].drop(['rating'], axis=1)[test_idx]
            y_train, y_valid =\
                data['train']['rating'][train_idx], data['train']['rating'][test_idx]

            data[xt+i], data[xv+i], data[yt+i], data[yv+i] = X_train, X_valid, y_train, y_valid

        data['n_splits'] = n_splits
    #########################

    return data


# def context_data_split(args, data):
#     count=data['train'].groupby("user_id").size()
#     dfcount = pd.DataFrame(count, columns=["count"])
#     data['train']=data['train'].merge(dfcount,on="user_id")
#     data['train']=data['train'][1::]
#     alrtrain = data['train'][data['train']["count"]!=1].drop(['count'],axis=1)
#     newtrain1 = data['train'][data['train']["count"]==1].drop(['count'],axis=1)

#     alr_train, alr_valid, alry_train, alry_valid = train_test_split(
#                                                     alrtrain.drop(['rating'], axis=1),
#                                                     alrtrain['rating'],
#                                                     test_size = 0.11,
#                                                     random_state=42, # args.SEED
#                                                     shuffle=True
#                                                     )


#     data['X_train'], data['y_train'],  = alr_train, alry_train
#     data['X_valid'] = pd.concat([newtrain1.drop(['rating'], axis=1),alr_valid])
#     data['y_valid'] = pd.concat([newtrain1['rating'],alry_valid])
#     return data

def context_data_loader(args, data):
    train_dataset = TensorDataset(torch.LongTensor(data['X_train'].values), torch.LongTensor(data['y_train'].values))
    valid_dataset = TensorDataset(torch.LongTensor(data['X_valid'].values), torch.LongTensor(data['y_valid'].values))
    test_dataset = TensorDataset(torch.LongTensor(data['test'].values))

    train_dataloader = DataLoader(train_dataset, batch_size=args.BATCH_SIZE, shuffle=args.DATA_SHUFFLE)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.BATCH_SIZE, shuffle=args.DATA_SHUFFLE)
    test_dataloader = DataLoader(test_dataset, batch_size=args.BATCH_SIZE, shuffle=False)

    data['train_dataloader'], data['valid_dataloader'], data['test_dataloader'] = train_dataloader, valid_dataloader, test_dataloader

    return train_dataset, valid_dataset, data
    # return data
