import requests

# https://ieeexplore.ieee.org/Xplorehelp/ieee-xplore-training/user-tips
# The default in IEEE Xplore is to search metadata only, which includes abstracts, indexing terms,
# and bibliographic citation information (article titles, author names, publication titles, etc.).
# To do a full-text search, you go to the Advanced Search page and select the "Full text & Metadata" radio button.

##### Searching with wildcards
# IEEE Xplore automatically retrieves pluralized nouns, verb tenses, and British/American spelling
# variations for most words. For example, if you search "simulate", your search results will include any articles with
# "simulate", "simulates", "simulated", or "simulating". If you would like to search for all variations of the word,
# such as "simulation" and "simulator", you can use the asterisk wildcard. Simulat* will find any word that begins with
# "simulat". You can use up to five asterisk wildcards per search, and the asterisk wildcard can be used at the beginning
# of a word (*optic), in the middle of a word (me*n), or at the end of a word (simulat*).

#####  Boolean operators
# You can use Boolean operators (AND, OR, NOT) in the basic, advanced and command search options on IEEE Xplore.
# Proximity operators (NEAR, ONEAR) can also be used in basic and command search. Learn more about using search operators.
#
##### Proximity searching
# IEEE Xplore has two proximity operators: NEAR (for unordered proximity searches) and ONEAR (for ordered proximity searches).
# Proximity searching is supported in both basic and command search. The Command Search page can be accessed from the Other
# Search Options drop down menu below the basic search bar.
# Example: implantable NEAR/3 cardiac Finds articles with the word implantable within three words of cardiac;
# cardiac can come before or after implantable.
# Example: implantable ONEAR/3 cardiac Finds articles with the word implantable within three words of cardiac;
# but implantable must come before cardiac.

url="http://ieeexploreapi.ieee.org/api/v1/search/articles?"
key="&apikey=ymuyw5jx2brg4n9gabgfadbw&format=json&format=json&max_records=200&start_record=1&sort_order=asc&sort_field=article_number&end_year=end_date&start_year=start_date"
# key="&apikey=ymuyw5jx2brg4n9gabgfadbw&format=json&max_records=20&start_record=1&sort_order=asc&sort_field=article_number&end_year=2000&start_year=2013"

# querytext="querytext=(Metadata%20OR%20%22predict%20OR%20%22software%20OR%20%22defect%20OR%20%22qulity%22)&max_records=20&start_record=1&sort_order=asc&sort_field=article_number&end_year=2000&start_year=2013"
querytext="https://ieeexploreapi.ieee.org/api/v1/search/articles?querytext=(rfid%20NOT%20%22internet%20of%20things%22)"
response = requests.get(url+querytext+key)
json_response = response.json()
print(json_response)
print(url+querytext+key)

# querytext="querytext=( software OR software_program OR computer_software OR software_system OR software_package OR package ) AND ( defect OR shortcoming OR fault OR flaw OR blemish OR mar OR desert ) AND ( prediction OR anticipation OR prevision OR foretelling OR forecasting OR prognostication ) AND ( method OR method_acting )&end_year=2019&start_year=2020"
# querytext="querytext=(Metadata+software+OR+software_program+OR+computer_software+OR+software_system+OR+software_package+OR+package+%29+AND+%28+defect+OR+shortcoming+OR+fault+OR+flaw+OR+blemish+OR+mar+OR+desert+%29+AND+%28+prediction+OR+anticipation+OR+prevision+OR+foretelling+OR+forecasting+OR+prognostication+%29+AND+%28+method+OR+method_acting+%29"
#(software %20AND%20 (defect %20OR%20 fault) %20AND%20 (predict %20OR%20 prevision))
# search_string = "&abstract=defect&end_year=2000&start_year=2000"

# query = xplore.xploreapi.XPLORE('api_access_key')
# query.abstractText('query')
# data = query.callAPI()
# url = "http://ieeexploreapi.ieee.org/api/v1/search/articles?&apikey=ymuyw5jx2brg4n9gabgfadbw&format=json&max_records=10&start_record=1&sort_order=asc&sort_field=article_number"
# url="http://ieeexploreapi.ieee.org/api/v1/search/articles?"
# querytext="querytext=(rfid%20OR%20%22prediction%20software%20defect%22)"
# querytext="querytext=(abstract%20OR%20%22predict%20OR%20%22software%20OR%20%22defect%20OR%20%22qulity%22)"
    # (software %20AND%20 (defect %20OR%20 fault) %20AND%20 (predict %20OR%20 prevision))
# key="&apikey=ymuyw5jx2brg4n9gabgfadbw&format=json&max_records=20&start_record=1&sort_order=asc&sort_field=article_number&end_year=2000&start_year=2013"
# search_string = "&abstract=defect&end_year=2000&start_year=2000"
# response = requests.get(url+querytext+key)
# json_response = response.json()
# print(json_response)
# print(url+querytext+key)

# Software quality prediction using mixture models with EM algorithm

# https://ieeexploreapi.ieee.org/api/v1/search/articles?querytext=(rfid%20OR%20%22internet%20of%20things%22)&apikey=

# http://ieeexploreapi.ieee.org/api/v1/search/articles?apikey=ymuyw5jx2brg4n9gabgfadbw&format=xml&max_records=25&start_record=1&sort_order=asc&sort_field=article_number&abstract=software+fault+prediction&end_year=2000
# http://ieeexploreapi.ieee.org/api/v1/search/articles?apikey=ymuyw5jx2brg4n9gabgfadbw&format=json&max_records=25&start_record=1&sort_order=asc&sort_field=article_number&abstract=software+fault&end_year=2000&start_year=2000


# prediction OR
# software OR
# defect OR
# https://ieeexploreapi.ieee.org/api/v1/search/articles?querytext=(rfid%20AND%20%22internet%20of%20things%22)&apikey=apikey=ymuyw5jx2brg4n9gabgfadbw&format=json&max_records=10&start_record=1&sort_order=asc&sort_field=article_number&end_year=2000&start_year=2013
# http: //ieeexploreapi.ieee.org/api/v1/search/articles?querytext=(abstract%20OR%20%22predict%20OR%20%22software%20OR%20%22defect%20OR%20%22qulity%22)&apikey=ymuyw5jx2brg4n9gabgfadbw&format=json&max_records=7000&start_record=1&sort_order=asc&sort_field=article_number&end_year=2000&start_year=2013
