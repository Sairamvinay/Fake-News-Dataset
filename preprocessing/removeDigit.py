import pandas as pd
import re

def removeDigit(mydf):
    size = mydf.shape[0]
    output =  pd.DataFrame(columns = ['title', 'text', 'label']) 
    for i in range(size):
        tmp_title = re.sub('\d', '', mydf.loc[i]['title'])
        tmp_text = re.sub('\d', '', mydf.loc[i]['text'])
        output.loc[i] = [tmp_title] + [tmp_text] + [mydf.loc[i]['label']]
    return output
