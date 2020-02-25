#!/usr/bin/env python
# coding: utf-8

# # Python til tabeldata

# ## Dagens workshop
# 
# I dag arbejder vi videre med, hvordan python bruges til tabeldata. Vi skal blandt andet igennem, hvordan python bruges til simpel statistik og modeller, samt hvordan vi løser forskellige datahåndteringsproblemer.
# 
# ### Program
# - Genopfriskning fra del 1
# - Lineær regression med `scikit-learn`
# - Plots med `seaborn`
# - Håndtering af kategoriske data
# - Håndtering af missing
# - Kombinering af data

# ## Genopfriskning fra del 1

# In[53]:


#Indlæs data
import pandas as pd
ess = pd.read_csv('https://github.com/CALDISS-AAU/workshop_python-table-data/raw/master/data/ESS2014DK_subset.csv')


# In[54]:


#Inspicer første 5 rækker af pandas dataframe
ess.head()


# In[55]:


#Kolonner kan referes til ved navn (returnerer som en pandas serie)
ess['cgtsday'].head()


# In[56]:


#Vi kan inspicere typen med .dtypes (float64 = decimaltal)
ess['cgtsday'].dtypes


# In[57]:


#Dele af data kan specificeres med .loc (rækker, kolonner)
ess.loc[10:15, 'cgtsday'] #Returneres som serie


# In[58]:


#Flere kolonner specificeres ved at sætte dem ind i liste
ess.loc[10:15, ['happy', 'cgtsday']] #Returneres som data frame


# In[59]:


#Bestemte rækker specificeres ved at sætte kriterie(r)
ess.loc[ess['cgtsday'] == 10, ['cgtsday','happy']].head() #Returneres som data frame


# In[60]:


#Nye variable dannes ved at referere til variabel/kolonne, som endnu ikke er i datasætttet
ess['bmi'] = ess['weight'] * (ess['height']/100)**2


# ## ØVELSE 0: Opvarmning
# 
# 1. Indlæs ESS datasættet fra sidste gang 
# 
#     `pd.read_csv('https://github.com/CALDISS-AAU/workshop_python-intro/raw/master/data/ESS2014DK_subset.csv')`
#     
#     
# 2. Lav en aldersvariabel (datasættet er fra 2014)

# In[61]:


get_ipython().run_line_magic('matplotlib', 'inline')
#import seaborn as sns
import matplotlib.pyplot as plt

#sns.set()
#sns.lineplot(ess['height'], ess['weight'])

plt.scatter(ess['height'], ess['weight'])


# ## Modeller/estimater i python
# 
# Typisk arbejder vi i samfundsvidenskab med statistisker modeller til at producere estimater: Hvor meget bliver Y påvirket af X, og hvor sikre er vi på det estimat?
# 
# I Neighborhood AI beskæftiger os vi med machine learning, hvor nogen af de samme modeller, som I kender fra statistik også bliver brugt. Her er det dog anderledes, da vi ikke beskæftiger os med, hvor godt et estimat, X, kan forklare Y, men i stedet er interesseret i, hvor godt vi kan forudsige Y. 
# 
# Vi skifter altså fokus væk fra forklaringskraften af et enkelte estimater til at fokusere på hvor mange estimater, der skal til for at kunne forudsige Y.
# 

# I denne workshop arbejder vi med scikit learn, som er et pakkebibliotek udviklet specifikt til machine learning. 
# 
# Måden man specificerer og bruger modeller (fx lineær regression) er derfor anderledes, da man netop har dette andet sigte.

# ## Hvordan fordeler data sig?
# 
# Som altid er det godt at gøre sig bekendt med sine data.
# 
# Vi har tidligere set, hvordan vi kan producere forskellige deskriptive mål (`.describe()`, `.min()`, `.max()` osv.)
# 
# En anden måde at blive bekendt med data er gennem plotting. Vi ser her nærmere på `seaborn` pakken til at lave forskellige visualiseringer.
# 
# `seaborn` bygger oven på pakken `matplotlib`, hvorfor begge pakker bruges.

# In[62]:


import matplotlib.pyplot as plt #importerer pyplot for matplotlib - bruges til at sætte størrelse osv.
import seaborn as sns #seaborn importeres
get_ipython().run_line_magic('matplotlib', 'inline')
#indstilling for jupyter notebook: plots skal printes i notebook
sns.set() #indstillinger for seaborn (her standardindstillinger)


# In[63]:


data = sns.load_dataset("iris") #vi indlæser et test datasæt

data.head() #inspicerer første 5 rækker


# `seaborn` kan danne et gitter af scatterplots i variable i datasættet med `pairplot`. Dette kan give os et umiddelbart overblik.

# In[64]:


sns.pairplot(data, hue='species', height = 2.5)


# Lad os se nærmere på sammenhængen mellem "petal length" og "petal width"

# In[65]:


plt.figure(figsize=(15,10)) #sætter størrelsen på figuren
sns.regplot(x = 'petal_length', y = 'petal_width', data = data) #regplot - scatterplot med fitted regressionslinje


# Grafen indikerer, at for at kunne forudsige bredden af petals, skal vi vide nogen om længden (sjovt nok).
# 
# For at gøre det simpel prøver vi at modellere med lineær regression. Som sagt er tankegangen dog lidt anderledes, når vi taler machine learning. Vi bruger ikke modellen til at forklare, hvad der øger bredden af petals, men vi laver en model, som kan bruges til at forudsige bredden af petals.
# 
# **Lineær regression i python med scikit learn**
# - Specificer modellen: `model = LinearRegression`
# - Specificer X og Y: `X = pd.DataFrame(data['petal_length']), Y = data['petal_width']`
# - Fit modellen: `model.fit(X, Y)`
# - Print ønskede outputs: `print(model.coef_, model.score(X, Y)`

# Vi bruger pakken scikit-learn (`sklearn`) til at modellere den lineære regression. Dette for at give en forståelse for, hvordan modeller specificeres i denne pakke.

# Først importeres og specificeres modellen. Der laves altså en tom model først, som vi bagefter giver et vis input.

# In[66]:


from sklearn.linear_model import LinearRegression #Modellen importeres fra scikit
model = LinearRegression() #Modellen specficeres
model


# X og Y specificeres.
# 
# Bemærk, at X skal specificeres som dataframe (2 dimensioner), mens Y specificeres som serie (1 dimension). Modellen forventer 2 dimensioner af X, da der kan være flere X-værdier.
# 
# `.to_frame()` konverterer en serie til en dataframe.

# In[67]:


X = data['petal_length'].to_frame() #X specificeret som dataframe
Y = data['petal_width'] #Y specificeret som serie


# Modellen "fodres" nu med input. Modellen "fittes".

# In[68]:


model.fit(X, Y)


# Når modellen er fitted, kan vi bede om forskellige outputs. Fx koefficienterne, skæringspunktet og R<sup>2</sup>.

# In[69]:


print("Modellens skæringspunk:", model.intercept_)
print("Model hældning:        ", model.coef_[0])
print("R2                     ", model.score(X, Y))


# ## ØVELSE 1: Lineær modellering
# 
# 1. Indlæs iris datasættet med `sns.data_load("iris")` (sørg for at `seaborn` er importeret med alias `sns`)
# 2. Inspicer fordelingen af petal length og sepal length med `sns.regplot(X, Y, data)`
# 3. Lav lineær regression model med sepal length som X værdi og petal length som Y værdi
#     - Husk at X skal specificeres som dataframe og Y som serie
# 4. Hvad er hældningen? (`.coef_`)

# ## Lineær modellering med ESS
# 
# Lad os nu se nærmere på ESS dataen. Det vil være rimeligt at antage, at personers højde hænger sammen med deres vægt, så lad os kigge nærmere på dette:

# In[70]:


plt.figure(figsize=(15,10)) #sætter størrelsen på figuren
sns.regplot(x = 'height', y = 'weight', data = ess) #scatterplot af højde og vægt med fitted regressionslinje


# Det tyder på en lineær sammenhæng, så lad os prøve at modellere det.
# 
# ESS data har samme format som iris: dataframe med rækker og kolonner. Kolonnerne vi kigger på indeholder også numeriske værdier (kan tjekkes med `.dtypes`).
# 
# Modellen bør derfor umiddelbart kunne laves på samme måde:

# In[71]:


from sklearn.linear_model import LinearRegression #Modellen importeres fra scikit
model = LinearRegression() #Modellen specficeres

X = ess['height'].to_frame() #X som dataframe
Y = ess['weight'] #Y som serie

model.fit(X, Y)


# Det giver dog fejl - hvorfor?

# ## Behandling af missing
# 
# Scikit-learn har ikke indbyggede funktioner til håndtering af missing. For at kunne bruge modellerne i scikit-learn, skal missingværdier derfor håndteres først. Missing håndteres på en af to måder:
# 
# - Rækker med missing fjernes (listwise deletion)
# - Værdier "imputeres"; dvs. erstattes med en bestemt værdi.
# 
# I dag gennemgås blot hvordan missing fjernes eller erstattes med enkelte værdier.

# ### Fjern eller erstat missing
# 
# En pandas dataframe (og serie) har flere indbyggede metoder til at håndtere missing. De mest simple er:
# 
# - `.dropna()`: listwise deletion af observationer, som indeholder missing værdier
# - `.fillna()`: erstatter missing med en angiven værdi
# 
# Derudover findes også disse metoder, som særligt er egnet til tidsserie data:
# 
# - `ffill()`: erstatter missing værdi med værdien i næste række eller kolonne
# - `bfill()`: erstatter missing værdi med værdien i forrige række eller kolonne

# In[101]:


ess['cgtsday'].head() #oprindelig data


# In[102]:


ess['cgtsday'].dropna().head() #data med missing fjernet - læg mærke til rækkenummer


# In[104]:


ess['cgtsday'].fillna(0).head() #data med missing erstattet med 0


# Fill kan også bruges til at erstatte med middelværdi eller prædikterede værdier.

# In[105]:


cgts_fill = ess['cgtsday'].mean()
ess['cgtsday'].fillna(cgts_fill).head() #erstat missing med middelværdi


# In[106]:


cgts_fill_m = round(ess.loc[ess['gndr'] == 'Male', 'cgtsday'].mean())
cgts_fill_f = round(ess.loc[ess['gndr'] == 'Female', 'cgtsday'].mean())

def cgts_fill( gndr ):
    if gndr == 'Male':
        return(cgts_fill_m)
    elif gndr == 'Female':
        return(cgts_fill_f)

ess['cgtsday'].fillna(ess['gndr'].map(cgts_fill)).head() #erstat med middel for køn


# Bemærk at vi ved at kalde metoderne ikke ændrer på noget. Vi skal derfor overskrive data eller lave kopi af data. Som altid er det bedst at bevare det oprindelige datasæt. Samtidig giver det i arbejde med modeller som disse mening at tænke det sådan, at man arbejder hen mod en udgave af datasættet, som skal bruges til en bestemt model.

# ## FÆLLESØVELSE: Hvordan får vi lavet en model mellem højde og vægt?
# 
# Lad os prøve at se om vi kan løse problemet med højde og vægt...

# In[ ]:


ess_mdata = ess.dropna() #kopi af datasæt hvor missingværdier fjernes

X = ess_mdata['height'].to_frame()
Y = ess_mdata['weight']

model.fit(X, Y)
print("Modellens skæringspunk:", model.intercept_)
print("Model hældning:        ", model.coef_[0])
print("R2                     ", model.score(X, Y))


# ## Hvad med andre variabeltyper?
# 
# Vi har nu set på, hvordan vi modellerer med numeriske variable. Som det ses af datasættet, er der også andre typer variable.

# In[107]:


ess.head()


# Lad os for øvelsens skyld behandle "happy" som interval og se, hvordan variablen udvikler sig med alderen:

# In[108]:


plt.figure(figsize=(15,10)) #sætter størrelsen på figuren
sns.regplot(x = 'yrbrn', y = 'happy', data = ess) #scatterplot af højde og vægt med fitted regressionslinje


# Det giver fejl - hvorfor?

# ## Kategoriske varialbe i python
# 
# Lad os se nærmere på variablen `happy`. Tællinger for de enkelte værdier kan fås med `.value_counts()`, hvilket samtidig giver os overblik over værdierne.

# In[72]:


ess['happy'].value_counts()


# In[73]:


ess['happy'].dtypes


# Som det kan ses, indeholder variablen både numeriske og tekstuelle værdier. Typen er et "objekt", som hverken er tekst eller tal. 
# 
# Er ens variable indlæst som typen "objekt", bør man forholde sig til dem og ændrer dem på en af følgende måder:
# - Konverter til tal: `pd.to_numeric(data)`
# - Konverter til kategorisk: `.astype('category')`

# Lad os først prøve at konvertere happy til kateogrisk.

# In[74]:


ess['happycat'] = ess['happy'].astype('category')
ess['happycat'].value_counts()


# In[75]:


ess['happycat'].dtypes


# `value_counts()` giver præcis det samme output efter konvertering, men vi kan nu se, at typen er anderledes (`CategoricalDtype`). Bemærk at `ordered` angiver, hvorvidt variabel skal betragtes nominalt eller ordinalt.

# ### Kategorisk og numerisk i python
# 
# Når man arbejder med kategoriske data i python, skal man tage aktiv beslutning om, hvordan variablen skal behandles. I modsætning til andre statistikprogrammer, har kategorier i python (pandas) ikke en underliggende numerisk værdi.
# 
# Vi kan godt få de underliggende koder, men det er ikke nogen, som vi kan refere til.

# In[76]:


ess['happycat'].cat.codes.unique()


# In[77]:


any(ess['happycat'] == 10)


# Bemærk også at kategorier både kan kodes som tal og tekst. I dette tilfælde er kategorierne kodet som tekst:

# In[78]:


any(ess['happycat'] == 1)


# In[79]:


any(ess['happycat'] == '1')


# Det betyder, hvis man koder variable som kategoriske, skal de kun behandles som kategorisk - enten nominalt eller ordinalt.
# 
# Skal variablen behandles som interval, skal variablen derfor kodes om til numerisk.

# ### Nominalt og ordinale variable i python
# 
# Inden vi ser på den numeriske konvertering, så lad os først kigge på, hvordan vi adskiller mellem nominale og ordinale i python.
# 
# Begge variabeltyper kodes som kategorisk, men forskellen ligger i, om variablen kodes som "ordered" eller ej.
# - `ordered = False`: Nominal
# - `ordered = True`: Ordinal
# 
# Lad os som eksempel kigge på 'alcfreq' variablen:

# In[80]:


ess['alcfreq'].value_counts()


# In[81]:


ess['alcfreq'].dtypes


# Ligesom 'happy' er 'alcfreq' indlæst som objekt-type - Det skal vi gøre noget ved!
# 
# Den simple løsning er blot at tvinge den til kategorisk, men specificerer ikke nærmere, bliver den lavet til nominalt, hvilket ikke stemmer med variablen (der er tydeligvis rangorden i værdierne).

# In[82]:


ess['alcfreq'].astype('category') #denne kommando returnerer variablen som nominal


# Måden vi løser dette, er ved først at lave vores eget "kategorisæt" eller kategoritype, som vi derefter sætter på variablen.
# 
# På vores egen type kan vi specificere, at værdier skal behandles ordinalt (`ordered = True`). De sættes i rangorden efter den rækkefølge, som de skrives ind.

# In[83]:


from pandas.api.types import CategoricalDtype

alc_cats = CategoricalDtype(categories = ['Never', 'Less than once a month', 'Once a month', '2-3 times a month', 
                                          'Once a week', 'Several times a week', 'Every day'], ordered = True)

  
ess['alcfreq'] = ess['alcfreq'].astype(alc_cats)


# Vi kan teste det efter ved at lave en variabel for, hvorvidt man drikker ugentligt.

# In[84]:


ess['drink_weekly'] = ess['alcfreq'] >= 'Once a week'


# In[85]:


ess.loc[0:10, ['alcfreq', 'drink_weekly']]


# ## ØVELSE 2: Kategorisk variabel
# 
# 1. Undersøg variablen `health` - hvilken type variablen er det? (og forstår python variablen rigtigt?)
# 2. Konverter variablen `health` til den korrekte type kategorisk variabel
# 3. Undersøg enten med `.sort_values(variabel)` eller med logiske operatorer (< >) om variablen vender rigtigt
# 
# Lav kategorisk type:
# `kategori = CategoricalDtype(categories, ordered)`
# 
# Ændr til kategorisk:
# `variabel = variabel.astype(category)`
# 
# ### Bonus
# 
# - Prøv at lave en krydstabel med `pd.crosstab()` mellem health og alcfreq

# ## Fra kategorisk til numerisk
# 
# Vi har stadig ikke løst, hvordan vi kan bruge "happy" i modellen.
# 
# For at bruge "happy" i en regression, må vi betragte den som interval. Derfor skal den laves om til numerisk.
# 
# Da variablen har tekstværdier kan vi dog ikke bare tvinge den om:

# In[95]:


ess['happy_num'] = pd.to_numeric(ess['happy'])


# ### Rekodning af variable med pandas
# 
# Før at vi kan tvinge værdierne til numeriske, må vi først fortælle python, hvad tekstværdierne skal forstås som - altså rekode variablen.
# 
# Den mest ligetil måde at rekode er ved specificere en "dictionary".
# 
# En dictionary er en form for navngivet liste bestående af sæt af nøgler og værdier. En dictionary specificeres ved at bruge `{}`: `a_dict = {key: value, key: value}`.

# For at bruge en dictionary til rekodning, danner vi først en dicionary, hvor keys er de værdier, som skal erstattes, og values er de værdier, som de skal erstattes med:

# In[96]:


happy_replace = {'Extremely happy': 10, 'Extremely unhappy': 0}


# Vi kan derefter bruge dictionary til at erstatte værdier med metoden `.replace()`

# In[98]:


ess['happy_num'] = ess['happy'].replace(happy_replace)
ess['happy_num'].value_counts()


# Variablen består nu kun af numeriske værdier, og vi kan derfor tvinge den til at være numerisk:

# In[99]:


ess['happy_num'] = pd.to_numeric(ess['happy_num'])


# Vi kan nu modellere sammenhængen mellem alder og happy:

# In[100]:


sns.regplot(x = 'yrbrn', y = 'happy_num', data = ess, y_jitter = 0.5)


# Det tyder ikke på den helt store sammenhæng, men vi modellerer det alligevel!

# ## ØVELSE 3: Modellering af alder og happy
# 
# 1. Sørg for at happy er konverteret til numerisk
# 2. Lav en lineær regressionsmodel for alder og happy
# 
# ### Bonus
# 
# - Lav en lineær regressionsmodel, som indeholder flere X variable
