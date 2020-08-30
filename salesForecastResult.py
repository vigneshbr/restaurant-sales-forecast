from joblib import load
import numpy as np
import pandas as pd
import requests

#pd.options.display.max_columns = None
#pd.options.display.max_rows = None



def getWeatherDescSource_df():
    desc = ["broken clouds","drizzle","few clouds","fog","haze","heavy intensity rain",
            "light intensity drizzle","light rain","light thunderstorm","mist","moderate rain",
            "overcast clouds","proximity moderate rain","proximity thunderstorm","scattered clouds",
            "sky is clear","thunderstorm","thunderstorm with heavy rain","thunderstorm with light rain",
            "thunderstorm with rain", "clear sky"]

    score = ["weatherScore__0","weatherScore__1","weatherScore__2","weatherScore__3",
            "weatherScore__4","weatherScore__5","weatherScore__6","weatherScore__7",
            "weatherScore__8","weatherScore__9","weatherScore__10","weatherScore__11",
            "weatherScore__12","weatherScore__13","weatherScore__14","weatherScore__15",
            "weatherScore__16","weatherScore__17","weatherScore__18","weatherScore__19","weatherScore__15"]

    d = {'Desc':desc, 'mainScore': score}

    weatherDescScoreSourceDF = pd.DataFrame(data=d)
    return weatherDescScoreSourceDF

#########################################################################

def getMaxTemp_df(apiData_df):
    # Extracting Data for 'maxTemp' column
    tempMaxDF = apiData_df[['Date', 'temp_max_prediction']]
    tempMaxDFGroupMean = tempMaxDF.groupby(['Date'])['temp_max_prediction'].mean().reset_index().rename(columns={'Date':'Date','temp_max_prediction':'tempMax'})

    return tempMaxDFGroupMean

#########################################################################

def getWeatherDesc_df(apiData_df, weatherDescScoreSourceDF):
    # Extracting Data for 'weather Desc' column
    # Date frame for weather description
    weatherDescDFRaw = apiData_df[['Date', 'main_weather_description_prediction']]
    weatherDescDF = weatherDescDFRaw.copy()

    weatherDescDF['main_weather_description_prediction'] = weatherDescDF['main_weather_description_prediction'].astype('category')
    weatherDescDF['main_weather_description_prediction'] = weatherDescDF['main_weather_description_prediction'].cat.codes


    # With One-Hot Encoding the weather desc data get categorised to values 1 and 0
    weatherValues = weatherDescDFRaw['main_weather_description_prediction'].unique()
    cat2DF = pd.DataFrame(weatherDescDF['main_weather_description_prediction'].values, columns=['Date'])
    oheWeatherDF = pd.get_dummies(cat2DF, columns=['Date'], prefix=['weatherScore_'])
    cat2DF = cat2DF.join(oheWeatherDF)

    cat2DFColumn = cat2DF.columns
    cat2dfValues = cat2DF.values[:,1:]
    cat2dfValues = np.hstack((np.atleast_2d(weatherDescDFRaw['Date'].values).T, cat2dfValues))
    oheWeatherScoreDF = pd.DataFrame(cat2dfValues, columns=cat2DFColumn)

    # comparing the weatherScore column names between the current dataframe and the original dataframe from trained model
    weatherValues.sort()
    oheWeatherDescScoreDF = pd.DataFrame({'Desc': weatherValues, 'oheDesc': cat2DF.columns[1:]})
    weatherDescUpdateScoreDF = oheWeatherDescScoreDF.merge(weatherDescScoreSourceDF, on='Desc', how='inner')

    # new Column array
    newColumn = weatherDescUpdateScoreDF['mainScore'].values
    newColumn = np.insert(newColumn, 0, ['Date'])

    # syncing the column names to the training data of Model
    oheWeatherScoreDF.columns = newColumn

    #for i in range(len(oheWeatherScoreDF.columns)-1):
     #   if weatherDescUpdateScoreDF['oheDesc'][i] == oheWeatherScoreDF.columns[i+1]:
      #      oheWeatherScoreDF = oheWeatherScoreDF.rename(columns={oheWeatherScoreDF.columns[i+1]: #weatherDescUpdateScoreDF['mainScore'][i]})

    # weather desc data for forcasted 5-days.
    oheWeatherColumns = oheWeatherScoreDF.columns[1:]
    oheWeatherDF = oheWeatherScoreDF.groupby(['Date'])[oheWeatherColumns].sum().reset_index()

    return oheWeatherDF

################################################################################

def getBase_df():
    # Dataset for new Sales forecast
    # creating a base dataframe

    baseColumn = ["Date", "year", "dayOfMonth__0", "dayOfMonth__1", "dayOfMonth__2", 
    "dayOfMonth__3", "dayOfMonth__4", "dayOfMonth__5", "dayOfMonth__6", "dayOfMonth__7", 
    "dayOfMonth__8", "dayOfMonth__9", "dayOfMonth__10", "dayOfMonth__11", "dayOfMonth__12", 
    "dayOfMonth__13", "dayOfMonth__14", "dayOfMonth__15", "dayOfMonth__16", "dayOfMonth__17", 
    "dayOfMonth__18", "dayOfMonth__19", "dayOfMonth__20", "dayOfMonth__21", "dayOfMonth__22", 
    "dayOfMonth__23", "dayOfMonth__24", "dayOfMonth__25", "dayOfMonth__26", "dayOfMonth__27", 
    "dayOfMonth__28", "dayOfMonth__29", "dayOfMonth__30", "dayOfWeek__0", "dayOfWeek__1", 
    "dayOfWeek__2", "dayOfWeek__3", "dayOfWeek__4", "dayOfWeek__5", "dayOfWeek__6", "monthOfYear__0", "monthOfYear__1", "monthOfYear__2", "monthOfYear__3", "monthOfYear__4", "monthOfYear__5", "monthOfYear__6", "monthOfYear__7", "monthOfYear__8", "monthOfYear__9", "monthOfYear__10", "monthOfYear__11", "feelsLikeMax", "weatherScore__0", "weatherScore__1", "weatherScore__2", "weatherScore__3", "weatherScore__4", "weatherScore__5", "weatherScore__6", "weatherScore__7", "weatherScore__8", "weatherScore__9", "weatherScore__10", "weatherScore__11", "weatherScore__12", "weatherScore__13", "weatherScore__14", "weatherScore__15", "weatherScore__16", "weatherScore__17", "weatherScore__18", "weatherScore__19", "currentHoliday"]

    baseData = [["15-08-2020", "2020", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1"],["16-08-2020", "2020", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0"],["17-08-2020", "2020", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0"],["18-08-2020", "2020", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0"],["19-08-2020", "2020", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0"],["20-08-2020", "2020", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0"],["21-08-2020", "2020", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0"],["22-08-2020", "2020", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1"],["23-08-2020", "2020", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0"],["24-08-2020", "2020", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0"],["25-08-2020", "2020", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0"],["26-08-2020", "2020", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0"],["27-08-2020", "2020", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0"],["28-08-2020", "2020", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0"],["29-08-2020", "2020", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0"],["30-08-2020", "2020", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0"],["31-08-2020", "2020", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0"],["01-09-2020", "2020", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0"],["02-09-2020", "2020", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0"],["03-09-2020", "2020", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0"],["04-09-2020", "2020", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0"],["05-09-2020", "2020", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0"],["06-09-2020", "2020", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0"],["07-09-2020", "2020", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0"],["08-09-2020", "2020", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0"],["09-09-2020", "2020", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0"],["10-09-2020", "2020", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0"],["11-09-2020", "2020", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0"],["12-09-2020", "2020", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0"],["13-09-2020", "2020", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0"],["14-09-2020", "2020", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0"],["15-09-2020", "2020", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0"],["16-09-2020", "2020", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0"],["17-09-2020", "2020", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0"],["18-09-2020", "2020", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0"],["19-09-2020", "2020", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0"],["20-09-2020", "2020", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0"],["21-09-2020", "2020", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0"],["22-09-2020", "2020", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0"],["23-09-2020", "2020", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0"],["24-09-2020", "2020", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0"],["25-09-2020", "2020", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0"],["26-09-2020", "2020", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0"],["27-09-2020", "2020", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0"],["28-09-2020", "2020", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0"],["29-09-2020", "2020", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0"],["30-09-2020", "2020", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0"],["01-10-2020", "2020", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0"],["02-10-2020", "2020", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1"],["03-10-2020", "2020", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0"],["04-10-2020", "2020", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0"],["05-10-2020", "2020", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0"],["06-10-2020", "2020", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0"],["07-10-2020", "2020", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0"],["08-10-2020", "2020", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0"],["09-10-2020", "2020", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0"],["10-10-2020", "2020", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0"],["11-10-2020", "2020", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0"],["12-10-2020", "2020", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0"],["13-10-2020", "2020", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0"],["14-10-2020", "2020", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0"],["15-10-2020", "2020", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0"],]

    predictBaseDF = pd.DataFrame(baseData, columns=baseColumn)
    predictBaseDF = predictBaseDF.drop('feelsLikeMax', axis=1)

    return predictBaseDF

################################################################################

def getPredictionReady_df(predictBaseDF, oheWeatherDF, tempMaxDFGroupMean):

    # Required Column
    requiredColumn = ['dayOfMonth__0', 'dayOfMonth__1', 'dayOfMonth__2', 'dayOfMonth__3', 'dayOfMonth__4', 'dayOfMonth__5', 'dayOfMonth__6', 'dayOfMonth__7', 'dayOfMonth__8', 'dayOfMonth__9', 'dayOfMonth__10', 'dayOfMonth__11', 'dayOfMonth__12', 'dayOfMonth__13', 'dayOfMonth__14', 'dayOfMonth__15', 'dayOfMonth__16', 'dayOfMonth__17', 'dayOfMonth__18', 'dayOfMonth__19', 'dayOfMonth__20', 'dayOfMonth__21', 'dayOfMonth__22', 'dayOfMonth__23','dayOfMonth__24', 'dayOfMonth__25', 'dayOfMonth__26', 'dayOfMonth__27','dayOfMonth__28', 'dayOfMonth__29', 'dayOfMonth__30', 'dayOfWeek__0','dayOfWeek__1', 'dayOfWeek__2', 'dayOfWeek__3', 'dayOfWeek__4', 'dayOfWeek__5', 'dayOfWeek__6', 'monthOfYear__0', 'monthOfYear__1','monthOfYear__2', 'monthOfYear__3', 'monthOfYear__4', 'monthOfYear__5','monthOfYear__6', 'monthOfYear__7', 'monthOfYear__8', 'monthOfYear__9','monthOfYear__10', 'monthOfYear__11', 'tempMax', 'weatherScore__0','weatherScore__1', 'weatherScore__3', 'weatherScore__4','weatherScore__5', 'weatherScore__6', 'weatherScore__7','weatherScore__8', 'weatherScore__9', 'weatherScore__10','weatherScore__11', 'weatherScore__12', 'weatherScore__13','weatherScore__14', 'weatherScore__15', 'weatherScore__16','weatherScore__17', 'weatherScore__18', 'weatherScore__19','currentHoliday']

    # Merging Base dataframe and tempMax dataframe
    newPredictDF1 = predictBaseDF.merge(tempMaxDFGroupMean, on='Date', how='inner')
    newPredictDF1

    # updating only the required weatherScore columns
    # other weatherScore columns will remain un-changed
    newPredictDF1.update(oheWeatherDF)

    # using only the required columns for new model prediction
    newPredictDF = newPredictDF1[requiredColumn]

    # converting the data-type of the columns
    colTypeConvert = ['dayOfMonth__0', 'dayOfMonth__1', 'dayOfMonth__2', 'dayOfMonth__3',
        'dayOfMonth__4', 'dayOfMonth__5', 'dayOfMonth__6', 'dayOfMonth__7',
        'dayOfMonth__8', 'dayOfMonth__9', 'dayOfMonth__10', 'dayOfMonth__11',
        'dayOfMonth__12', 'dayOfMonth__13', 'dayOfMonth__14', 'dayOfMonth__15',
        'dayOfMonth__16', 'dayOfMonth__17', 'dayOfMonth__18', 'dayOfMonth__19',
        'dayOfMonth__20', 'dayOfMonth__21', 'dayOfMonth__22', 'dayOfMonth__23',
        'dayOfMonth__24', 'dayOfMonth__25', 'dayOfMonth__26', 'dayOfMonth__27',
        'dayOfMonth__28', 'dayOfMonth__29', 'dayOfMonth__30', 'dayOfWeek__0',
        'dayOfWeek__1', 'dayOfWeek__2', 'dayOfWeek__3', 'dayOfWeek__4',
        'dayOfWeek__5', 'dayOfWeek__6', 'monthOfYear__0', 'monthOfYear__1',
        'monthOfYear__2', 'monthOfYear__3', 'monthOfYear__4', 'monthOfYear__5',
        'monthOfYear__6', 'monthOfYear__7', 'monthOfYear__8', 'monthOfYear__9',
        'monthOfYear__10', 'monthOfYear__11', 'weatherScore__0',
        'weatherScore__1', 'weatherScore__3', 'weatherScore__4',
        'weatherScore__5', 'weatherScore__6', 'weatherScore__7',
        'weatherScore__8', 'weatherScore__9', 'weatherScore__10',
        'weatherScore__11', 'weatherScore__12', 'weatherScore__13',
        'weatherScore__14', 'weatherScore__15', 'weatherScore__16',
        'weatherScore__17', 'weatherScore__18', 'weatherScore__19',
        'currentHoliday']

    newPredictDF[colTypeConvert] = newPredictDF[colTypeConvert].astype(str).astype(int)

    return newPredictDF, newPredictDF1[['Date']]

################################################################
# API Setup to Extract Weather Forecast

def get_apiData():
    # Forecast for 5-days
    lat = 12.9762
    longi = 77.6033

    coord_API_endpoint = "http://api.openweathermap.org/data/2.5/forecast?"

    lat_long = "lat=" + str(lat)+ "&lon=" + str(longi)
    join_key = "&appid=" + 'c93e15bb1da33b516dc4978f84c4ca22'
    units = "&units=metric"
    city_forecast = coord_API_endpoint + lat_long + join_key + units

    # extracted data in json
    forecast_json_data = requests.get(city_forecast).json()

    forecast_main = forecast_json_data['list'][0]['main']
    num_forecasts = len(forecast_json_data['list'])
    num_forecasts
    forecast_json_data['city']

    # JSON data to Dataframe

    df_predictions = pd.DataFrame()

    # Creating empty lists
    prediction_num = 0
    list_prediction_num = []
    date_time_prediction = []
    owm_city_id = []
    city_name = []
    latitude = []
    longitude = []
    country_name = []
    population = []
    timezone = [] # Shift in seconds from UTC
    sunrise = []
    sunset = []
    # Main
    temp_prediction = []
    temp_feels_like_prediction = []
    temp_min_prediction = []
    temp_max_prediction = []
    pressure_prediction = []
    sea_level_prediction = []
    grnd_level_prediction = []
    humidity_prediction = []
    temp_kf_prediction = []
    # Weather
    main_weather_prediction = []
    main_weather_description_prediction = []
    # Clouds
    clouds_prediction = []
    # Wind
    wind_speed_prediction = []
    wind_degree_prediction = []

    # Loop Through the JSON
    for num_forecasts in forecast_json_data['list']:
        df_predictions['prediction_num'] = prediction_num
        list_prediction_num.append(prediction_num)
        date_time_prediction.append(forecast_json_data['list'][prediction_num]['dt_txt'])
        
        owm_city_id.append(forecast_json_data['city']['id'])
        city_name.append(forecast_json_data['city']['name'])
        latitude.append(forecast_json_data['city']['coord']['lat'])
        longitude.append(forecast_json_data['city']['coord']['lon'])
        country_name.append(forecast_json_data['city']['country'])
        population.append(forecast_json_data['city']['population'])
        
        if forecast_json_data['city']['timezone'] >0 :
            timezone.append("+" + str((forecast_json_data['city']['timezone'])/3600))
        else:
            timezone.append((forecast_json_data['city']['timezone'])/3600)
            
        sunrise.append(forecast_json_data['city']['sunrise'])
        sunset.append(forecast_json_data['city']['sunset'])
        
        # Main
        temp_prediction.append(forecast_json_data['list'][prediction_num]['main']['temp'])
        temp_feels_like_prediction.append(forecast_json_data['list'][prediction_num]['main']['feels_like'])
        temp_min_prediction.append(forecast_json_data['list'][prediction_num]['main']['temp_min'])
        temp_max_prediction.append(forecast_json_data['list'][prediction_num]['main']['temp_max'])
        pressure_prediction.append(forecast_json_data['list'][prediction_num]['main']['pressure'])
        sea_level_prediction.append(forecast_json_data['list'][prediction_num]['main']['sea_level'])
        grnd_level_prediction.append(forecast_json_data['list'][prediction_num]['main']['grnd_level'])
        humidity_prediction.append(forecast_json_data['list'][prediction_num]['main']['humidity'])
        temp_kf_prediction.append(forecast_json_data['list'][prediction_num]['main']['temp_kf'])
        # Weather
        main_weather_prediction.append(forecast_json_data['list'][prediction_num]['weather'][0]['main'])
        main_weather_description_prediction.append(forecast_json_data['list'][prediction_num]['weather'][0]['description'])
        # Clouds
        clouds_prediction.append(forecast_json_data['list'][prediction_num]['clouds']['all'])
        # Wind
        wind_speed_prediction.append(forecast_json_data['list'][prediction_num]['wind']['speed'])
        wind_degree_prediction.append(forecast_json_data['list'][prediction_num]['wind']['deg'])
        
        prediction_num += 1

    # Put data into a dataframe
    df_predictions['prediction_num'] = list_prediction_num
    df_predictions['date_time_prediction'] = date_time_prediction
    df_predictions['owm_city_id'] = owm_city_id
    df_predictions['city_name'] = city_name
    df_predictions['latitude'] = latitude
    df_predictions['longitude'] = longitude
    df_predictions['country_name'] = country_name
    df_predictions['population'] = population
    df_predictions['timezone'] = timezone
    df_predictions['sunrise'] = sunrise
    df_predictions['sunset'] = sunset

        # Main
    df_predictions['temp_prediction'] = temp_prediction
    df_predictions['temp_feels_like_prediction'] = temp_feels_like_prediction
    df_predictions['temp_min_prediction'] = temp_min_prediction
    df_predictions['temp_max_prediction'] = temp_max_prediction
    df_predictions['pressure_prediction'] = pressure_prediction
    df_predictions['sea_level_prediction'] = sea_level_prediction
    df_predictions['grnd_level_prediction'] = grnd_level_prediction
    df_predictions['humidity_prediction'] = humidity_prediction
    df_predictions['temp_kf_prediction'] = temp_kf_prediction
        # Weather
    df_predictions['main_weather_prediction'] = main_weather_prediction
    df_predictions['main_weather_description_prediction'] = main_weather_description_prediction
        # Clouds
    df_predictions['clouds_prediction'] = clouds_prediction
        # Wind
    df_predictions['wind_speed_prediction'] = wind_speed_prediction
    df_predictions['wind_degree_prediction'] = wind_degree_prediction

        # Using only necessary columns
    apiData_df = df_predictions[['date_time_prediction', 'temp_max_prediction','main_weather_description_prediction']]


    # Extracting the correct Date format
    apiData_df['Date'] = ''

    for i in range(len(apiData_df)):
        date = apiData_df['date_time_prediction'][i][:10]
        date = date[-2:]+'-'+date[-5:-3]+'-'+date[:4]
        apiData_df['Date'][i] = date

    tempMaxDFGroupMean = getMaxTemp_df(apiData_df)

    weatherDescScoreSourceDF = getWeatherDescSource_df()
    tempMaxDFGroupMean = getMaxTemp_df(apiData_df)
    oheWeatherDF = getWeatherDesc_df(apiData_df, weatherDescScoreSourceDF)
    predictBaseDF = getBase_df()
    
    dfForPredict = getPredictionReady_df(predictBaseDF, oheWeatherDF, tempMaxDFGroupMean)
    newPredictDF = dfForPredict[0]
    newPredictDF1 = dfForPredict[1]

    # Normalisation the new dataframe - for sales forecast
    # Get column names first
    #names = newPredictDF.columns
    # Create the Scaler object
    #scaler = preprocessing.StandardScaler()
    # Fit your data on the scaler object
    #normalizedX = scaler.fit_transform(newPredictDF)
    #normalizedX = pd.DataFrame(normalizedX, columns=names)

    # Loading the trained model

    # model uses normalised data
    #trainedModel = load('salesForecast.joblib')

    # model - TOTAL Sales does not use normalized data
    trainedModel2 = load('salesForecast_v2.joblib')

    # using the model trained on non-normalized data will produce real number results
    yPredNew2 = trainedModel2.predict(newPredictDF)
    # forming npArray - joining date and forcasted values
    forecastValues = np.concatenate((np.atleast_2d(newPredictDF1['Date'].values).T, yPredNew2), axis=1)
    # created a dataframe of forcasted values and their respective date.
    forecastSalesDF = pd.DataFrame(forecastValues, columns=['Date', 'newForecast'])
    # changing the datatype
    forecastSalesDF['newForecast'] = forecastSalesDF['newForecast'].astype(str).astype(float)

    ###########################################################################
    # model - ITEM Sales does not use normalized data
    itemSalesModel = load('itemSalesModel_v0.joblib')

    # forecasting ITEM Sales
    yPredItem = itemSalesModel.predict(newPredictDF)

    # forming npArray - joining date and forcasted ITEM values
    itemForecastValues = np.concatenate((np.atleast_2d(newPredictDF1['Date'].values).T, yPredItem), axis=1)

    # Column names to be set for new forecasted values
    itemForecastColumns = ['Date', 'barjariOotaF', 'chaatF', 'hotBeveragesF', 'iceCreamF',
                        'idlyF', 'juiceF', 'masalaDosaF', 'southMealsF']


    # created a dataframe of forcasted ITEM values and their respective date.
    forecastItemSalesDF = pd.DataFrame(itemForecastValues, columns=itemForecastColumns)

    # changing the datatype of forecasted ITEM dataframe
    forecastItemSalesDF[itemForecastColumns[1:]] = forecastItemSalesDF[itemForecastColumns[1:]].astype(str).astype(float)

    # Merging ITEM and TotalSales result to one dataframe
    forecastResultDF = forecastSalesDF.merge(forecastItemSalesDF, on='Date', how='inner')

    return forecastResultDF

################################################################

