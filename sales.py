from flask import Flask, render_template, request, redirect, url_for
from joblib import load
import numpy as np

import random
from flask import Flask, render_template, abort, Response, request
from salesForecastResult import get_apiData

######### TOTAL-SALES ############
# trained model
model = load('salesForecast_v2.joblib')
# trained result dataframe
actualVsPredictDF = load('comparisionDF_v2.joblib')

######### TOTAL-SALES ############
# item Sales - trained Model
itemModel = load('itemSalesModel_v0.joblib')
# item Sales - result dataframe
itemResult = load('itemSalesResult_v0.joblib')

# forecast result dataframe
forecastResultDF = get_apiData()

# creating dataframe with model trained result and forecast result
salesResultDF = actualVsPredictDF.append(forecastResultDF[['Date', 'newForecast']], ignore_index=True)

itemSalesResultDF = itemResult.append(forecastResultDF[['Date', 'barjariOotaF', 'chaatF', 'hotBeveragesF', 'iceCreamF', 'idlyF', 'juiceF', 'masalaDosaF', 'southMealsF']], ignore_index=True)


app = Flask(__name__)

colors = [
    "#F7464A", "#46BFBD", "#FDB45C", "#FEDCBA",
    "#ABCDEF", "#DDDDDD", "#ABCABC", "#4169E1",
    "#C71585", "#FF4500", "#FEDCBA", "#46BFBD"]

#

@app.route('/')
@app.route('/home')
@app.route('/index')
def home():
    
    return render_template('index81.html', 
    title='Daily Sales Forecast', 
    salesDate = salesResultDF['Date'], itemSalesDate = itemSalesResultDF['Date'],
    labels=actualVsPredictDF['Date'], 
    actual=actualVsPredictDF['Actual'], predicted=actualVsPredictDF['Predicted'], 
    fSalesLegend = "Daily Sales Forecast",
    fDate = forecastResultDF['Date'], fSales = forecastResultDF['newForecast'],
    titleItem='Item-wise Daily Sales Forecast', 
    iLabel = itemSalesResultDF['Date'], 
    iBO_A = itemSalesResultDF['barjariOota_A'], iBO_P = itemSalesResultDF['barjariOota_P'],
    iC_A = itemSalesResultDF['chaat_A'], iC_P = itemSalesResultDF['chaat_P'], 
    iHB_A = itemSalesResultDF['hotBeverages_A'], iHB_P = itemSalesResultDF['hotBeverages_P'], 
    iCC_A = itemSalesResultDF['iceCream_A'], iCC_P = itemSalesResultDF['iceCream_P'], 
    iI_A = itemSalesResultDF['idly_A'], iI_P = itemSalesResultDF['idly_P'], 
    iJ_A = itemSalesResultDF['juice_A'], iJ_P = itemSalesResultDF['juice_P'],
    iMD_A = itemSalesResultDF['masalaDosa_A'], iMD_P = itemSalesResultDF['masalaDosa_P'], 
    iSM_A = itemSalesResultDF['southMeals_A'], iSM_P = itemSalesResultDF['southMeals_P'],
    iLableF = itemSalesResultDF['Date'], 
    iBO_F = itemSalesResultDF['barjariOotaF'], 
    iC_F = itemSalesResultDF['chaatF'], iHB_F = itemSalesResultDF['hotBeveragesF'], 
    iCC_F = itemSalesResultDF['iceCreamF'], iI_F = itemSalesResultDF['idlyF'],
    iJ_F = itemSalesResultDF['juiceF'], iMD_F = itemSalesResultDF['masalaDosaF'], 
    iSM_F = itemSalesResultDF['southMealsF'])

# ['Date', 'barjariOota_A', 'barjariOota_P', 'chaat_A', 'chaat_P',
       # 'hotBeverages_A', 'hotBeverages_P', 'iceCream_A', 'iceCream_P',
       # 'idly_A', 'idly_P', 'juice_A', 'juice_P', 'masalaDosa_A',
       # 'masalaDosa_P', 'southMeals_A', 'southMeals_P']


if __name__ == '__main__' :
    app.run(debug=True)