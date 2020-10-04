from flask import Flask,render_template,request
import numpy as np 

app = Flask(__name__)
import pickle

model = pickle.load(open('osi_knn', 'rb'))


@app.route("/",methods=['GET','POST'])
def hello_world():
    return render_template("index.html")


def Encoding_Month(month):
    num = 0
    if month == 'February':
        num=2
    elif month == 'March'or month == 'April':
        num=5
    elif month == 'May':
        num=6
    elif month == 'June':
        num=4
    elif month == 'July':
        num=3
    elif month == 'August':
        num=0
    elif month == 'September':
        num=9
    elif month == 'October':
        num=8
    elif month == 'November':
        num=7
    elif month == 'Descember' or month=='January':
        num=1
    
    return num

def Encoding_Visitor(vis):
    num = 0
    if vis == 'Returning Visitor':
        num = 2
    elif vis == 'New Visitor':
        num = 0
    elif vis == 'Other':
        num = 1

    return num

def Encoding_Region(region):
    num = 0
    if region == 'United States':
        num = 1
    elif region == 'Germany':
        num = 2
    elif region == 'India':
        num = 3
    elif region == 'France':
        num = 4  
    elif region == 'United Kingdom':
        num = 5
    elif region == 'Brazil':
        num = 6
    elif region == 'Italy':
        num = 7
    elif region == 'Austalia':
        num = 8
    elif region == 'Spain':
        num = 9

    return num


mean = [5228.289588354559,
        1350.3003767343155,
        2228.8511406021103,
        1.6308603553860055,
        0.04701054003755943,
        15.070402452799382,
        3.063447445035639,
        5.232506776428069]

std =  [33855.16236319768,
        11714.049416369837,
        15173.139612558776,
        0.7571679042941348,
        0.169662213903642,
        29.5744024822749,
        2.3508388396800353,
        2.3483951485421475]

@app.route("/out",methods=['GET','POST'])
def Out():
    return render_template('out.html')

@app.route("/features",methods=['GET','POST'])
def features():

    if (request.method=='POST'):
        browser = request.form.get('browser')
        pro_page = request.form.get('pro_page')
        pro_time = request.form.get('pro_time')
        inf_page = request.form.get('inf_page')
        inf_time = request.form.get('inf_time')
        adm_page = request.form.get('adm_page')
        adm_time = request.form.get('adm_time')
        visType = request.form.get('visType')
        spec_day = request.form.get('spec_day')
        Region = request.form.get('Region')
        pageValue = request.form.get('pageValue')
        month = request.form.get('month')

        pro_page_pr_time = float(pro_page)/(float(pro_time)+0.00001)
        inf_page_pr_time = float(inf_page)/(float(inf_time)+0.00001)
        adm_page_pr_time = float(adm_page)/(float(adm_time)+0.00001)
        visType = Encoding_Visitor(visType)
        Region = Encoding_Region(Region)
        month = Encoding_Month(month)


        points = [pro_page_pr_time,adm_page_pr_time,inf_page_pr_time,visType,spec_day,pageValue,Region,month]
        da = [[float(i)] for i in points]

        data = np.asarray(da).T

        data = (data - mean)/std
        output = model.predict(data)

        if output:
            color = "text-success"
            messege = "Model predicts, The revenue will be Generated"
        else:
            color = "text-danger"
            messege = "Model predicts revenue will not be generated"
        return render_template('out.html',messege = messege, color=color)
    return render_template('out.html')

if __name__ == '__main__':
    app.run(debug=True)
