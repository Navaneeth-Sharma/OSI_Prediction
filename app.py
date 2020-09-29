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

mean = [5218.859029205611,1387.9485510837133,2194.767099990961,
 0.24628551609082644,14.984616045190817,5.289006914520493]

std = [33825.34232265553,11869.013378146772,15064.67360529188,
 0.3170625256472935,29.364790319103296,2.354823839699579]

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
        bounce = request.form.get('bounce')
        exit1 = request.form.get('exit')
        pageValue = request.form.get('pageValue')
        month = request.form.get('month')
        os = request.form.get('os')

        pro_page_pr_time = float(pro_page)/(float(pro_time)+0.00001)
        inf_page_pr_time = float(inf_page)/(float(inf_time)+0.00001)
        adm_page_pr_time = float(adm_page)/(float(adm_time)+0.00001)
        boun_pr_exit = float(bounce)/(float(exit1)+0.00001)

        points = [pro_page_pr_time,adm_page_pr_time,inf_page_pr_time,boun_pr_exit,pageValue]
        da = [[float(i)] for i in points]

        da.append([Encoding_Month(month)])
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