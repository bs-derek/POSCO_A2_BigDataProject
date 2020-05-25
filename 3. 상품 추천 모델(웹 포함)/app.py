from flask import Flask, render_template, request, redirect, session, jsonify
import pandas as pd
import numpy as np
import datetime
import dbMgr

# Flask Class 객체를 선언한다 . 
app = Flask(__name__, template_folder = './templates', static_folder='./static')

# Main Page
@app.route('/') # Local Host 
def index():
        return render_template("index.html")

###########################################################################

@app.route('/about')
def data_model():
        return render_template('about.html')

@app.route('/contact')
def contact():
        return render_template('contact_v1.html')

@app.route('/FAQs')
def faq():
        return render_template('faqs_right_sidebar.html')

@app.route('/insurance_product')
def insurance_product():
        return render_template('insurance_product.html')

@app.route('/our_agent')
def our_agent():
        return render_template('our_agent.html')

@app.route('/pension_product')
def pension_product():
        return render_template('pension_product.html')

@app.route('/pricing_table')
def pricing_table():
        return render_template('pricing_tables_v2.html')

@app.route('/product_recommend')
def product_recommend():
        return render_template('product_recommend.html')


# Data Predict & Insert Data Base
@app.route('/product_recommend_result',methods=['POST'])
def predict_data():
        
        print(request.form)

        sick_name = request.form['sick_name']
        if sick_name == '이상 없음' :
                sick_name = 'R05'
        else :
                sick_name = sick_name[0:3]

        gender = request.form['gender']

        if gender == '남' :
            gender = 1
        else :
            gender = 2

        birth = request.form['birth']
        convert_birth = datetime.datetime.strptime(birth, "%Y.%m.%d").date()
        year = convert_birth.year
        month = convert_birth.month

        age = datetime.datetime.today().year - year
        if age <= 17 : 
            age = 0 
        elif age >= 18 and age <= 24 :
            age = 1
        elif age >= 25 and age <= 34 :
            age = 2
        elif age >= 35 and age <= 59 :
            age = 3
        else :
            age = 4

        height = int(request.form['height'])
        if height <= 139 :
            height = 0
        elif height >= 140 and height <= 164 :
            height = 1
        elif height >= 165 and height < 180 :
            height = 2
        else :
            height = 3

        weight = int(request.form['weight'])
        if weight <= 44 :
            weight = 0
        elif weight >= 45 and weight < 80 :
            weight = 1
        elif weight >= 80 and weight < 100 :
            weight = 2
        else :
            weight = 3

        bust = int(request.form['bust'])
        if bust < 80 : 
            bust = 0
        elif bust >= 80 and bust < 100 :
            bust = 1
        else :
            bust = 2

        waist = int(request.form['waist'])
        if waist <= 69 :
            waist = 0
        elif waist >= 70 and waist <95 :
            waist = 1
        else :
            waist = 2

        bmi = weight / ((height/100)**2)
        if bmi <= 18.4 : 
            bmi = 0 
        elif bmi >= 18.5 and bmi < 23 :
            bmi = 1
        elif bmi >= 23 and bmi < 25 :
            bmi = 2
        elif bmi >= 25 and bmi < 30 :
            bmi = 3
        else :
            bmi = 4

        bp_min = int(request.form['bp_min'])
        bp_max = int(request.form['bp_max'])
        if bp_min<85 and bp_max<130 :
                bp_judge = 0
        else :
                bp_judge = 1

        pulse_count = int(request.form['pulse_count'])
        if 60<=pulse_count<=120 :
                pulse_count_judge = 0
        else :
                pulse_count_judge = 1
        
        # Input Data 
        data = [gender, age, height, weight, bust, waist, bmi, bp_judge, pulse_count_judge, month, sick_name]
    

        rec_prod = dbMgr.prod_recommend(data)
        result = { 'data' : rec_prod }
        # return render_template('product_recommend.html', rec_prod=rec_prod)
        return result

# Finished Code 
if __name__ == '__main__' :
    app. debug = True
    app.run()