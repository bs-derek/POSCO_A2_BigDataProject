import pymysql

# Data Analyze Library
import pandas as pd
import numpy as np
import math

from sklearn.metrics import mean_absolute_error, precision_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.externals import joblib

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

def prod_recommend(l) :
    res = ''
    df_req = pd.read_csv("insu_request.csv", engine="python", encoding="euckr")
    a = pd.read_csv("a.csv", engine="python", encoding="euckr")
    price = pd.read_csv("insu_price.csv", engine = 'python', encoding = 'euckr')
    # ZZZ 결측치 제거
    df_reqz = df_req[df_req['sick_main'] != 'ZZZ']
    df_temp = pd.DataFrame({'req': df_reqz.groupby(["customer_id", "insu_prod_name", "req_id", "sick_main"])[
        'insu_req_amount'].sum()}).reset_index()
    df_pay = pd.DataFrame(
        {'pay': df_reqz.groupby(["customer_id", "insu_prod_name", "req_id"])['insu_pay_amount'].max()}).reset_index()
    df_temp['pay'] = df_pay['pay']
    df_temp = df_temp[df_temp['req'] != 0]
    df_temp.reset_index(inplace=True)
    df_temp = df_temp.drop('index', axis=1)
    df_temp['profit'] = 0
    a = a.drop('Unnamed: 0', axis=1)
    sick_name = l[10]
    prod_list = list(set(df_temp['insu_prod_name']))
    recommend = [0 for _ in range(21)]
    prod_prior = ['단체보험(상해)', 'All My Life 2', '평생 건강 보장 1', 'All My Life 1', '통합보험 +2',
                  '통합보험 +3', '건강+행복 보험', '평생 보험 1', '울라트 보험', '안심보험', '통합보험 +1',
                  '가족 만족 보험 2+', '건강보험', '평생 보험 2', '건강보험 3', '가족 만족 보험 2',
                  '가족 만족 보험 1', '조심조심 보험 1', '건강 보살핌', '조심조심 보험 4', '조심조심 보험 2']

    new_cust = a[a['customer_id'] == 'C112346']
    new_cust = new_cust.drop(['customer_id', 'judge_score', 'judge', 'review_cat'], axis=1)
    print(new_cust)
    new_cust['gender'][0] = l[0]
    new_cust['bp_judge'][0] = l[7]
    new_cust['pulse_count_judge'][0] = l[8]
    new_cust['month'][0] = l[9]
    new_cust['age_group'][0] = l[1]
    new_cust['height_group'][0] = l[2]
    new_cust['weight_group'][0] = l[3]
    new_cust['bust_group'][0] = l[4]
    new_cust['bmi_group'][0] = l[6]
    new_cust['waist_group'][0] = l[5]

    pro = []
    mo = []
    acc = []
    pre = []
    ans = []

    # 보험 상품만큼 for문
    for i in range(len(prod_list)):
        df = df_temp[(df_temp['insu_prod_name'] == prod_list[i]) & (df_temp['sick_main'] == sick_name)]
        # df = df.drop_duplicates(['customer_id'])
        df.reset_index(inplace=True)

        if len(df) >= 15:
            m = df['pay'].mean()
            for j in range(len(df)):
                if df['pay'][j] < m:
                    df['profit'][j] = 1
            df = df.drop(['index', 'insu_prod_name', 'req_id', 'sick_main', 'req', 'pay'], axis=1)
            df = pd.merge(df, a, on='customer_id')
            df = df.drop(['customer_id', 'judge_score', 'judge', 'review_cat'], axis=1)
            predictors = df.drop(['profit'], axis=1)
            target = df['profit']
            x_train, x_val, y_train, y_val = train_test_split(predictors, target, test_size=0.3, random_state=1234)
            #     print('Gaussian Naive Bayes')

            gaussian = GaussianNB()
            gaussian.fit(x_train, y_train)
            y_pred = gaussian.predict(x_val)
            ans_gaussian = gaussian.predict(new_cust)
            acc_gaussian = round(accuracy_score(y_pred, y_val) * 100, 2)
            pre_gaussian = round(precision_score(y_true=y_val, y_pred=y_pred) * 100, 2)
            #     print(acc_gaussian)
            #     print("Confusion matrix:\n{}".format(confusion_matrix(y_val, y_pred)))
            #     print('Precision : %.3f' %precision_score(y_true = y_val, y_pred = y_pred))
            #     print('============================================')
            #     print('Logistic Regression')

            logreg = LogisticRegression()
            logreg.fit(x_train, y_train)
            y_pred = logreg.predict(x_val)
            ans_logreg = logreg.predict(new_cust)
            acc_logreg = round(accuracy_score(y_pred, y_val) * 100, 2)
            pre_logreg = round(precision_score(y_true=y_val, y_pred=y_pred) * 100, 2)
            #         print(acc_logreg)
            #         print("Confusion matrix:\n{}".format(confusion_matrix(y_val, y_pred)))
            #         print('Precision : %.3f' %precision_score(y_true = y_val, y_pred = y_pred))
            #         print('============================================')
            #         print('Support vector Machines')

            svc = SVC()
            svc.fit(x_train, y_train)
            y_pred = svc.predict(x_val)
            ans_svc = svc.predict(new_cust)
            acc_svc = round(accuracy_score(y_pred, y_val) * 100, 2)
            pre_svc = round(precision_score(y_true=y_val, y_pred=y_pred) * 100, 2)
            #         print(acc_svc)
            #         print("Confusion matrix:\n{}".format(confusion_matrix(y_val, y_pred)))
            #         print('Precision : %.3f' %precision_score(y_true = y_val, y_pred = y_pred))
            #         print('============================================')
            #         print('Linear SVC')

            linear_svc = LinearSVC()
            linear_svc.fit(x_train, y_train)
            y_pred = linear_svc.predict(x_val)
            ans_linear_svc = linear_svc.predict(new_cust)
            acc_linear_svc = round(accuracy_score(y_pred, y_val) * 100, 2)
            pre_linear_svc = round(precision_score(y_true=y_val, y_pred=y_pred) * 100, 2)
            #     print(acc_linear_svc)
            #     print("Confusion matrix:\n{}".format(confusion_matrix(y_val, y_pred)))
            #     print('Precision : %.3f' %precision_score(y_true = y_val, y_pred = y_pred))
            #     print('============================================')
            #     print('Perceptron')

            perceptron = Perceptron()
            perceptron.fit(x_train, y_train)
            y_pred = perceptron.predict(x_val)
            ans_perceptron = perceptron.predict(new_cust)
            acc_perceptron = round(accuracy_score(y_pred, y_val) * 100, 2)
            pre_perceptron = round(precision_score(y_true=y_val, y_pred=y_pred) * 100, 2)
            #     print(acc_perceptron)
            #     print("Confusion matrix:\n{}".format(confusion_matrix(y_val, y_pred)))
            #     print('Precision : %.3f' %precision_score(y_true = y_val, y_pred = y_pred))
            #     print('============================================')
            #     print('Decision Tree')

            decisiontree = DecisionTreeClassifier()
            decisiontree.fit(x_train, y_train)
            y_pred = decisiontree.predict(x_val)
            ans_decisiontree = decisiontree.predict(new_cust)
            acc_decisiontree = round(accuracy_score(y_pred, y_val) * 100, 2)
            pre_decisiontree = round(precision_score(y_true=y_val, y_pred=y_pred) * 100, 2)
            #     print(acc_decisiontree)
            #     print("Confusion matrix:\n{}".format(confusion_matrix(y_val, y_pred)))
            #     print('Precision : %.3f' %precision_score(y_true = y_val, y_pred = y_pred))
            #     print('============================================')
            #     print('Random Forest')

            randomforest = RandomForestClassifier()
            randomforest.fit(x_train, y_train)
            y_pred = randomforest.predict(x_val)
            ans_randomforest = randomforest.predict(new_cust)
            acc_randomforest = round(accuracy_score(y_pred, y_val) * 100, 2)
            pre_randomforest = round(precision_score(y_true=y_val, y_pred=y_pred) * 100, 2)
            #     print(acc_randomforest)
            #     print("Confusion matrix:\n{}".format(confusion_matrix(y_val, y_pred)))
            #     print('Precision : %.3f' %precision_score(y_true = y_val, y_pred = y_pred))
            #     print('============================================')
            #     print('Gradient Boosting')

            gbk = GradientBoostingClassifier()
            gbk.fit(x_train, y_train)
            y_pred = gbk.predict(x_val)
            ans_gbk = gbk.predict(new_cust)
            acc_gbk = round(accuracy_score(y_pred, y_val) * 100, 2)
            pre_gbk = round(precision_score(y_true=y_val, y_pred=y_pred) * 100, 2)
            #         print(acc_gbk)
            #         print("Confusion matrix:\n{}".format(confusion_matrix(y_val, y_pred)))
            #         print('Precision : %.3f' %precision_score(y_true = y_val, y_pred = y_pred))
            #         print('============================================')
            #         print('KNN')

            knn = KNeighborsClassifier()
            knn.fit(x_train, y_train)
            y_pred = knn.predict(x_val)
            ans_knn = knn.predict(new_cust)
            acc_knn = round(accuracy_score(y_pred, y_val) * 100, 2)
            pre_knn = round(precision_score(y_true=y_val, y_pred=y_pred) * 100, 2)
            #         print(acc_knn)
            #         print("Confusion matrix:\n{}".format(confusion_matrix(y_val, y_pred)))
            #         print('Precision : %.3f' %precision_score(y_true = y_val, y_pred = y_pred))
            #         print('============================================')
            #         print('ANN')

            ann = MLPClassifier()
            ann.fit(x_train, y_train)
            y_pred = ann.predict(x_val)
            ans_ann = ann.predict(new_cust)
            acc_ann = round(accuracy_score(y_pred, y_val) * 100, 2)
            pre_ann = round(precision_score(y_true=y_val, y_pred=y_pred) * 100, 2)
            #     print(acc_ann)
            #     print("Confusion matrix:\n{}".format(confusion_matrix(y_val, y_pred)))
            #     print('Precision : %.3f' %precision_score(y_true = y_val, y_pred = y_pred))
            #     print('============================================')
            models = pd.DataFrame({'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 'Random Forest'
                , 'Naive Bayes', 'Perceptron', 'Linear SVC', 'Decision Tree', 'ANN', 'Gradient Boosting'],
                                   'Accuracy': [acc_svc, acc_knn, acc_logreg, acc_randomforest, acc_gaussian,
                                                acc_perceptron,
                                                acc_linear_svc, acc_decisiontree, acc_ann, acc_gbk],
                                   'Precision': [pre_svc, pre_knn, pre_logreg, pre_randomforest, pre_gaussian,
                                                 pre_perceptron,
                                                 pre_linear_svc, pre_decisiontree, pre_ann, pre_gbk],
                                   'Ans': [ans_svc[0], ans_knn[0], ans_logreg[0], ans_randomforest[0], ans_gaussian[0],
                                           ans_perceptron[0],
                                           ans_linear_svc[0], ans_decisiontree[0], ans_ann[0], ans_gbk[0]]})
            models = models.sort_values(by='Precision', ascending=False)
            models.reset_index(inplace=True)
            for j in range(len(models)):
                if models['Accuracy'][j] == 100.0 or models['Precision'][j] == 100.0:
                    continue
                elif models['Accuracy'][j] == models['Precision'][j]:
                    continue
                elif models['Accuracy'][j] == 0.0 or models['Precision'][j] == 0.0:
                    continue
                else:
                    # print('보험 :', prod_list[i], '모델 :', models['Model'][j], 'Accuracy :', models['Accuracy'][j], 'Precision :', models['Precision'][j])
                    # print("보험 : %-20s모델 : %-30sAccuracy : %-10.2fPrecision : %-10.2f" %(prod_list[i], models['Model'][j], models['Accuracy'][j], models['Precision'][j]))
                    pro.append(prod_list[i])
                    mo.append(models['Model'][j])
                    acc.append(models['Accuracy'][j])
                    pre.append(models['Precision'][j])
                    ans.append(models['Ans'][j])
                    break
    final = pd.DataFrame({'보험': pro, '모델': mo, 'Accuracy': acc, 'Precision': pre, '분류 결과': ans})

    if len(final) >= 1:
        recommend = [100]
        for k in range(len(final)):
            if final['분류 결과'][k] == 1:
                recommend[0] = min(recommend[0], prod_prior.index(final['보험'][k]))
    p = [0]
    if recommend[0] != 100:
        res += prod_prior[recommend[0]]
        for k in range(len(price)) :
            if price['insu_prod_name'][k] == prod_prior[recommend[0]] :
                p[0] = float(price['base_price'][k].replace(',',''))
                break
        p[0] = math.trunc(p[0])
    else:
        res += final['보험'][0]
        res += ' [10% 할증]'
        for k in range(len(price)) :
            if price['insu_prod_name'][k] == final['보험'][0] :
                p[0] = float(price['base_price'][k].replace(',','')) * 1.1
                p[0] = math.trunc(p[0])
                break
        p[0] = math.trunc(p[0])
    res += ' → 월 '
    p[0] = str(p[0])
    n = ''
    cnt = 0
    for i in range(len(p[0])-1, -1, -1) :
        if cnt % 3 == 0 and cnt != 0:
            n += ','
        cnt += 1
        n += p[0][i]

    r = ''
    for i in range(len(n) - 1, -1, -1) :
        r += n[i]

    res += r
    res += '원'
    return res

#Finished Code
if __name__ == '__main__':
    pass
