import requests
import json
import time
import os
import random


# нужен python3.9 и выше

# токен авторизации (изменить при получении нового)
auth = "ZV9uZXN0ZXJvdl9pbnRlZ3JhdGlvbkBpbnZlbnRvczoxNDgzNjM0NzIzOjk5OTk5OTk5OTpUbWJFK2R2NFpOQUgrVmJDenRQdy9BPT0"
# url для запроса на отчет report_autocomplete_test (может измениться со сменой токена авторизации)
report_url = "https://b2b-api.spectrumdata.ru/b2b/api/v1/user/reports/report_autocomplete_plus%40inventos/_make" # было report_autocomplete_test

# английская раскладка номера тоже работает (проверить).

def make_url(url):
    '''
    Требуется python версии 3.9 и выше.
    Преобразовывает uid отчета в ссылку для GET запроса по нему.
    Аргументы:
        url - uid отчета.
    return - возвращает GET ссылку.
    '''
    x = url.removesuffix("@inventos")
    x = x.replace("=", "%3D")
    x = 'https://b2b-api.spectrumdata.ru/b2b/api/v1/user/reports/' + x + "%40inventos?_detailed=true&_content=true"
    return x


def get_model(url, authorization, accept="application/json"):
    '''
    Выводит номер, марку, модель, год и цвет автомобиля.
    Аргументы:
        url - ссылка GET запроса отчета;
        authorization - токен авторизации;
        accept - формат данных (лучше не трогать этот параметр) (default="application/json").
    '''
    time.sleep(3)
    flag = False
    while flag == False:
         x = json.loads(requests.get(url, headers={"Accept": accept,
                                                   "Authorization": authorization}).text)
         if x["data"][0].get("status") == None or x["data"][0].get("status") == "FINISH":
             #print("-"*20)
             print("Отчет готов.")
             #print("Get_model----", x)
             #print("-"*20)
             flag = True
         else:
             print("Отчет не готов {0}, ждем 5 секунд...".format(x["data"][0].get("status")))
             time.sleep(5)

    # проблема: может отсутствовать поле model, а информация о модели
    # содержаться в x["data"][0]["content"]["tech_data"]["brand"]["name"]["original"]
    # однако это мог быть частынй случай с Камазом и для остальных авто он неприменим.
    print("Номер автомобиля: {0}".format(x["data"][0]["vehicle_id"]))
    print("Модель автомобиля: {0} {1} {2}".format(x["data"][0]["content"]["tech_data"]["brand"]["name"]["normalized"],
                                                  x["data"][0]["content"]["tech_data"].get("model")["name"]["normalized"] if x["data"][0]["content"]["tech_data"].get("model") != None else "No model",
                                                  x["data"][0]["content"]["tech_data"]["year"]))
    print(x["data"][0]["content"]["tech_data"]["body"]["color"]["name"])

    return x["data"][0]["content"]["tech_data"]["brand"]["name"]["normalized"] + " " + (x["data"][0]["content"]["tech_data"].get("model")["name"]["normalized"] if x["data"][0]["content"]["tech_data"].get("model") != None else "No model") + " " + str(x["data"][0]["content"]["tech_data"]["year"])



def make_report(start_url, authorization, grz, accept="application/json"):
    '''
    Делает POST запрос на формирование отчета по номеру автомобиля.
    Аргументы:
        start_url - ссылка для POST запроса на формирование отчета (переменная report_url выше, но может измениться со сменой токена авторизации);
        authorization - токен авторизации;
        grz - государственный номер транспортного средства (пример: "А111АА77");
        accept - формат данных (лучше не трогать этот параметр) (default="application/json").
    return - возвращает uid сформированного отчета.
    '''
    data = {"queryType": "GRZ", "query": grz}
    headers = {"Accept": accept, "Authorization": authorization}
    x = requests.post(start_url, headers=headers, json=data)
    #print("-" * 20)
    x = json.loads(x.text)
    #print(x)
    return x["data"][0]["uid"]


def get_all_reports(url, auth,accept="application/json"):
    '''
    Выводит uid всех отчетов.
    Аргументы:
        url - ссылка GET запроса списка отчетов (получается на сайте https://b2b-api.spectrumdata.ru/swagger-ui.html).
        auth - токен авторизации.
        accept - формат данных (лучше не трогать этот параметр) (default="application/json").
    '''
    x = json.loads(requests.get(url, headers={"Accept": accept,
                                      "Authorization": auth}).text)
    for i in x["data"]:
        print(i["uid"])

# делаем запрос и извлекаем инофрмацию из отчета
#get_model(make_url(make_report(report_url, auth, 'E020BX154')), auth)


def final_step():
    f = open('data.txt', 'a')
    i = 1
    grz_list = {}
    dir = "/home/pavel/Desktop/University/Diplom//car_rectangle/frames1" # папка с изображениями (изменить)
    for filename in os.listdir(dir):
        start_filename = filename
        filename = filename.replace("(", "").replace(".jpg", "")
        if filename in grz_list.keys():
            tar = grz_list[filename] + '_' + str(random.randint(1, 100000000))
        else:
            try:
                tar = get_model(make_url(make_report(report_url, auth, filename)), auth)
                grz_list[filename] = tar
                f.write(filename +' : ' + tar + '\n')
                tar = tar + '_' + str(random.randint(1, 100000000))
            except Exception as err:
                print("Fail!" * 5)
                print(err)
                if str(err) == "'tech_data'":
                    os.remove(dir + "/" + start_filename)
                continue

        os.rename(dir + "/" + start_filename, dir + "/" + tar + ".jpg")
        print("Success!", i, "/", len(os.listdir(dir)))
        time.sleep(3)
        i += 1
    f.close()

final_step()

# curl -X POST --header 'Content-Type: application/json' --header 'Accept: application/json'
# --header 'Authorization: ZV9uZXN0ZXJvdl9pbnRlZ3JhdGlvbkBpbnZlbnRvczoxNjI5NzE4NTAwOjk5OTk5OTk5OTozQ2x2cTNJUFY3akNCKzAzZHRHS0N3PT0' -d
# '{"queryType": "GRZ","query": "А111АА77"}' 'https://b2b-api.spectrumdata.ru/b2b/api/v1/user/reports/report_autocomplete_plus%40inventos/_make'
