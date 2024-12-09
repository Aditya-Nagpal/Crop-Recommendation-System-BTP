from flask import Flask,request,render_template
import numpy as np
import pandas
import sklearn
import pickle

# importing model
model = pickle.load(open('model.pkl','rb'))

# creating flask app
app = Flask(__name__)

def GetImageURL(crop):
    crop_url={"apple": "https://www.croptrust.org/fileadmin/_processed_/a/3/csm_Apple_2e6cc719c3.jpeg",
              "banana": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSB1qSojWH8xWNk1H8o7FEto-Mxt71WtdWH4g&s",
              "blackgram": "https://geolife.com/assets/images/black-gram-561x398.jpg",
              "chickpea": "https://media.istockphoto.com/id/638538708/photo/woman-showing-chickpeas-in-close-up.jpg?s=612x612&w=0&k=20&c=ZAZ-5i5KuuteCEOZrrwQ3S30yh-ptUVwZ752-LG90cg=",
              "coconut": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRPNE4tYJi3mQrbluxTRQUxh1z9IJSv4sY1YA&s",
              "coffee": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRK-o-62_RDDWuEZOMsEk_WMu4iNVu8b_R36A&s",
              "cotton": "https://cdn.pixabay.com/photo/2019/11/24/17/08/cotton-4649804_640.jpg",
              "grapes": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcS8xYaB3nNWGVxoxeYSkOKOJTaSxoUhmK4krg&s",
              "jute": "https://t3.ftcdn.net/jpg/05/61/99/80/360_F_561998023_YmOc0Qe3VTK0o5uhJ9eH3BSX49z5dDVl.jpg",
              "kidneybeans": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSIC_IEeESlbmNEchVfns3xCNsYV8vCHg4WUg&s",
              "lentil": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQFoXxrz5yaLVf0kn-v2QzqMi4zI3LOa27zlA&s",
              "maize": "https://media.istockphoto.com/id/1061097354/photo/the-corn-plant-in-the-field.jpg?s=612x612&w=0&k=20&c=NEEzE5il-up8g7NZj_7HJUpyVep18zBRfhnMZ5laLiQ=",
              "mango": "https://media.istockphoto.com/id/1435602229/photo/close-up-of-red-mangoes.jpg?s=612x612&w=0&k=20&c=a2uO7Ly-irGjtfqZC0ZTt9ps_Sh8S3a6ulf-TMRebao=",
              "mothbeans": "https://kj1bcdn.b-cdn.net/media/52062/vigna-aconitifolia_leavesflowers-mjussoorie-chakrata-road-near-bharatkhai-1-dsc09876.jpg",
              "mungbean": "https://www.pulseaus.com.au/storage/app/uploads/public/569/9e6/01b/5699e601bbbe8831660831.jpg",
              "muskmelon": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQlYKIcNoVpzIz1JT8xIsdRiL7ohBF8y7V4Eg&s",
              "orange": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTACoTOHo4aMZ4ovxxM3KRJwCrLj3VB5UdpRg&s",
              "papaya": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRGDlhohW4uBYguzppHOAG5hKPYgdqRtE4Alw&s",
              "pigeonpeas": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRqcl-qbVAZN2dVLnOb2sKYa7fZeVRCAdGEhQ&s",
              "pomegranate": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRd4pexxp_Z31nXPw4ufNq1xn_24qMBq12gYw&s",
              "rice": "https://media.istockphoto.com/id/622925154/photo/ripe-rice-in-the-field-of-farmland.jpg?s=612x612&w=0&k=20&c=grtA7L3dm_SP80Fdt-PpIwu5GYacZygErTDUDNIKHwY=",
              "watermelon": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRoqrczkzxkV9K7mqTCSto9ME02mNG42bGH3A&s"
            }
    return crop_url[crop]

def CompareNutrients(N,P,K,crop):
    crop_npk_ranges = {    "apple": {"N": (0, 40), "P": (120, 145), "K": (195, 205)},
    "banana": {"N": (80, 120), "P": (70, 95), "K": (45, 55)},    "blackgram": {"N": (20, 60), "P": (55, 80), "K": (15, 25)},
    "chickpea": {"N": (20, 60), "P": (55, 80), "K": (75, 85)},    "coconut": {"N": (0, 40), "P": (5, 30), "K": (25, 35)},
    "coffee": {"N": (80, 120), "P": (15, 40), "K": (25, 35)},    "cotton": {"N": (100, 140), "P": (35, 60), "K": (15, 25)},
    "grapes": {"N": (0, 40), "P": (120, 145), "K": (195, 205)},    "jute": {"N": (60, 100), "P": (35, 60), "K": (35, 45)},
    "kidneybeans": {"N": (0, 40), "P": (55, 80), "K": (15, 25)},    "lentil": {"N": (0, 40), "P": (55, 80), "K": (15, 25)},
    "maize": {"N": (60, 100), "P": (35, 60), "K": (15, 25)},    "mango": {"N": (0, 40), "P": (15, 40), "K": (25, 35)},
    "mothbeans": {"N": (0, 40), "P": (35, 60), "K": (15, 25)},    "mungbean": {"N": (0, 40), "P": (35, 60), "K": (15, 25)},
    "muskmelon": {"N": (80, 120), "P": (5, 30), "K": (45, 55)},    "orange": {"N": (0, 40), "P": (5, 30), "K": (5, 15)},
    "papaya": {"N": (31, 70), "P": (46, 70), "K": (45, 55)},    "pigeonpeas": {"N": (0, 40), "P": (55, 80), "K": (15, 25)},
    "pomegranate": {"N": (0, 40), "P": (5, 30), "K": (35, 45)},    "rice": {"N": (60, 99), "P": (35, 60), "K": (35, 45)},
    "watermelon": {"N": (80, 120), "P": (5, 30), "K": (45, 55)},}

    ideal_values=crop_npk_ranges[crop]
    N_report=""
    P_report=""
    K_report=""

    N_low=ideal_values["N"][0]
    N_high=ideal_values["N"][1]

    P_low=ideal_values["P"][0]
    P_high=ideal_values["P"][1]

    K_low=ideal_values["K"][0]
    K_high=ideal_values["K"][1]

    if N < N_low:
        N_report="Nitrogen is deficient"
    elif N > N_high:
        N_report="Nitrogen is surplus"
    else:
        N_report="Nitrogen is ideal"
    
    if P < P_low:
        P_report="Phosporous is deficient"
    elif P > P_high:
        P_report="Phosporous is surplus"
    else:
        P_report="Phosporous is ideal"
    
    if K < K_low:
        K_report="Potassium is deficient"
    elif K > K_high:
        K_report="Potassium is surplus"
    else:
        K_report="Potassium is ideal"

    return [N_report, P_report, K_report]



@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predict",methods=['POST'])
def predict():
    N=float(request.form.get('Nitrogen'))
    P=float(request.form.get('Phosporus'))
    K=float(request.form.get('Potassium'))
    temp=float(request.form.get('Temperature'))
    humidity=float(request.form.get('Humidity'))
    ph=float(request.form.get('Ph'))
    rainfall=float(request.form.get('Rainfall'))

    feature_list=np.array([N,P,K,temp,humidity,ph,rainfall]).reshape(1,-1)
    prediction=model.predict(feature_list)
    nutrient_comparisons=CompareNutrients(N,P,K,prediction[0])
    crop_img_url=GetImageURL(prediction[0])
    return render_template('index.html',result=prediction[0],N_result=nutrient_comparisons[0],P_result=nutrient_comparisons[1],K_result=nutrient_comparisons[2],crop_url=crop_img_url)




# python main
if __name__ == "__main__":
    app.debug=True
    app.run()