import streamlit as st
import pandas as pd
import pickle
from PIL import Image
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

st.write("""
# Мини-приложение для определения вида ириса.
По размерам ***цветка ириса*** предсказывает его вид.
""")

image = Image.open("Iris_logo.png")
st.image(image, width=300)

st.markdown(
    """
* **Библиотеки Python:** pandas, streamlit, sklearn
* **Источник:** [Классический Iris Dataset из scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html).
""")

st.sidebar.header('Параметры ириса')

def user_input_features():
    sepal_length = st.sidebar.slider('Длина чашелистника', 4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider('Ширина чашелистника', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('Длина лепестка', 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider('Ширина лепестка', 0.1, 2.5, 0.2)
    data = {'Длина чашелистника': sepal_length,
            'Ширина чашелистника': sepal_width,
            'Длина лепестка': petal_length,
            'Ширина лепестка': petal_width}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('Введённые параметры:')
st.write(df)

iris = datasets.load_iris()
X = iris.data
Y = iris.target

# если нужно использовать предобученную модель:
# with open('model_iris.rds', 'rb') as f:
#     clf = pickle.load(f)
clf = RandomForestClassifier()
clf.fit(X, Y)

prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

st.subheader('Сорт ириса и индекс этого таргета')
st.write(iris.target_names)

st.subheader('Предсказанный сорт')
st.write(iris.target_names[prediction])
#st.write(prediction)

st.subheader('Предсказанная вероятность')
st.write(prediction_proba)