# Importing useful packages
from tkinter import filedialog
import numpy as np
import matplotlib.pyplot as plt
from tkinter import *
import os
import cv2
import pickle
from PIL import Image

# Creating Tkinter object-GUI
window = Tk()
window.title("Bacterial Spot Prediction")
window.geometry('700x300')
window.configure(background='seagreen')

# Creating pie charts for descriptive Statistics


def graph():
    labels = "Pepperbell_bacterial_spots", "Pepperbell_healthy"
    sizes = [40.2, 59.8]
    colors = ['gold', 'lightskyblue']
    explode = (0.1, 0)
    plt.pie(sizes, explode=explode, labels=labels, colors=colors,
            autopct='%1.1f%%', shadow=True, startangle=250)
    plt.axis('equal')
    fig = plt.gcf()
    fig.set_size_inches(8, 8)
    plt.show()


def graph1():
    labels = "Potato_early_blight", "Potato_healthy", "Potato_late_blight"
    sizes = [46.4, 7.2, 46.4]
    colors = ['gold', 'lightskyblue', 'yellowgreen']
    explode = (0.1, 0, 0)
    plt.pie(sizes, explode=explode, labels=labels, colors=colors,
            autopct='%1.1f%%', shadow=True, startangle=250)
    plt.axis('equal')
    fig = plt.gcf()
    fig.set_size_inches(8, 8)
    plt.show()


def graph2():
    labels = ["Tomato_bacterial_spot", "Tomato_early_blight", "Tomato_healthy", "Tomato_late_blight",
              "Tomato_leaf_mold", "Tomato_septoria_leaf_spot", "Tomato_target_spot", "Tomato-mosaic_virus",
              "Tomato-spider_mites_2_spotted_spider_mite", "Tomato-yellow_leaf_curl_virus"]
    sizes = [13.3, 6.3, 9.9, 11.9, 5.9, 11, 8.8, 2.4, 10.4, 20.1]
    colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue', 'brown']
    explode = (0, 0.1, 0, 0, 0, 0, 0, 0, 0, 0)
    plt.pie(sizes, explode=explode, labels=labels, colors=colors,
            autopct='%1.1f%%', shadow=True, startangle=250)
    plt.axis('equal')
    fig = plt.gcf()
    fig.set_size_inches(8, 8)
    plt.show()


def descrstats():
    labels = ["Bacterial_Spots", "Pepperbell_healthy", "Potato_early_blight", "Potato_healthy", "Potato_late_blight",
              "Tomato_early_blight", "Tomato_healthy", "Tomato_late_blight", "Tomato_leaf_mold",
              "Tomato_septoria_leaf_spot", "Tomato_target_spot", "Tomato-mosaic_virus",
              "Tomato-spider_mites_2_spotted_spider_mite", "Tomato-yellow_leaf_curl_virus"]
    sizes = [15.13, 7.1, 4.8, 0.7, 4.8, 4.8, 7.7, 9.2, 4.6, 8.5, 6.8, 1.8, 8.1, 15.5]
    colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue', 'brown']
    explode = (0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,)
    plt.pie(sizes, explode=explode, labels=labels, colors=colors,
            autopct='%1.1f%%', shadow=True, startangle=250)
    plt.axis('equal')
    fig = plt.gcf()
    fig.set_size_inches(8, 8)
    plt.show()

# Create button and function where user can upload his/her dataset


def browse_button():
    global folder_path
    global DATADIR
    global dataset
    global images

    filename = filedialog.askdirectory()
    folder_path.set(filename)
    DATADIR = filename
    dataset = list()
    images = list()

    for item in os.listdir(DATADIR):
        directory = os.path.join(DATADIR, item)
        im = Image.open(directory)
        f, e = os.path.splitext(DATADIR + item)
        imresize = im.resize((32, 32), Image.ANTIALIAS)
        imresize.save(directory, 'JPEG', quality=90)

    for img in os.listdir(DATADIR):
        img_array = cv2.imread(os.path.join(DATADIR, img))
        dataset.append(img_array)
        images_names = os.path.split(img)[-1]
        images.append(images_names)


folder_path = StringVar()
lbl1 = Label(master=window, textvariable=folder_path, text="Path", height=1, width=40)
lbl1.grid(row=80, column=201)
button2 = Button(text="Browse", command=browse_button, height=3, width=60)
button2.grid(row=80, column=200)


# Creating function that yields to user's dataset distribution


def distribution():
    filename = 'finalized_model.sav'
    loaded_model = pickle.load(open(filename, 'rb'))

    distribution_dataset = np.asarray(dataset)
    distribution_dataset = distribution_dataset / 255.0

    predictions_distr = loaded_model.predict(distribution_dataset)
    y_pred_distr = [np.argmax(probas) for probas in predictions_distr]

    pepperbell_bact_spot = list()
    pepperbell_healthy = list()
    potato_early_blight = list()
    potato_healthy = list()
    potato_late_blight = list()
    tomato_bact_spot = list()
    tomato_early_blight = list()
    tomato_healthy = list()
    tomato_late_blight = list()
    tomato_leaf_mold = list()
    tomato_septoria_leaf_spot = list()
    tomato_target_spot = list()
    tomato_mosaic_virus = list()
    tomato_spider_mites_2_spotted_spider_mite = list()
    tomato_yellow_leaf_curl_virus = list()

    for i in range(0, len(y_pred_distr)):

        if y_pred_distr[i] == 0:
            pepperbell_bact_spot.append(i)
        elif y_pred_distr[i] == 1:
            pepperbell_healthy.append(i)
        elif y_pred_distr[i] == 2:
            potato_early_blight.append(i)
        elif y_pred_distr[i] == 3:
            potato_healthy.append(i)
        elif y_pred_distr[i] == 4:
            potato_late_blight.append(i)
        elif y_pred_distr[i] == 5:
            tomato_bact_spot.append(i)
        elif y_pred_distr[i] == 6:
            tomato_early_blight.append(i)
        elif y_pred_distr[i] == 7:
            tomato_healthy.append(i)
        elif y_pred_distr[i] == 8:
            tomato_late_blight.append(i)
        elif y_pred_distr[i] == 9:
            tomato_leaf_mold.append(i)
        elif y_pred_distr[i] == 10:
            tomato_septoria_leaf_spot.append(i)
        elif y_pred_distr[i] == 11:
            tomato_target_spot.append(i)
        elif y_pred_distr[i] == 12:
            tomato_mosaic_virus.append(i)
        elif y_pred_distr[i] == 13:
            tomato_spider_mites_2_spotted_spider_mite.append(i)
        elif y_pred_distr[i] == 14:
            tomato_yellow_leaf_curl_virus.append(i)

    root_1 = Toplevel()
    root_1.title("User's Dataset Distribution")
    root_1.geometry('1000x450')
    root_1.configure(background='seagreen')

    def print_all():

        x = 0
        if len(pepperbell_bact_spot) == 0:
            x = 0
        elif len(pepperbell_bact_spot) != 0:
            x = len(pepperbell_bact_spot)
        count = StringVar()
        count.set("The pepperbells with Bacterial spots are {}:".format(x))
        label_1 = Label(master=root_1, textvariable=count)
        label_1.grid(row=2, column=3)

        x_2 = 0
        if len(pepperbell_healthy) == 0:
            x_2 = 0
        elif len(pepperbell_healthy) != 0:
            x_2 = len(pepperbell_healthy)

        count_2 = StringVar()
        count_2.set("The healthy pepperbells are {}:".format(x_2))
        label_2 = Label(master=root_1, textvariable=count_2)
        label_2.grid(row=4, column=3)

        x_3 = 0
        if len(potato_early_blight) == 0:
            x_3 = 0
        elif len(potato_early_blight) != 0:
            x_3 = len(potato_early_blight)

        count_3 = StringVar()
        count_3.set("The potato early blight are {}:".format(x_3))
        label_3 = Label(master=root_1, textvariable=count_3)
        label_3.grid(row=6, column=3)

        x_4 = 0
        if len(potato_healthy) == 0:
            x_4 = 0
        elif len(potato_healthy) != 0:
            x_4 = len(potato_healthy)

        count_4 = StringVar()
        count_4.set("The potato healthy  are {}:".format(x_4))
        label_4 = Label(master=root_1, textvariable=count_4)
        label_4.grid(row=8, column=3)

        x_5 = 0
        if len(potato_late_blight) == 0:
            x_5 = 0
        elif len(potato_late_blight) != 0:
            x_5 = len(potato_late_blight)

        count_5 = StringVar()
        count_5.set("The potato late blight are {}:".format(x_5))
        label_5 = Label(master=root_1, textvariable=count_5)
        label_5.grid(row=10, column=3)

        x_6 = 0
        if len(tomato_bact_spot) == 0:
            x_6 = 0
        elif len(tomato_bact_spot) != 0:
            x_6 = len(tomato_bact_spot)

        count_6 = StringVar()
        count_6.set("The tomato bacterial spot are {}:".format(x_6))
        label_6 = Label(master=root_1, textvariable=count_6)
        label_6.grid(row=12, column=3)

        x_7 = 0
        if len(tomato_early_blight) == 0:
            x_7 = 0
        elif len(tomato_early_blight) != 0:
            x_7 = len(tomato_early_blight)

        count_7 = StringVar()
        count_7.set("The tomato with early blight are {}:".format(x_7))
        label_7 = Label(master=root_1, textvariable=count_7)
        label_7.grid(row=14, column=3)

        x_8 = 0
        if len(tomato_healthy) == 0:
            x_8 = 0
        elif len(tomato_healthy) != 0:
            x_8 = len(tomato_healthy)

        count_8 = StringVar()
        count_8.set("The healthy tomato are {}:".format(x_8))
        label_8 = Label(master=root_1, textvariable=count_8)
        label_8.grid(row=16, column=3)

        x_9 = 0
        if len(tomato_late_blight) == 0:
            x_9 = 0
        elif len(tomato_late_blight) != 0:
            x_9 = len(tomato_late_blight)

        count_9 = StringVar()
        count_9.set("The tomato late blight are {}:".format(x_9))
        label_9 = Label(master=root_1, textvariable=count_9)
        label_9.grid(row=2, column=11)

        x_10 = 0
        if len(tomato_leaf_mold) == 0:
            x_10 = 0
        elif len(tomato_leaf_mold) != 0:
            x_10 = len(tomato_leaf_mold)

        count_10 = StringVar()
        count_10.set("The tomato leaf mold are {}:".format(x_10))
        label_10 = Label(master=root_1, textvariable=count_10)
        label_10.grid(row=4, column=11)

        x_11 = 0
        if len(tomato_septoria_leaf_spot) == 0:
            x_11 = 0
        elif len(tomato_septoria_leaf_spot) != 0:
            x_11 = len(tomato_septoria_leaf_spot)

        count_11 = StringVar()
        count_11.set("The tomato septoria leaf spot are {}:".format(x_11))
        label_11 = Label(master=root_1, textvariable=count_11)
        label_11.grid(row=6, column=11)

        x_12 = 0
        if len(tomato_target_spot) == 0:
            x_12 = 0
        elif len(tomato_target_spot) != 0:
            x_12 = len(tomato_target_spot)

        count_12 = StringVar()
        count_12.set("The tomato target spot are {}:".format(x_12))
        label_12 = Label(master=root_1, textvariable=count_12)
        label_12.grid(row=8, column=11)

        x_13 = 0
        if len(tomato_mosaic_virus) == 0:
            x_13 = 0
        elif len(tomato_mosaic_virus) != 0:
            x_13 = len(tomato_mosaic_virus)

        count_13 = StringVar()
        count_13.set("The tomato mosaic virus are {}:".format(x_13))
        label_13 = Label(master=root_1, textvariable=count_13)
        label_13.grid(row=10, column=11)

        x_14 = 0
        if len(tomato_spider_mites_2_spotted_spider_mite) == 0:
            x_14 = 0
        elif len(tomato_spider_mites_2_spotted_spider_mite) != 0:
            x_14 = len(tomato_spider_mites_2_spotted_spider_mite)

        count_14 = StringVar()
        count_14.set("The tomato spider mites are {}:".format(x_14))
        label_14 = Label(master=root_1, textvariable=count_14)
        label_14.grid(row=12, column=11)

        x_15 = 0
        if len(tomato_yellow_leaf_curl_virus) == 0:
            x_15 = 0
        elif len(tomato_yellow_leaf_curl_virus) != 0:
            x_15 = len(tomato_yellow_leaf_curl_virus)

        count_15 = StringVar()
        count_15.set("The tomato yellow leaf curl virus are {}:".format(x_15))
        label_15 = Label(master=root_1, textvariable=count_15)
        label_15.grid(row=14, column=11)

    btn1 = Button(root_1, text="Pepperbell Bacterial Spot", height=3, width=30)
    btn1.grid(column=2, row=2)
    btn2 = Button(root_1, text="Healthy Pepperbell ", height=3, width=30)
    btn2.grid(column=2, row=4)
    btn3 = Button(root_1, text="Potato early blight ", height=3, width=30)
    btn3.grid(column=2, row=6)
    btn4 = Button(root_1, text="Potato Healthy ", height=3, width=30)
    btn4.grid(column=2, row=8)
    btn5 = Button(root_1, text="Potato late blight ", height=3, width=30)
    btn5.grid(column=2, row=10)
    btn6 = Button(root_1, text="Tomato bacterial spots ", height=3, width=30)
    btn6.grid(column=2, row=12)
    btn7 = Button(root_1, text="Tomato early blight ", height=3, width=30)
    btn7.grid(column=2, row=14)
    btn8 = Button(root_1, text="Tomato Healthy ", height=3, width=30)
    btn8.grid(column=2, row=16)
    btn9 = Button(root_1, text="Tomato late blight ", height=3, width=30)
    btn9.grid(column=10, row=2)
    btn10 = Button(root_1, text="Tomato leaf mold ", height=3, width=30)
    btn10.grid(column=10, row=4)
    btn11 = Button(root_1, text="Tomato septoria leaf spot ", height=3, width=30)
    btn11.grid(column=10, row=6)
    btn12 = Button(root_1, text="Tomato target spot ", height=3, width=30)
    btn12.grid(column=10, row=8)
    btn13 = Button(root_1, text="Tomato mosaic virus ", height=3, width=30)
    btn13.grid(column=10, row=10)
    btn14 = Button(root_1, text="Tomato spider mites ", height=3, width=30)
    btn14.grid(column=10, row=12)
    btn15 = Button(root_1, text="Tomato yellow leaf curl virus ", height=3, width=30)
    btn15.grid(column=10, row=14)
    btn35 = Button(root_1, text="Show the distribution", command=print_all, height=3, width=30)
    btn35.grid(column=10, row=16)


button_20 = Button(text="User's Dataset Distribution", command=distribution, height=3, width=60)
button_20.grid(row=240, column=200)
lbl2 = Label(master=window, text="New window", height=1, width=40)
lbl2.grid(row=160, column=201)

# Creating function that makes the predictions of dataset and yields the results in buttons


def predictions():
    filename = 'finalized_model.sav'
    loaded_model = pickle.load(open(filename, 'rb'))

    dataset_array = np.asarray(dataset)
    dataset_array = dataset_array / 255.0

    predictions_2 = loaded_model.predict(dataset_array)
    y_pred = [np.argmax(probas) for probas in predictions_2]

    pictures = list()
    for i in range(0, len(y_pred)):
        if y_pred[i] == 0 or y_pred[i] == 5:
            pictures.append(i)

    root_2 = Toplevel()
    root_2.title("Predictions")
    root_2.geometry('700x400')
    root_2.configure(background='seagreen')

    def pred_print():
        x_1 = StringVar()
        x_1.set("The length of your dataset is: {} ".format(len(dataset_array)))
        label_1 = Label(master=root_2, textvariable=x_1)
        label_1.grid(row=2, column=3)

    btn1 = Button(root_2, text="Users Dataset Length", command=pred_print, height=3, width=40)
    btn1.grid(column=2, row=2)

    def pred_print_2():
        x_2 = StringVar()
        x_2.set("Your dataset has {} images with bacterial spot".format(len(pictures)))
        label_2 = Label(master=root_2, textvariable=x_2)
        label_2.grid(row=4, column=3)

    btn2 = Button(root_2, text="How many Images have bacterial spots?", command=pred_print_2,  height=3, width=40)
    btn2.grid(column=2, row=4)

    def pred_print_3():
        percentage = (len(pictures)) / (len(dataset_array)) * 100
        x_3 = StringVar()
        x_3.set("That is {0:.2f} % of your dataset".format(percentage))
        label_3 = Label(master=root_2, textvariable=x_3)
        label_3.grid(row=6, column=3)

    btn3 = Button(root_2, text="Percentage", command=pred_print_3, height=3, width=40)
    btn3.grid(column=2, row=6)

    def pred_print_4():
        for h in range(0, len(pictures)):

            x_20 = StringVar()
            x_20.set("The names of the images with bacterial spots are saved in txt")
            label_4 = Label(master=root_2, textvariable=x_20)
            label_4.grid(row=8, column=3)
            f = open('names_bacterial_spot_images.txt', 'a')
            f.write("The name of the image with bacterial spots is: {}\n".format(images[pictures[h]]))
            f.close()
    btn4 = Button(root_2, text="Names of images containing bacterial spots", command=pred_print_4, height=3, width=40)
    btn4.grid(column=2, row=8)

    healthy = list()
    for k in range(0, len(y_pred)):
        if y_pred[k] == 1 or y_pred[k] == 3 or y_pred[k] == 7:
            healthy.append(k)

    def pred_print_5():
        for t in range(0, len(healthy)):

            x_25 = StringVar()
            x_25.set("The names of the images that are healthy are saved in txt")
            label_5 = Label(master=root_2, textvariable=x_25)
            label_5.grid(row=10, column=3)
            f = open('names_healthy_images.txt', 'a')
            f.write("The name of the image that is healthy is: {}\n".format(images[healthy[t]]))
            f.close()
    btn4 = Button(root_2, text="Names of images with healthy plants", command=pred_print_5, height=3, width=40)
    btn4.grid(column=2, row=10)


button3 = Button(text="Which images contain bacterial spots and which are healthy?",
                 command=predictions, height=3, width=60)
button3.grid(row=160, column=200)
lbl3 = Label(master=window, text="New window", height=1, width=40)
lbl3.grid(row=240, column=201)

# Creating a new pop up window for the Descriptive Statistics


def new_window():
    root = Toplevel()
    root.title("Descriptive Statistics")
    root.geometry('500x400')
    root.configure(background='seagreen')

    btn1 = Button(root, text="Pepperbell Set", command=graph, height=3, width=30)
    btn1.grid(column=4, row=6)
    btn2 = Button(root, text="Potato Set", command=graph1, height=3, width=30)
    btn2.grid(column=8, row=6)
    btn3 = Button(root, text="Tomato Set", command=graph2, height=3, width=30)
    btn3.grid(column=4, row=12)
    btn4 = Button(root, text="Plant Leaf Dataset", command=descrstats, height=3, width=30)
    btn4.grid(column=8, row=12)
    label_120 = Label(master=root, text="Pie charts of the original dataset")
    label_120.grid(row=14, column=4)


Button10 = Button(window, text="Descriptive Statistics", command=new_window, height=3, width=60)
Button10.grid(column=200, row=320)
lbl4 = Label(master=window, text="New window", height=1, width=40)
lbl4.grid(row=320, column=201)

window.mainloop()
