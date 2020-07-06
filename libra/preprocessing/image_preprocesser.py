import os
import shutil
import cv2
import glob
import pandas as pd
from libra.data_generation.dataset_labelmatcher import get_similar_column
from libra.data_generation.grammartree import get_value_instruction

# Preprocesses images from images to median of heighs/widths


def setwise_preprocessing(data_path, new_folder, height, width):
    training_path = data_path + "/training_set"
    testing_path = data_path + "/testing_set"

    heights = []
    widths = []

    # first dictionary is training and second is testing set
    paths = [training_path, testing_path]
    dict = []
    data_size = []
    num_classes = [0, 0]
    for index in range(2):
        info = process_class_folders(paths[index])
        data_size.append(len(info[0]))
        heights += info[0]
        widths += info[1]
        if info[2] < 2:
            raise BaseException(
                f"Directory: {paths[index]} contains {info[2]} classes. Need at least two classification folders.")
        num_classes[index] += info[2]
        for key, value in info[3].items():
            if len(value) == 0:
                raise BaseException(
                    f"Class: {key} contans {len(value)} images. Need at least one image in this class.")
        dict.append(info[3])

    if num_classes[0] != num_classes[1]:
        raise BaseException(
            f"Number of classes in testing_set and training_set are not equal. Training set has {num_classes[0]} and testing set has {num_classes[1]}")

    classification = num_classes[0]
    height1, width1 = calculate_medians(heights, widths)
    if height is None:
        height = height1
    if width is None:
        width = width1

    # resize images
    for index, p in enumerate(paths):
        for class_folder, images in dict[index].items():
            for image_name, image in images.items():
                dict[index][class_folder][image_name] = process_color_channel(
                    image, height, width)
        if new_folder:
            folder_names = ["proc_training_set", "proc_testing_set"]
            create_folder(data_path, folder_names[index])
            for class_folder, images in dict[index].items():
                add_resized_images(
                    data_path + "/" + folder_names[index],
                    class_folder,
                    images)
        else:
            for class_folder, images in dict[index].items():
                replace_images(p + "/" + class_folder, images)

    return {"num_categories": classification,
            "height": height,
            "width": width,
            "train_size": data_size[0],
            "test_size": data_size[1]}


# processes a csv_file containing image paths and creates a testing/training
# folders with resized images inside the same directory as the csv file
def csv_preprocessing(csv_file,
                      data_path,
                      instruction,
                      image_column,
                      training_ratio,
                      height,
                      width):

    df = pd.read_csv(csv_file)
    if instruction is None:
        raise BaseException(
            "Instruction was not given for csv file to be processed.")

    label = get_similar_column(get_value_instruction(instruction), df)
    avoid_directories = ["proc_training_set", "proc_testing_set"]
    data_paths = [
        data_path +
        "/" +
        d for d in os.listdir(data_path) if os.path.isdir(
            data_path +
            "/" +
            d) and d not in avoid_directories]

    file_extensions = ["jpg", "jpeg", "png", "gif"]
    need_file_extension = False
    path_included = False

    count = 0
    while image_column is None:
        if count > 20:
            raise BaseException(
                f"Could not locate column containing image information.")
        count += 1
        random_row = df.sample()
        for column, value in random_row.iloc[0].items():
            if isinstance(value, str):
                if os.path.exists(data_path + "/" +
                                  (value if value[0] != "/" else value[1:])):
                    path_included = True
                    image_column = column
                    break
                # add file extension if not included
                if value.split(".")[-1] in file_extensions:
                    file = [value]
                else:
                    file = []
                    for extension in file_extensions:
                        file.append(value + "." + extension)

                # look through all data_paths for file
                for path in data_paths:
                    for file_option in file:
                        if os.path.exists(path + "/" + file_option):
                            if file_option.split(".")[-1] in file_extensions:
                                need_file_extension = True
                            image_column = column
                            break
            if image_column is not None:
                break

    else:
        if os.path.exists(data_path + "/" + df.iloc[0][image_column]):
            path_included = True
        elif df.iloc[0][image_column].split(".")[-1]:
            need_file_extension = True

    df = df[[image_column, label]].dropna()

    heights = []
    widths = []
    classifications = df[label].value_counts()
    if len(classifications) < 2:
        raise BaseException(
            f"{csv_file} contains {len(classifications)} classes. Need at least two classification labels.")
    for key, value in classifications.items():
        if value < 2:
            raise BaseException(
                f"Class: {key} contans {value} images. Need at least two images in this class.")

    image_list = []

    # get the median heights and widths
    for index, row in df.iterrows():
        if path_included:
            p = data_path + "/" + \
                (row[image_column][1:] if row[image_column][0] == "/" else row[image_column])
            img = cv2.imread(p)
        else:
            for path in data_paths:
                if need_file_extension:
                    for extension in file_extensions:
                        p = path + "/" + row[image_column] + "." + extension
                        img = cv2.imread(p)
                        if img is not None:
                            break
                else:
                    p = path + "/" + row[image_column]
                    img = cv2.imread(p)
                if img is not None:
                    break
        if img is None:
            raise BaseException(
                f"{row[image_column]} could not be found in any directories.")
        image_list.append(img)
        heights.append(img.shape[0])
        widths.append(img.shape[1])

    height1, width1 = calculate_medians(heights, widths)
    if height is None:
        height = height1
    if width is None:
        width = width1

    # create training and testing folders
    create_folder(data_path, "proc_training_set")
    create_folder(data_path, "proc_testing_set")
    # create classification folders
    for classification in classifications.keys():
        create_folder(data_path + "/proc_training_set", classification)
        create_folder(data_path + "/proc_testing_set", classification)

    data_size = [0, 0]
    class_count = dict.fromkeys(classifications.keys(), 0)

    # save images into correct folder
    for index, row in df.iterrows():
        # resize images
        img = process_color_channel(image_list[index], height, width)
        p = "proc_" + (os.path.basename(row[image_column])
                       if path_included else row[image_column])
        if need_file_extension:
            p += ".jpg"
        if class_count[row[label]] / \
                classifications[row[label]] < training_ratio:
            data_size[0] += 1
            class_count[row[label]] += 1
            save_image(data_path + "/proc_training_set", img, p, row[label])
        else:
            data_size[1] += 1
            class_count[row[label]] += 1
            save_image(data_path + "/proc_testing_set", img, p, row[label])

    return {"num_categories": len(classifications),
            "height": height,
            "width": width,
            "train_size": data_size[0],
            "test_size": data_size[1]}


# preprocesses images when given a folder containing class folders
def classwise_preprocessing(data_path, training_ratio, height, width):
    info = process_class_folders(data_path)
    if info[2] < 2:
        raise BaseException(
            f"Directory: {data_path} contains {info[2]} classes. Need at least two classification folders.")
    img_dict = info[3]
    for key, value in img_dict.items():
        if len(value) < 2:
            raise BaseException(
                f"Class: {key} contans {len(value)} images. Need at least two images in this class.")
    height1, width1 = calculate_medians(info[0], info[1])
    if height is None:
        height = height1
    if width is None:
        width = width1
    num_classifications = info[2]

    # create training and testing folders
    create_folder(data_path, "proc_training_set")
    create_folder(data_path, "proc_testing_set")

    # create classification folders
    for classification in img_dict.keys():
        create_folder(data_path + "/proc_training_set", classification)
        create_folder(data_path + "/proc_testing_set", classification)

    data_size = [0, 0]
    for class_folder, images in img_dict.items():
        count = 0
        for image_name, image in images.items():
            resized_img = process_color_channel(image, height, width)
            if count / len(images) < training_ratio:
                data_size[0] += 1
                save_image(data_path + "/proc_training_set",
                           resized_img,
                           "proc_" + image_name,
                           class_folder)
            else:
                data_size[1] += 1
                save_image(data_path + "/proc_testing_set",
                           resized_img,
                           "proc_" + image_name,
                           class_folder)
            count += 1

    return {"num_categories": num_classifications,
            "height": height,
            "width": width,
            "train_size": data_size[0],
            "test_size": data_size[1]}


# process a class folder by getting return a list of all the heights and widths
def process_class_folders(data_path):
    heights = []
    widths = []
    num_classifications = 0
    img_dict = {}
    avoid_directories = ["proc_training_set", "proc_testing_set"]

    for class_folder in os.listdir(data_path):
        if not os.path.isdir(
            data_path +
            "/" +
                class_folder) or class_folder in avoid_directories:
            continue

        folder_dict = {}
        folder_height = []
        folder_width = []

        for image in os.listdir(data_path + "/" + class_folder):
            if image == ".DS_Store":
                continue
            try:
                img = cv2.imread(data_path + "/" + class_folder + "/" + image)
                folder_height.append(img.shape[0])
                folder_width.append(img.shape[1])
                folder_dict[image] = img
            except BaseException:
                break
        else:
            heights += folder_height
            widths += folder_width
            img_dict[class_folder] = folder_dict
            num_classifications += 1

    return [heights, widths, num_classifications, img_dict]


# creates a folder with the given folder name and fills the folders
# with the images given
def add_resized_images(data_path, folder_name, images):
    # create processed folder
    os.mkdir(data_path + "/proc_" + folder_name)
    # add images to processed folder
    for img_name, img in images.items():
        cv2.imwrite(
            data_path +
            "/proc_" +
            folder_name +
            "/proc_" +
            img_name,
            img)


# writes an image into the given path
def replace_images(data_path, loaded_shaped):
    for img_name, img in loaded_shaped.items():
        cv2.imwrite(data_path + "/" + img_name, img)


# creates a folder in the given path with the given folder name
def create_folder(path, folder_name):
    folder_name = "/" + folder_name
    if os.path.isdir(path + folder_name):
        shutil.rmtree(path + folder_name)
    os.mkdir(path + folder_name)


# saves an image to the given path inside the classification folder
# specified with the img_name
def save_image(path, img, img_name, classification):
    cv2.imwrite(
        path + "/" + classification + "/" + img_name,
        img)

# calculates the medians of the given lists of height and widths


def calculate_medians(heights, widths):
    heights.sort()
    widths.sort()
    height = heights[int(len(heights) / 2)]
    width = widths[int(len(widths) / 2)]
    return height, width


# resizes the image with the given height and width
def process_color_channel(img, height, width):
    chanels = [chanel for chanel in cv2.split(img)]

    for index, chanel in enumerate(chanels):
        if chanel.shape[0] > height:
            chanel = cv2.resize(
                chanel,
                dsize=(
                    chanel.shape[1],
                    height),
                interpolation=cv2.INTER_CUBIC)
        else:
            chanel = cv2.resize(
                chanel,
                dsize=(
                    chanel.shape[1],
                    height),
                interpolation=cv2.INTER_AREA)
        if chanel.shape[1] > width:
            chanel = cv2.resize(
                chanel,
                dsize=(
                    width,
                    height),
                interpolation=cv2.INTER_CUBIC)
        else:
            chanel = cv2.resize(
                chanel,
                dsize=(
                    width,
                    height),
                interpolation=cv2.INTER_AREA)
        chanels[index] = chanel

    return cv2.merge(chanels)

# distinguishes the preprocess mode to do


def set_distinguisher(data_path, read_mode):
    if read_mode is not None:
        if read_mode == "setwise":
            if os.path.isdir(data_path +
                             "/training_set") and os.path.isdir(data_path +
                                                                "/testing_set"):
                return {"read_mode": "setwise"}
            raise BaseException(
                f"training_set or testing_set folder not in f{data_path}")
        elif read_mode == "classwise":
            return {"read_mode": "classwise"}
        elif read_mode == "csvwise":
            csv_path = glob.glob(data_path + "/*.csv")
            if len(csv_path) == 1:
                return {
                    "read_mode": "csvwise",
                    "csv_path": csv_path[0]}
            elif len(csv_path) > 1:
                raise BaseException(
                    f"Too many csv files in {data_path}: {[os.path.basename(path) for path in csv_path]}")
            else:
                raise BaseException(f"No csv file in {data_path}")
        else:
            raise BaseException(f"{read_mode}, is an invalid read mode.")

    # check if setwise
    if os.path.isdir(data_path +
                     "/training_set") and os.path.isdir(data_path +
                                                        "/testing_set"):
        return {"read_mode": "setwise"}

    # check if contains a csv file
    csv_path = glob.glob(data_path + "/*.csv")
    if len(csv_path) == 1:
        return {"read_mode": "csvwise", "csv_path": csv_path[0]}
    elif len(csv_path) > 1:
        raise BaseException(
            f"Too many csv files in directory: {[os.path.basename(path) for path in csv_path]}")

    return {"read_mode": "classwise"}


# get the dataset info for the CNN model
def already_processed(data_path):
    training_path = data_path + "/training_set"
    testing_path = data_path + "/testing_set"

    if not os.path.isdir(training_path) or not os.path.isdir(testing_path):
        raise BaseException(
            f"Missing training_set or testing_set folder in directory: {data_path}")

    paths = [training_path, testing_path]
    num_categories = [0, 0]
    sizes = [0, 0]
    for index, path in enumerate(paths):
        for directory in os.listdir(path):
            class_path = path + "/" + directory
            if os.path.isdir(class_path):
                num_categories[index] += 1
                images = [img for img in os.listdir(
                    class_path) if img != ".DS_Store"]
                if len(images) == 0:
                    raise BaseException(
                        f"Class: {directory} in {path} contains no images.")
                height, width, _ = cv2.imread(
                    class_path + "/" + images[0]).shape
                sizes[index] += len(images)
        if num_categories[index] < 2:
            raise BaseException(
                f"Directory: {path} contains {num_categories[index]} classes. Need at least two classification folders.")

    if num_categories[0] != num_categories[1]:
        raise BaseException(
            f"Number of classes in testing_set and training_set are not equal. Training set has {num_categories[0]} and testing set has {num_categories[1]}")

    return {"num_categories": num_categories[0],
            "height": height,
            "width": width,
            "train_size": sizes[0],
            "test_size": sizes[1]}
