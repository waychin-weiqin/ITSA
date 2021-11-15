import os
filepath = '/home/wei/data/SceneFlow'

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def dataloader(filepath):

    train_left_img = []
    train_right_img = []
    train_disp = []
    test_left_img = []
    test_right_img = []
    test_disp = []
    
    # Monkaa
    monkaa_basepath = os.path.join(filepath, 'Monkaa')
    monkaa_path = os.path.join(monkaa_basepath, 'frames_cleanpass')
    monkaa_disp = os.path.join(monkaa_basepath, 'disparity')

    subdir = os.listdir(monkaa_path)

    for dir in subdir:
        img_dir = os.listdir(os.path.join(monkaa_path, dir, 'left'))

        for im in img_dir:
            if is_image_file(os.path.join(monkaa_path, dir, 'left', im)):
                train_left_img.append(os.path.join(monkaa_path, dir, 'left', im))
            train_disp.append(os.path.join(monkaa_disp, dir, 'left', im).split(".")[0] + '.pfm')

            if is_image_file(os.path.join(monkaa_path, dir, 'right', im)):
                train_right_img.append(os.path.join(monkaa_path, dir, 'right', im))

    # FlyingThings3D
    flying_basepath = os.path.join(filepath, 'FlyingThings3D')
    flying_path = os.path.join(flying_basepath, 'frames_cleanpass')
    flying_disp = os.path.join(flying_basepath, 'disparity')

    subdir = ['A', 'B', 'C']

    # Load TRAIN data
    flying_dir = os.path.join(flying_path, 'TRAIN')
    for ss in subdir:
        im_folders = os.listdir(os.path.join(flying_dir, ss))

        for idx in im_folders:
            im_dir = os.listdir(os.path.join(flying_dir, ss, idx, 'left'))

            for im in im_dir:
                if is_image_file(os.path.join(flying_dir, ss, idx, 'left', im)):
                    train_left_img.append(os.path.join(flying_dir, ss, idx, 'left', im))

                train_disp.append(os.path.join(flying_disp, 'TRAIN', ss, idx, 'left', im).split(".")[0] + '.pfm')

                if is_image_file(os.path.join(flying_dir, ss, idx, 'right', im)):
                    train_right_img.append(os.path.join(flying_dir, ss, idx, 'right', im))
                    
    # Driving
    # No test image
    driving_basepath = os.path.join(filepath, 'Driving')
    driving_path = os.path.join(driving_basepath, 'frames_cleanpass')
    driving_disp = os.path.join(driving_basepath, 'disparity')

    subdir1 = os.listdir(driving_path)
    subdir2 = os.listdir(os.path.join(driving_path, subdir1[0]))
    subdir3 = os.listdir(os.path.join(driving_path, subdir1[0], subdir2[0]))

    for fc in subdir1:
        for scene in subdir2:
            for speed in subdir3:
                img_dir = os.listdir(os.path.join(driving_path, fc, scene, speed, 'left'))
    
                for im in img_dir:
                    if is_image_file(os.path.join(driving_path, fc, scene, speed, 'left', im)):
                        train_left_img.append(os.path.join(driving_path, fc, scene, speed, 'left', im))
    
                    train_disp.append(os.path.join(driving_disp, fc, scene, speed, 'left', im).split(".")[0]+ '.pfm')
    
                    if is_image_file(os.path.join(driving_path, fc, scene, speed, 'right', im)):
                        train_right_img.append(os.path.join(driving_path, fc, scene, speed, 'right', im))

    # Load TEST data
    flying_dir = os.path.join(flying_path, 'TEST')
    for ss in subdir:
        im_folders = os.listdir(os.path.join(flying_dir, ss))

        for idx in im_folders:
            im_dir = os.listdir(os.path.join(flying_dir, ss, idx, 'left'))

            for im in im_dir:
                if is_image_file(os.path.join(flying_dir, ss, idx, 'left', im)):
                    test_left_img.append(os.path.join(flying_dir, ss, idx, 'left', im))

                test_disp.append(os.path.join(flying_disp, 'TEST', ss, idx, 'left', im).split(".")[0] + '.pfm')

                if is_image_file(os.path.join(flying_dir, ss, idx, 'right', im)):
                    test_right_img.append(os.path.join(flying_dir, ss, idx, 'right', im))



    return train_left_img, train_right_img, train_disp, test_left_img, test_right_img, test_disp

"""
# Monkaa
    monkaa_basepath = os.path.join(filepath, 'Monkaa')
    monkaa_path = os.path.join(monkaa_basepath, 'frames_cleanpass')
    monkaa_disp = os.path.join(monkaa_basepath, 'disparity')

    subdir = os.listdir(monkaa_path)

    for dir in subdir:
        img_dir = os.listdir(os.path.join(monkaa_path, dir, 'left'))

        for im in img_dir:
            if is_image_file(os.path.join(monkaa_path, dir, 'left', im)):
                train_left_img.append(os.path.join(monkaa_path, dir, 'left', im))
            train_disp.append(os.path.join(monkaa_disp, dir, 'left', im).split(".")[0] + '.pfm')

            if is_image_file(os.path.join(monkaa_path, dir, 'right', im)):
                train_right_img.append(os.path.join(monkaa_path, dir, 'right', im))
                    
    # Driving
    # No test image
    driving_basepath = os.path.join(filepath, 'Driving')
    driving_path = os.path.join(driving_basepath, 'frames_cleanpass')
    driving_disp = os.path.join(driving_basepath, 'disparity')

    subdir1 = os.listdir(driving_path)
    subdir2 = os.listdir(os.path.join(driving_path, subdir1[0]))
    subdir3 = os.listdir(os.path.join(driving_path, subdir1[0], subdir2[0]))

    for fc in subdir1:
        for scene in subdir2:
            for speed in subdir3:
                img_dir = os.listdir(os.path.join(driving_path, fc, scene, speed, 'left'))
    
                for im in img_dir:
                    if is_image_file(os.path.join(driving_path, fc, scene, speed, 'left', im)):
                        train_left_img.append(os.path.join(driving_path, fc, scene, speed, 'left', im))
    
                    train_disp.append(os.path.join(driving_disp, fc, scene, speed, 'left', im).split(".")[0]+ '.pfm')
    
                    if is_image_file(os.path.join(driving_path, fc, scene, speed, 'right', im)):
                        train_right_img.append(os.path.join(driving_path, fc, scene, speed, 'right', im))

"""


