import os
filepath = '/home/wei/data/SceneFlow'

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def dataloader(filepath):

    test_left_img = []
    test_right_img = []
    test_disp = []

    # FlyingThings3D
    flying_basepath = os.path.join(filepath, 'FlyingThings3D')
    flying_path = os.path.join(flying_basepath, 'frames_cleanpass')
    flying_disp = os.path.join(flying_basepath, 'disparity')

    subdir = ['A', 'B', 'C']

    # Load TEST data
    flying_dir = os.path.join(flying_path, 'TEST')
    ss = 'A' # Only load test data from C for validation. The rest use for testing. 
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


    return test_left_img, test_right_img, test_disp


