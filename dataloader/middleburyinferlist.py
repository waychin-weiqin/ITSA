import os
filepath = '/home/wei/data2/Dataset/Middlebury/'

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def dataloader(filepath, test=False):
    test_left_img = []
    test_right_img = []
    test_disp = []
    test_mask = []
    if test:
      filepath = os.path.join(filepath, "test")
    else:
      filepath = os.path.join(filepath, "training")
    subdir = os.listdir(filepath)

    if test:
      # Load TEST data
      for ss in subdir:
          test_left_img.append(os.path.join(filepath, ss, 'im0.png'))
          test_right_img.append(os.path.join(filepath, ss, 'im1.png'))
      return test_left_img, test_right_img
      
    else:
      # Load TEST data
      for ss in subdir:
          test_left_img.append(os.path.join(filepath, ss, 'im0.png'))
          test_right_img.append(os.path.join(filepath, ss, 'im1.png'))
          test_disp.append(os.path.join(filepath, ss, 'disp0GT.pfm'))
          test_mask.append(os.path.join(filepath, ss, 'mask0nocc.png'))
  
      return test_left_img, test_right_img, test_disp, test_mask
