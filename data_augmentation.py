import nutszebra_data_augmentation_picture


class DataAugmentationNormalizeBigger(object):

    def __init__(self):
        self.da = nutszebra_data_augmentation_picture.DataAugmentationPicture()

    def train(self, img):
        self.da()
        self.da.load_picture(img).resize_image_randomly(1.0, size_range=(356, 512)).crop_picture_randomly(1.0, sizes=(320, 320)).scale_to_one(1.0, constant=255.).fixed_normalization(1.0, each_rgb=True).fixed_color_normalization(1.0, alphastd=0.1, eigval=(0.2175, 0.0188, 0.0045), eigvec=((-0.5675, 0.7192, 0.4009), (-0.5808, -0.0045, -0.8140), (-0.5836, -0.6948, 0.4203))).horizontal_flipping(0.5).convert_to_chainer_format(1.0)
        return self.da.x, self.da.info

    def test(self, img):
        self.da()
        self.da.load_picture(img).resize_image_randomly(1.0, size_range=(434, 434), interpolation='bicubic').crop_center(sizes=(320, 320)).scale_to_one(1.0, constant=255.).fixed_normalization(1.0, each_rgb=True).convert_to_chainer_format(1.0)
        return self.da.x, self.da.info


class Multicrop100(object):

    def __init__(self):
        self.da = nutszebra_data_augmentation_picture.DataAugmentationPicture()

    def test(self, img):
        self.da()
        self.da.load_picture(img).resize_image_randomly(1.0, size_range=(434, 434), interpolation='bicubic').crop_picture_randomly(1.0, sizes=(320, 320)).scale_to_one(1.0, constant=255.).fixed_normalization(1.0, each_rgb=True).horizontal_flipping(0.5).convert_to_chainer_format(1.0)
        return self.da.x, self.da.info


class Multicrop101(object):

    def __init__(self):
        self.da = nutszebra_data_augmentation_picture.DataAugmentationPicture()

    def test(self, img):
        self.da()
        self.da.load_picture(img).resize_image_randomly(1.0, size_range=(512, 512), interpolation='bicubic').crop_picture_randomly(1.0, sizes=(320, 320)).scale_to_one(1.0, constant=255.).fixed_normalization(1.0, each_rgb=True).horizontal_flipping(0.5).convert_to_chainer_format(1.0)
        return self.da.x, self.da.info


class Multicrop102(object):

    def __init__(self):
        self.da = nutszebra_data_augmentation_picture.DataAugmentationPicture()

    def test(self, img):
        self.da()
        self.da.load_picture(img).resize_image_randomly(1.0, size_range=(356, 356), interpolation='bicubic').crop_picture_randomly(1.0, sizes=(320, 320)).scale_to_one(1.0, constant=255.).fixed_normalization(1.0, each_rgb=True).horizontal_flipping(0.5).convert_to_chainer_format(1.0)
        return self.da.x, self.da.info


class Multicrop1(object):

    def __init__(self):
        self.da = nutszebra_data_augmentation_picture.DataAugmentationPicture()

    def test(self, img):
        self.da()
        self.da.load_picture(img).resize_image_randomly(1.0, size_range=(434, 434), interpolation='bicubic').crop_center(sizes=(320, 320)).scale_to_one(1.0, constant=255.).fixed_normalization(1.0, each_rgb=True).convert_to_chainer_format(1.0)
        return self.da.x, self.da.info


class Multicrop2(object):

    def __init__(self):
        self.da = nutszebra_data_augmentation_picture.DataAugmentationPicture()

    def test(self, img):
        self.da()
        self.da.load_picture(img).resize_image_randomly(1.0, size_range=(434, 434), interpolation='bicubic').crop_center(sizes=(320, 320)).scale_to_one(1.0, constant=255.).fixed_normalization(1.0, each_rgb=True).horizontal_flipping(1.0).convert_to_chainer_format(1.0)
        return self.da.x, self.da.info


class Multicrop3(object):

    def __init__(self):
        self.da = nutszebra_data_augmentation_picture.DataAugmentationPicture()

    def test(self, img):
        self.da()
        self.da.load_picture(img).resize_image_randomly(1.0, size_range=(434, 434), interpolation='bicubic').crop_top_left(sizes=(320, 320)).scale_to_one(1.0, constant=255.).fixed_normalization(1.0, each_rgb=True).convert_to_chainer_format(1.0)
        return self.da.x, self.da.info


class Multicrop4(object):

    def __init__(self):
        self.da = nutszebra_data_augmentation_picture.DataAugmentationPicture()

    def test(self, img):
        self.da()
        self.da.load_picture(img).resize_image_randomly(1.0, size_range=(434, 434), interpolation='bicubic').crop_top_left(sizes=(320, 320)).scale_to_one(1.0, constant=255.).fixed_normalization(1.0, each_rgb=True).horizontal_flipping(1.0).convert_to_chainer_format(1.0)
        return self.da.x, self.da.info


class Multicrop5(object):

    def __init__(self):
        self.da = nutszebra_data_augmentation_picture.DataAugmentationPicture()

    def test(self, img):
        self.da()
        self.da.load_picture(img).resize_image_randomly(1.0, size_range=(434, 434), interpolation='bicubic').crop_top_right(sizes=(320, 320)).scale_to_one(1.0, constant=255.).fixed_normalization(1.0, each_rgb=True).convert_to_chainer_format(1.0)
        return self.da.x, self.da.info


class Multicrop6(object):

    def __init__(self):
        self.da = nutszebra_data_augmentation_picture.DataAugmentationPicture()

    def test(self, img):
        self.da()
        self.da.load_picture(img).resize_image_randomly(1.0, size_range=(434, 434), interpolation='bicubic').crop_top_right(sizes=(320, 320)).scale_to_one(1.0, constant=255.).fixed_normalization(1.0, each_rgb=True).horizontal_flipping(1.0).convert_to_chainer_format(1.0)
        return self.da.x, self.da.info


class Multicrop7(object):

    def __init__(self):
        self.da = nutszebra_data_augmentation_picture.DataAugmentationPicture()

    def test(self, img):
        self.da()
        self.da.load_picture(img).resize_image_randomly(1.0, size_range=(434, 434), interpolation='bicubic').crop_bottom_left(sizes=(320, 320)).scale_to_one(1.0, constant=255.).fixed_normalization(1.0, each_rgb=True).convert_to_chainer_format(1.0)
        return self.da.x, self.da.info


class Multicrop8(object):

    def __init__(self):
        self.da = nutszebra_data_augmentation_picture.DataAugmentationPicture()

    def test(self, img):
        self.da()
        self.da.load_picture(img).resize_image_randomly(1.0, size_range=(434, 434), interpolation='bicubic').crop_bottom_left(sizes=(320, 320)).scale_to_one(1.0, constant=255.).fixed_normalization(1.0, each_rgb=True).horizontal_flipping(1.0).convert_to_chainer_format(1.0)
        return self.da.x, self.da.info


class Multicrop9(object):

    def __init__(self):
        self.da = nutszebra_data_augmentation_picture.DataAugmentationPicture()

    def test(self, img):
        self.da()
        self.da.load_picture(img).resize_image_randomly(1.0, size_range=(434, 434), interpolation='bicubic').crop_bottom_right(sizes=(320, 320)).scale_to_one(1.0, constant=255.).fixed_normalization(1.0, each_rgb=True).convert_to_chainer_format(1.0)
        return self.da.x, self.da.info


class Multicrop10(object):

    def __init__(self):
        self.da = nutszebra_data_augmentation_picture.DataAugmentationPicture()

    def test(self, img):
        self.da()
        self.da.load_picture(img).resize_image_randomly(1.0, size_range=(434, 434), interpolation='bicubic').crop_bottom_right(sizes=(320, 320)).scale_to_one(1.0, constant=255.).fixed_normalization(1.0, each_rgb=True).horizontal_flipping(1.0).convert_to_chainer_format(1.0)
        return self.da.x, self.da.info


class Multicrop11(object):

    def __init__(self):
        self.da = nutszebra_data_augmentation_picture.DataAugmentationPicture()

    def test(self, img):
        self.da()
        self.da.load_picture(img).resize_image_randomly(1.0, size_range=(512, 512), interpolation='bicubic').crop_center(sizes=(320, 320)).scale_to_one(1.0, constant=255.).fixed_normalization(1.0, each_rgb=True).convert_to_chainer_format(1.0)
        return self.da.x, self.da.info


class Multicrop12(object):

    def __init__(self):
        self.da = nutszebra_data_augmentation_picture.DataAugmentationPicture()

    def test(self, img):
        self.da()
        self.da.load_picture(img).resize_image_randomly(1.0, size_range=(512, 512), interpolation='bicubic').crop_center(sizes=(320, 320)).scale_to_one(1.0, constant=255.).fixed_normalization(1.0, each_rgb=True).horizontal_flipping(1.0).convert_to_chainer_format(1.0)
        return self.da.x, self.da.info


class Multicrop13(object):

    def __init__(self):
        self.da = nutszebra_data_augmentation_picture.DataAugmentationPicture()

    def test(self, img):
        self.da()
        self.da.load_picture(img).resize_image_randomly(1.0, size_range=(512, 512), interpolation='bicubic').crop_top_left(sizes=(320, 320)).scale_to_one(1.0, constant=255.).fixed_normalization(1.0, each_rgb=True).convert_to_chainer_format(1.0)
        return self.da.x, self.da.info


class Multicrop14(object):

    def __init__(self):
        self.da = nutszebra_data_augmentation_picture.DataAugmentationPicture()

    def test(self, img):
        self.da()
        self.da.load_picture(img).resize_image_randomly(1.0, size_range=(512, 512), interpolation='bicubic').crop_top_left(sizes=(320, 320)).scale_to_one(1.0, constant=255.).fixed_normalization(1.0, each_rgb=True).horizontal_flipping(1.0).convert_to_chainer_format(1.0)
        return self.da.x, self.da.info


class Multicrop15(object):

    def __init__(self):
        self.da = nutszebra_data_augmentation_picture.DataAugmentationPicture()

    def test(self, img):
        self.da()
        self.da.load_picture(img).resize_image_randomly(1.0, size_range=(512, 512), interpolation='bicubic').crop_top_right(sizes=(320, 320)).scale_to_one(1.0, constant=255.).fixed_normalization(1.0, each_rgb=True).convert_to_chainer_format(1.0)
        return self.da.x, self.da.info


class Multicrop16(object):

    def __init__(self):
        self.da = nutszebra_data_augmentation_picture.DataAugmentationPicture()

    def test(self, img):
        self.da()
        self.da.load_picture(img).resize_image_randomly(1.0, size_range=(512, 512), interpolation='bicubic').crop_top_right(sizes=(320, 320)).scale_to_one(1.0, constant=255.).fixed_normalization(1.0, each_rgb=True).horizontal_flipping(1.0).convert_to_chainer_format(1.0)
        return self.da.x, self.da.info


class Multicrop17(object):

    def __init__(self):
        self.da = nutszebra_data_augmentation_picture.DataAugmentationPicture()

    def test(self, img):
        self.da()
        self.da.load_picture(img).resize_image_randomly(1.0, size_range=(512, 512), interpolation='bicubic').crop_bottom_left(sizes=(320, 320)).scale_to_one(1.0, constant=255.).fixed_normalization(1.0, each_rgb=True).convert_to_chainer_format(1.0)
        return self.da.x, self.da.info


class Multicrop18(object):

    def __init__(self):
        self.da = nutszebra_data_augmentation_picture.DataAugmentationPicture()

    def test(self, img):
        self.da()
        self.da.load_picture(img).resize_image_randomly(1.0, size_range=(512, 512), interpolation='bicubic').crop_bottom_left(sizes=(320, 320)).scale_to_one(1.0, constant=255.).fixed_normalization(1.0, each_rgb=True).horizontal_flipping(1.0).convert_to_chainer_format(1.0)
        return self.da.x, self.da.info


class Multicrop19(object):

    def __init__(self):
        self.da = nutszebra_data_augmentation_picture.DataAugmentationPicture()

    def test(self, img):
        self.da()
        self.da.load_picture(img).resize_image_randomly(1.0, size_range=(512, 512), interpolation='bicubic').crop_bottom_right(sizes=(320, 320)).scale_to_one(1.0, constant=255.).fixed_normalization(1.0, each_rgb=True).convert_to_chainer_format(1.0)
        return self.da.x, self.da.info


class Multicrop20(object):

    def __init__(self):
        self.da = nutszebra_data_augmentation_picture.DataAugmentationPicture()

    def test(self, img):
        self.da()
        self.da.load_picture(img).resize_image_randomly(1.0, size_range=(512, 512), interpolation='bicubic').crop_bottom_right(sizes=(320, 320)).scale_to_one(1.0, constant=255.).fixed_normalization(1.0, each_rgb=True).horizontal_flipping(1.0).convert_to_chainer_format(1.0)
        return self.da.x, self.da.info


class Multicrop21(object):

    def __init__(self):
        self.da = nutszebra_data_augmentation_picture.DataAugmentationPicture()

    def test(self, img):
        self.da()
        self.da.load_picture(img).resize_image_randomly(1.0, size_range=(356, 356), interpolation='bicubic').crop_center(sizes=(320, 320)).scale_to_one(1.0, constant=255.).fixed_normalization(1.0, each_rgb=True).convert_to_chainer_format(1.0)
        return self.da.x, self.da.info


class Multicrop22(object):

    def __init__(self):
        self.da = nutszebra_data_augmentation_picture.DataAugmentationPicture()

    def test(self, img):
        self.da()
        self.da.load_picture(img).resize_image_randomly(1.0, size_range=(356, 356), interpolation='bicubic').crop_center(sizes=(320, 320)).scale_to_one(1.0, constant=255.).fixed_normalization(1.0, each_rgb=True).horizontal_flipping(1.0).convert_to_chainer_format(1.0)
        return self.da.x, self.da.info


class Multicrop23(object):

    def __init__(self):
        self.da = nutszebra_data_augmentation_picture.DataAugmentationPicture()

    def test(self, img):
        self.da()
        self.da.load_picture(img).resize_image_randomly(1.0, size_range=(356, 356), interpolation='bicubic').crop_top_left(sizes=(320, 320)).scale_to_one(1.0, constant=255.).fixed_normalization(1.0, each_rgb=True).convert_to_chainer_format(1.0)
        return self.da.x, self.da.info


class Multicrop24(object):

    def __init__(self):
        self.da = nutszebra_data_augmentation_picture.DataAugmentationPicture()

    def test(self, img):
        self.da()
        self.da.load_picture(img).resize_image_randomly(1.0, size_range=(356, 356), interpolation='bicubic').crop_top_left(sizes=(320, 320)).scale_to_one(1.0, constant=255.).fixed_normalization(1.0, each_rgb=True).horizontal_flipping(1.0).convert_to_chainer_format(1.0)
        return self.da.x, self.da.info


class Multicrop25(object):

    def __init__(self):
        self.da = nutszebra_data_augmentation_picture.DataAugmentationPicture()

    def test(self, img):
        self.da()
        self.da.load_picture(img).resize_image_randomly(1.0, size_range=(356, 356), interpolation='bicubic').crop_top_right(sizes=(320, 320)).scale_to_one(1.0, constant=255.).fixed_normalization(1.0, each_rgb=True).convert_to_chainer_format(1.0)
        return self.da.x, self.da.info


class Multicrop26(object):

    def __init__(self):
        self.da = nutszebra_data_augmentation_picture.DataAugmentationPicture()

    def test(self, img):
        self.da()
        self.da.load_picture(img).resize_image_randomly(1.0, size_range=(356, 356), interpolation='bicubic').crop_top_right(sizes=(320, 320)).scale_to_one(1.0, constant=255.).fixed_normalization(1.0, each_rgb=True).horizontal_flipping(1.0).convert_to_chainer_format(1.0)
        return self.da.x, self.da.info


class Multicrop27(object):

    def __init__(self):
        self.da = nutszebra_data_augmentation_picture.DataAugmentationPicture()

    def test(self, img):
        self.da()
        self.da.load_picture(img).resize_image_randomly(1.0, size_range=(356, 356), interpolation='bicubic').crop_bottom_left(sizes=(320, 320)).scale_to_one(1.0, constant=255.).fixed_normalization(1.0, each_rgb=True).convert_to_chainer_format(1.0)
        return self.da.x, self.da.info


class Multicrop28(object):

    def __init__(self):
        self.da = nutszebra_data_augmentation_picture.DataAugmentationPicture()

    def test(self, img):
        self.da()
        self.da.load_picture(img).resize_image_randomly(1.0, size_range=(356, 356), interpolation='bicubic').crop_bottom_left(sizes=(320, 320)).scale_to_one(1.0, constant=255.).fixed_normalization(1.0, each_rgb=True).horizontal_flipping(1.0).convert_to_chainer_format(1.0)
        return self.da.x, self.da.info


class Multicrop29(object):

    def __init__(self):
        self.da = nutszebra_data_augmentation_picture.DataAugmentationPicture()

    def test(self, img):
        self.da()
        self.da.load_picture(img).resize_image_randomly(1.0, size_range=(356, 356), interpolation='bicubic').crop_bottom_right(sizes=(320, 320)).scale_to_one(1.0, constant=255.).fixed_normalization(1.0, each_rgb=True).convert_to_chainer_format(1.0)
        return self.da.x, self.da.info


class Multicrop30(object):

    def __init__(self):
        self.da = nutszebra_data_augmentation_picture.DataAugmentationPicture()

    def test(self, img):
        self.da()
        self.da.load_picture(img).resize_image_randomly(1.0, size_range=(356, 356), interpolation='bicubic').crop_bottom_right(sizes=(320, 320)).scale_to_one(1.0, constant=255.).fixed_normalization(1.0, each_rgb=True).horizontal_flipping(1.0).convert_to_chainer_format(1.0)
        return self.da.x, self.da.info
