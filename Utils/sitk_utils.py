'''
version 8.7.2025
'''
import os
import SimpleITK as sitk
import numpy as np
import nibabel as nib
from .file_and_folder_operations import *
import nrrd
import pydicom
import pandas as pd
import ants

'''
pip install simpleitk pydicom pynrrd pandas
'''

# ------------------------------------
# read / write nrrd
# ------------------------------------
def read_itk(filename):
    nrrd_itk = None
    try:
        nrrd_itk = sitk.ReadImage(filename, )
        # nrrd_itk = sitk.Cast(nrrd_itk, sitk.sitkInt16)
        if 'complex' in nrrd_itk.GetPixelIDTypeAsString() or \
            'vector' in nrrd_itk.GetPixelIDTypeAsString():
            raise Exception("complex image")
    except:
        print(filename, " sitk.ReadImage not work, try nrrd")
        try:
            nrrd_data, nrrd_opt = nrrd.read(filename)
            if len(nrrd_data.shape) == 4:
                if nrrd_opt['kinds'][0]=='RGB-color':
                    nrrd_data = nrrd_data[0].transpose(2, 1, 0)
                elif nrrd_opt['kinds'][0]=='complex':
                    # nrrd_data = nrrd_data[0].astype('int16').transpose(2, 1, 0)
                    nrrd_data = nrrd_data.mean(0).transpose(2, 1, 0)

                elif nrrd_opt['kinds'][0]=='RGBA-color':
                    nrrd_data = nrrd_data.mean(0).transpose(2, 1, 0)
                else:
                    nrrd_data = nrrd_data.mean(0).transpose(2, 1, 0)
            else:
                nrrd_data = nrrd_data.transpose(2, 1, 0)
            nrrd_data = nrrd_data.astype('int16')
            if nrrd_itk:
                new_itk = sitk.GetImageFromArray(nrrd_data)
                new_itk.CopyInformation(nrrd_itk)
            else:
                origin = nrrd_opt['space origin']
                # origin[0:2] = -origin[0:2]
                space_directions = nrrd_opt['space directions'][-3:]
                spacing = np.linalg.norm(space_directions, axis=1, keepdims=True)
                direction = space_directions / spacing*[[-1],[-1],[1]]
                new_itk = sitk.GetImageFromArray(nrrd_data)
                new_itk.SetOrigin(origin)
                new_itk.SetSpacing(spacing.reshape(-1))
                new_itk.SetDirection(direction.T.reshape(-1))
            nrrd_itk = new_itk
        except:
            print(filename, " nrrd not work, try nibabel")
            img = nib.load(filename)
            qform = img.get_qform()
            img.set_qform(qform)
            sfrom = img.get_sform()
            img.set_sform(sfrom)
            nib.save(img, "tmp.nii.gz")
            nrrd_itk = read_itk("tmp.nii.gz")

    if nrrd_itk==None:
        raise RuntimeError('sitkImage read error')
    return nrrd_itk

def write_nrrd(nrrd, outpath, if_compress=True):
    if not os.path.exists(os.path.dirname(outpath)):
        maybe_mkdir_p(os.path.dirname(outpath))
    try:
        nrrd = sitk.Cast(nrrd, sitk.sitkInt16)
    except:
        filter = sitk.VectorIndexSelectionCastImageFilter()
        filter.SetIndex(0)
        filter.SetOutputPixelType(sitk.sitkInt16)
        filter.Execute(nrrd)
    if if_compress:
        sitk.WriteImage(nrrd, outpath, useCompression=True)
    else:
        sitk.WriteImage(nrrd, outpath)

# ------------------------------------
# read / write dicom
# ------------------------------------
def read_dicom(dicom_folder):
    reader = sitk.ImageSeriesReader()
    img_names = reader.GetGDCMSeriesFileNames(dicom_folder)
    reader.SetFileNames(img_names)
    image = reader.Execute()
    return image

def read_dicom_z(dicom_folder):
    dicom_files = subfiles(dicom_folder)
    modal_dicom_files = np.array(dicom_files).tolist()
    modal_dicom_files = z_reorder(modal_dicom_files)
    modal_itk = read_dicoms_from_list(modal_dicom_files)
    return modal_itk

def read_dicoms_from_list(dicom_list):
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(dicom_list)
    image = reader.Execute()
    return image

def maybe_split_dicom(dicom_folder, with_key=[]):
    dicom_index_list, key_values_list = separate_dicoms(dicom_folder, with_key)
    lens = [len(a) for a in dicom_index_list]
    # if dicom separate to multi parts with same slice number
    lens_unq = np.unique(lens)
    if len(lens_unq) == 1 and len(lens)>1 and lens_unq.min() > 10:
        dicom_files = subfiles(dicom_folder)
        itk_list = {}
        for dicom_index, key_value in zip(dicom_index_list, key_values_list):
            modal_dicom_files = np.array(dicom_files)[dicom_index].tolist()
            modal_dicom_files = z_reorder(modal_dicom_files)
            modal_itk = read_dicoms_from_list(modal_dicom_files)
            if len(key_value) > 1:
                itk_list['&'.join(key_value)] = modal_itk
            else:
                itk_list[key_value[0]] = modal_itk
    elif len(lens_unq) > 1 and len(lens)>1 and lens_unq.min() > 10:
        dicom_files = subfiles(dicom_folder)
        itk_list = {}
        for dicom_index, key_value in zip(dicom_index_list, key_values_list):
            modal_dicom_files = np.array(dicom_files)[dicom_index].tolist()
            modal_dicom_files = z_reorder(modal_dicom_files)
            modal_itk = read_dicoms_from_list(modal_dicom_files)
            if len(key_value) > 1:
                itk_list['&'.join(key_value)] = modal_itk
            else:
                itk_list[key_value[0]] = modal_itk
    else:
        print('cannot split docim folder')
        try:
            itk_list = {'no_split': read_dicom_safe(dicom_folder)}
        except:
            itk_list = {'error ': None}
    return itk_list

def read_dicom_safe(dicom_folder, ):
    reader = sitk.ImageSeriesReader()
    img_names = reader.GetGDCMSeriesFileNames(dicom_folder)
        # find group length
    # pydicom_tag = list(pydicom.dcmread(img_names[0]).keys())
    use_tags = ['0008|0000', '0008|0050', '0020|0012', '0008|0008', '0002|0000', '0028|0010', '0028|0011', '0018|0024']
    dicom_metadata = read_dicoms_MetaData(dicom_folder, use_tags)
    dicom_tags = dicom_metadata.keys()
    check = np.ones(len(img_names)).astype(bool)
    for utag in use_tags:
        if utag in dicom_tags:
            indices = np.array(dicom_metadata[utag])
            if len(indices)!=len(img_names):
                continue
            uniq, uniq_nums = np.unique(indices, return_counts=True)
            if len(uniq)>1:
                try:
                    check = check * (indices == uniq[uniq_nums==uniq_nums.max()])
                except:
                    raise "multiple modals with same slice number, need maybe_split_dicom(dicom_folder, with_key=['(0018, 0024)'])"
    img_names = np.array(img_names)[check].tolist()
    reader.SetFileNames(img_names)
    image = reader.Execute()
    return image

def read_dicoms_MetaData(dicom_folder, tags=None):
    reader = sitk.ImageSeriesReader()
    img_names = reader.GetGDCMSeriesFileNames(dicom_folder)
    adict = {}
    if tags==None:
        reader_f = sitk.ImageFileReader()
        reader_f.SetFileName(img_names[0])
        reader_f.LoadPrivateTagsOn()
        reader_f.ReadImageInformation()
        tags = reader_f.GetMetaDataKeys()
    for k in tags:
        content = []
        for n in img_names:
            reader_f = sitk.ImageFileReader()
            reader_f.SetFileName(n)
            reader_f.LoadPrivateTagsOn()
            reader_f.ReadImageInformation()
            try:
                v = reader_f.GetMetaData(k)
                content.append(str(v))
            except:
                break

        adict[str(k)] = content
    return adict

def get_dicom_origin(dicom_list):
    image_file_reader = sitk.ImageFileReader()
    z_list = [ ]
    for file_name in dicom_list:
        image_file_reader.SetFileName(file_name)
        image_file_reader.ReadImageInformation()
        z_list.append(np.array(image_file_reader.GetMetaData("0020|0032").split('\\')).astype('float'))
    z_arr = np.array(z_list)
    return z_arr

# ------------------------------------
# separate multi modal chaotic dicom
# ------------------------------------
def dicom2dict(dicom):
    tmp_dict = {}
    for k in dicom.keys():
        try:
            k_len = len(dicom[k].value)
        except:
            k_len = 0
        if k_len < 50:
            tmp_dict[str(k)] = [str(dicom[k].value)]
    return tmp_dict

def find_most_common_count(dicoms_uniq_df):
    uniq_count_dict = {}
    for uniq_k in dicoms_uniq_df.keys():
        uniq_count_dict[uniq_k] = (dicoms_uniq_df.groupby(uniq_k).size().reset_index(name='Count')['Count'].values).tolist()
    count_df = pd.DataFrame.from_dict({'keys': uniq_count_dict.keys(), 'count': uniq_count_dict.values()}, )
    max_count = count_df['count'].value_counts().idxmax()
    return max_count

def find_most_common_count_keys(dicoms_uniq_df):
    max_count = find_most_common_count(dicoms_uniq_df)
    uniq_count_dict = {}
    for uniq_k in dicoms_uniq_df.keys():
        uniq_count_dict[uniq_k] = str(
            (dicoms_uniq_df.groupby(uniq_k).size().reset_index(name='Count')['Count'].values).tolist())
    count_df = pd.DataFrame.from_dict({'keys': uniq_count_dict.keys(), 'count': uniq_count_dict.values()}, )
    max_count_keys = list(count_df[count_df['count'] == str(max_count)]['keys'])
    return max_count_keys

def z_reorder(modal_dicom_files):
    dicom_origin = get_dicom_origin(modal_dicom_files)
    dicom_origin_z = dicom_origin[:, -1]
    modal_dicom_files = np.array(modal_dicom_files)[np.argsort(dicom_origin_z)].tolist()
    return modal_dicom_files

def print_dicoms(dicom_folder):
    adict = read_dicoms_MetaData(dicom_folder)
    a_df = pd.DataFrame(adict)
    out_csv = 'test.csv'
    a_df.to_csv(out_csv, index=True, sep=',')

def separate_dicoms(dicom_folder, with_key=[]):
    # 0 read dicom to dict
    dicom_files = subfiles(dicom_folder)
    dicom_df_list = []
    for i, dicom_file in enumerate(dicom_files):
        dicom = pydicom.read_file(dicom_file, force=True)
        d_dict = dicom2dict(dicom)
        dicom_df_list.append(pd.DataFrame(d_dict))
    dicoms_df = pd.concat(dicom_df_list, axis=0, join='outer').reset_index(drop=True)

    # calculate unique parameter numbers and filter unique parameter number in [2:4]
    uniq_num = np.array(dicoms_df.nunique(axis=0).array)
    uniq_num_list = np.where((uniq_num < 4) & (uniq_num > 1))[0].tolist()

    if len(uniq_num_list) > 0:
        dicoms_uniq_df = dicoms_df.iloc[:, uniq_num_list]
        # use the unique prams to re-group, calculate number every parameter
        # find most common parameter split
        if len(with_key) > 0:
            max_count_keys = with_key
        else:
            max_count_keys = find_most_common_count_keys(dicoms_uniq_df)
        dicoms_df_grouped = list(dicoms_df.groupby(max_count_keys))
        dicoms_df_grouped_values = [list(df[0]) for df in dicoms_df_grouped]
        dicoms_df_grouped_index = [list(df[1].index) for df in dicoms_df_grouped]
        first_index = [index[0] for index in dicoms_df_grouped_index]
        volume_index = np.argsort(first_index[::-1])
        dicoms_df_grouped_index = np.array(dicoms_df_grouped_index)[volume_index].tolist()
    else:
        dicoms_df_grouped_index = [list(dicoms_df.index)]
        dicoms_df_grouped_values = ['unique']
    return dicoms_df_grouped_index, dicoms_df_grouped_values


# use dicom2nifti


# ------------------------------------
# resample sitk, crop orgain
# ------------------------------------

def get_center(img):
    """
    This function returns the physical center point of a 3d sitk image
    :param img: The sitk image we are trying to find the center of
    :return: The physical center point of the image
    """
    width, height, depth = img.GetSize()
    return img.TransformIndexToPhysicalPoint((int(np.ceil(width/2)),
                                              int(np.ceil(height/2)),
                                              int(np.ceil(depth/2))))

def calmin_max(nonzeros):
    return nonzeros[0].min(), nonzeros[0].max(),\
        nonzeros[1].min(), nonzeros[1].max(),\
        nonzeros[2].min(), nonzeros[2].max(),

def calsize(arr):
    try:
        x1,x2,y1,y2,z1,z2 = calmin_max(arr.nonzero())
    except:
        x1,x2,y1,y2,z1,z2 = calmin_max(sitk.GetArrayFromImage(arr).nonzero())
    return x2-x1, y2-y1, z2-z1

import itertools

def rotation3d(image, theta_x=0, theta_y=0, theta_z=0, output_spacing=None, output_size=None,
               output_direction=None, background_value=0.0, type='img'):
    """
    This function rotates an image across each of the x, y, z axes by theta_x, theta_y, and theta_z degrees
    respectively (euler ZXY orientation) and resamples it to be isotropic.
    :param image: An sitk 3D image
    :param theta_x: The amount of degrees the user wants the image rotated around the x axis
    :param theta_y: The amount of degrees the user wants the image rotated around the y axis
    :param theta_z: The amount of degrees the user wants the image rotated around the z axis
    :param output_spacing: Scalar denoting the isotropic output image spacing. If None, then use the smallest
                           spacing from original image.
                           
    output_size [256, 256, 128] 
    :return: The rotated image
    """
    euler_transform = sitk.Euler3DTransform(
        image.TransformContinuousIndexToPhysicalPoint([(sz - 1) / 2.0 for sz in image.GetSize()]),
        np.deg2rad(theta_x),
        np.deg2rad(theta_y),
        np.deg2rad(theta_z))
    image_center = get_center(image)
    euler_transform.SetCenter(image_center)
    # compute the resampling grid for the transformed image
    max_indexes = [sz - 1 for sz in image.GetSize()]
    extreme_indexes = list(itertools.product(*(list(zip([0] * image.GetDimension(), max_indexes)))))
    extreme_points_transformed = [euler_transform.TransformPoint(image.TransformContinuousIndexToPhysicalPoint(p)) for p
                                  in extreme_indexes]

    output_min_coordinates = np.min(extreme_points_transformed, axis=0)
    output_max_coordinates = np.max(extreme_points_transformed, axis=0)

    # isotropic ouput spacing
    if output_spacing is None:
        output_spacing = min(image.GetSpacing())
        output_spacing = [output_spacing] * image.GetDimension()

    if output_size is None:
        output_origin = output_min_coordinates
        output_size = [round(((omx - omn) / ospc) + 0.6) for ospc, omn, omx in
                       zip(output_spacing, output_min_coordinates, output_max_coordinates)]
    else:
        output_origin = output_min_coordinates
        output_spacing = np.array((output_max_coordinates - output_min_coordinates)) / output_size


    if output_direction is None:
        output_direction = [1, 0, 0, 0, 1, 0, 0, 0, 1]
    output_pixeltype = image.GetPixelIDValue()

    if type=='img':
        interpolation = sitk.sitkLinear
    elif type=='mask':
        interpolation = sitk.sitkNearestNeighbor
        
    return sitk.Resample(image,
                         output_size,
                         euler_transform.GetInverse(),
                         interpolation,
                         output_origin,
                         output_spacing,
                         output_direction,
                         background_value,
                         output_pixeltype)

def flip_itk(nii_itk):
    nii_itk = rotation3d(nii_itk, output_direction=[1, 0, 0, 0, 1, 0, 0, 0, 1])
    nii_itk = sitk.Flip(nii_itk, [False, False, True])
    nii_itk.SetDirection([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
    return nii_itk

def if_overlap(right_itk, maybe_itk):
    try:
        a = right_itk == maybe_itk
        a = sitk.GetArrayFromImage(a)
    except:
        a = np.array(0)
    return a.mean() > 0.95


def sample_ref(process_sitk, target_sitk,  type='img'):
    resampler_img = sitk.ResampleImageFilter()
    resampler_img.SetReferenceImage(target_sitk)
    if type=='img':
        resampler_img.SetInterpolator(sitk.sitkLinear)
        resampler_img.SetOutputPixelType(sitk.sitkInt16)  # -32768 ~ +32767
    elif type=='mask':
        resampler_img.SetInterpolator(sitk.sitkNearestNeighbor)
        resampler_img.SetOutputPixelType(sitk.sitkInt16)
    return resampler_img.Execute(process_sitk)

def itk_crop_foreground(itk_image, thread_value=0, crop_size=[355, 355, 300], if_square=False):
    array = sitk.GetArrayFromImage(itk_image)
    mask = array > thread_value
    
    mask_itk = sitk.GetImageFromArray(mask.astype(np.uint8))
    mask_itk.CopyInformation(itk_image)
    
    cropped_itk_image, cropped_itk_mask = organ_crop(itk_image, mask_itk, crop_size, if_crop_squre=if_square, margin=0, crop_axis=['x', 'y', 'z'])
    return cropped_itk_image

def organ_crop(img_itk, organ_mask_itk, crop_size=[0,0,0], if_crop_squre=True, margin=20, crop_axis=['x', 'y', 'z']):
    crop_size = np.array(crop_size)
    organ_arr = sitk.GetArrayFromImage(organ_mask_itk)
    img_size = img_itk.GetSize()
    nonz = np.nonzero(organ_arr)
    z0,z1, x0,x1, y0,y1 = calmin_max(nonz)
    lung_box = np.array([y0, y1, x0, x1, z0, z1]).reshape(-1,2)
    lung_size = lung_box[:, 1] - lung_box[:, 0]
    bbox_w, bbox_h, bbox_d = lung_size

    out_size = np.max(np.concatenate([crop_size[None], lung_size[None]], axis=0), axis=0)
    out_size = out_size + margin

    if if_crop_squre:
        out_size[:2][out_size[:2]==out_size[:2].min()] = out_size[:2].max()
    d_y0 = int((out_size[0] - bbox_w)/2)
    d_y1 = int(out_size[0] - bbox_w - d_y0)
    d_x0 = int((out_size[1] - bbox_h)/2)
    d_x1 = int(out_size[1] - bbox_h - d_x0)
    d_z0 = int((out_size[2] - bbox_d)/2)
    d_z1 = int(out_size[2] - bbox_d - d_z0)
    lung_box_ext = lung_box + np.array([[-d_y0, d_y1], [-d_x0, d_x1], [-d_z0, d_z1]])

    lung_box_ext[:,0] = np.max(np.array([lung_box_ext[:,0], np.zeros_like(lung_box_ext[:,0])]),axis=0)
    lung_box_ext[:,1] = np.min(np.array([lung_box_ext[:,1], img_size]),axis=0)
    
    if 'y' not in crop_axis:
        lung_box_ext[0,0] = 0
        lung_box_ext[0,1] = img_size[0]
    if 'x' not in crop_axis:
        lung_box_ext[1,0] = 0
        lung_box_ext[1,1] = img_size[1]
    if 'z' not in crop_axis:
        lung_box_ext[2,0] = 0
        lung_box_ext[2,1] = img_size[2]
    
    lung_box_size = lung_box_ext[:,1] - lung_box_ext[:,0]
    lung_box_begin = lung_box_ext[:,0]

    img_extract_itk = sitk.Extract(
        img_itk,
        lung_box_size.tolist(),
        lung_box_begin.tolist())
    organ_mask_extract_itk = sitk.Extract(
        organ_mask_itk,
        lung_box_size.tolist(),
        lung_box_begin.tolist())
    return img_extract_itk, organ_mask_extract_itk

def max_ConnectedComponen(right_itk):
    right_itk = sitk.ConnectedComponent(right_itk)
    sorted_component_image = sitk.RelabelComponent(right_itk, sortByObjectSize=True)
    right_itk = sorted_component_image == 1
    return right_itk

def dilate_itk(mask_itk, ks = 7):
    mask_itk = sitk.GrayscaleDilate(mask_itk, kernelRadius=(ks, ks, ks),)
    return mask_itk

def erode_itk(mask_itk, ks =3 ):
    mask_itk = sitk.GrayscaleErode(mask_itk, kernelRadius=(ks, ks, ks),)
    return mask_itk
    
def center_crop(img_itk, crop_center=None, crop_size=[5,5,5], if_pad=False):
    h, w, d = img_itk.GetSize()
    if crop_center == None:
        crop_center = np.floor([h/2, w/2, d/2])
    crop_half = np.floor(np.array(crop_size) / 2)
    h_0 = max(0, int(crop_center[0] - crop_half[0]))
    h_1 = min(h, int(crop_center[0] + (crop_size[0] - crop_half[0])))
    w_0 = max(0, int(crop_center[1] - crop_half[1]))
    w_1 = min(w, int(crop_center[1] + (crop_size[1] - crop_half[1])))
    d_0 = max(0, int(crop_center[2] - crop_half[2]))
    d_1 = min(d, int(crop_center[2] + (crop_size[2] - crop_half[2])))
    img_itk = img_itk[h_0:h_1, w_0:w_1, d_0:d_1]
    if if_pad:
        pad_size = crop_size - np.array([h, w, d])
        pad_size = np.max(np.array([pad_size, np.zeros(3)]),axis=0)
        LB_pad = np.floor(pad_size/2).astype(int)
        UB_pad = (pad_size - LB_pad).astype(int)
        img_itk = zero_pad(img_itk, LB_pad, UB_pad)
    return img_itk

# ------------------------------------
# check mask
# ------------------------------------

def check_binary(segitk, mod):
    seg_np = sitk.GetArrayFromImage(segitk)

    uniques = np.unique(seg_np)
    for u in uniques:
        if u not in np.arange(0,23):
            return mod + str(u)+' unexpected label'

def check_tumor(segitk, mod):
    seg_np = sitk.GetArrayFromImage(segitk)

    if seg_np.sum() == 0:
        return mod + ' no tumor'

def check_noimg(segitk, mod):
    seg_np = sitk.GetArrayFromImage(segitk)

    if seg_np.sum() == 0:
        return mod + ' no img'

def check_size(imgsitk, segsitk, mod):
    img_size = imgsitk.GetSize()
    seg_size = segsitk.GetSize()
    return_list = []
    if not img_size == seg_size: return_list.append(mod + ' Size inconsistent')

    img_space = np.int16(np.array(imgsitk.GetSpacing())*10000)
    seg_space = np.int16(np.array(segsitk.GetSpacing())*10000)
    if not (img_space == seg_space).all(): return_list.append(mod + ' Spacing inconsistent')
    img_ori = np.int16(np.array(imgsitk.GetOrigin())*10000)
    seg_ori = np.int16(np.array(segsitk.GetOrigin())*10000)
    if not (img_ori == seg_ori).all(): return_list.append(mod + ' Origin inconsistent')
    img_dir = np.int16(np.array(imgsitk.GetDirection())*10000)
    seg_dir = np.int16(np.array(segsitk.GetDirection())*10000)
    if not (img_dir == seg_dir).all(): return_list.append(mod + ' Direction inconsistent')
    return return_list

def sitk2ants(img_itk):
    img_arr = sitk.GetArrayFromImage(img_itk)
    img_arr = img_arr.astype(float).transpose([2,1,0])
    spacing = img_itk.GetSpacing()
    origin = img_itk.GetOrigin()
    direction = np.array(img_itk.GetDirection()).reshape(3, 3)
    img_ants = ants.from_numpy(img_arr, origin=origin, spacing=spacing, direction=direction,)
    return img_ants

def ants2sitk(ants_img):
    # 将 numpy 数据从 (x, y, z) 转换为 (z, y, x)
    data_np = ants_img.numpy().transpose(2, 1, 0)

    # 创建 SimpleITK 图像
    sitk_img = sitk.GetImageFromArray(data_np)

    # 设置空间信息
    sitk_img.SetSpacing(ants_img.spacing)
    sitk_img.SetOrigin(ants_img.origin)
    sitk_img.SetDirection(tuple(np.ravel(ants_img.direction)))

    return sitk_img
    
    
# def convert_image(input_file_name, output_file_name, new_width=None):
#     try:
#         image_file_reader = sitk.ImageFileReader()
#         # only read DICOM images
#         image_file_reader.SetImageIO('GDCMImageIO')
#         image_file_reader.SetFileName(input_file_name)
#         image_file_reader.ReadImageInformation()
#         image_size = list(image_file_reader.GetSize())
#         if len(image_size) == 3 and image_size[2] == 1:
#             image_size[2] = 0
#         image_file_reader.SetExtractSize(image_size)
#         image = image_file_reader.Execute()
#         if new_width:
#             original_size = image.GetSize()
#             original_spacing = image.GetSpacing()
#             new_spacing = [(original_size[0] - 1) * original_spacing[0]
#                            / (new_width - 1)] * 2
#             new_size = [new_width, int((original_size[1] - 1)
#                                        * original_spacing[1] / new_spacing[1])]
#             image = sitk.Resample(image1=image, size=new_size,
#                                   transform=sitk.Transform(),
#                                   interpolator=sitk.sitkLinear,
#                                   outputOrigin=image.GetOrigin(),
#                                   outputSpacing=new_spacing,
#                                   outputDirection=image.GetDirection(),
#                                   defaultPixelValue=0,
#                                   outputPixelType=image.GetPixelID())
#         # If a single channel image, rescale to [0,255]. Also modify the
#         # intensity values based on the photometric interpretation. If
#         # MONOCHROME2 (minimum should be displayed as black) we don't need to
#         # do anything, if image has MONOCRHOME1 (minimum should be displayed as
#         # white) we flip # the intensities. This is a constraint imposed by ITK
#         # which always assumes MONOCHROME2.
#         if image.GetNumberOfComponentsPerPixel() == 1:
#             image = sitk.RescaleIntensity(image, 0, 255)
#             if image_file_reader.GetMetaData('0028|0004').strip() \
#                     == 'MONOCHROME1':
#                 image = sitk.InvertIntensity(image, maximum=255)
#             image = sitk.Cast(image, sitk.sitkUInt8)
#         sitk.WriteImage(image, output_file_name)
#         return True
#     except BaseException:
#         return False
#
# import multiprocessing
# import functools
# def convert_images(input_file_names, output_file_names, new_width):
#     MAX_PROCESSES = 15
#     with multiprocessing.Pool(processes=MAX_PROCESSES) as pool:
#         return pool.starmap(functools.partial(convert_image,
#                                               new_width=new_width),
#                             zip(input_file_names, output_file_names))
#

def safe_mkdir(path):
    try:
        os.mkdir(path)
    except OSError:
        pass


def read_json(path):
    with open(path) as json_file:
        json_data = json.load(json_file)
    return json_data


def resample_new_spacing(image, target_spacing, interplator=sitk.sitkLinear):
    resample = sitk.ResampleImageFilter()
    input_size = image.GetSize()
    pixel_type = image.GetPixelID()
    input_spacing = image.GetSpacing()
    input_spacing = np.round(input_spacing, 2)
    output_spacing = target_spacing
    output_size = [int(input_spacing[0] / output_spacing[0] * input_size[0]),
                   int(input_spacing[1] / output_spacing[1] * input_size[1]),
                   int(input_spacing[2] / output_spacing[2] * input_size[2])]
    resample.SetInterpolator(interplator)
    resample.SetOutputSpacing(output_spacing)
    resample.SetSize(output_size)
    resample.SetOutputPixelType(pixel_type)
    resample.SetOutputOrigin(image.GetOrigin())
    resample.SetOutputDirection(image.GetDirection())
    return resample.Execute(image)


def resample_new_spacing_save(img_nii_path, new_spacing = 0.7):
    # 读取原始图像
    image = sitk.ReadImage(img_nii_path)

    # 获取原图信息
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()

    # 设置目标体素间距

    # 根据间距缩放比调整图像尺寸
    new_size = [
        int(round(osz * ospc / new_spacing)) 
        for osz, ospc in zip(original_size, original_spacing)
    ]

    # 设置重采样器
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing([new_spacing] * 3)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetTransform(sitk.Transform())  # 使用恒等变换
    resampler.SetDefaultPixelValue(image.GetPixelIDValue())  # 保持原像素类型
    resampler.SetInterpolator(sitk.sitkLinear)  # 或 sitk.sitkNearestNeighbor

    # 执行重采样
    resampled_image = resampler.Execute(image)

    # 保存结果
    new_image_path = img_nii_path.replace('.nii.gz', "_resampled_"+str(new_spacing)+"mm.nii.gz")
    sitk.WriteImage(resampled_image, new_image_path)
    return new_image_path

def window_intensity(image, bounding_box=[0, 1, 0.1, 0.9, 0.1, 0.9], pl=1.0, ph=99.0):
    rescaler = sitk.IntensityWindowingImageFilter()
    nda = sitk.GetArrayFromImage(image)
    a = int(nda.shape[0] * bounding_box[0])
    b = int(nda.shape[0] * bounding_box[1])
    c = int(nda.shape[1] * bounding_box[2])
    d = int(nda.shape[1] * bounding_box[3])
    e = int(nda.shape[2] * bounding_box[4])
    f = int(nda.shape[2] * bounding_box[5])
    val_pl = np.percentile(nda[a:b, c:d, e:f], pl)
    val_ph = np.percentile(nda[a:b, c:d, e:f], ph)
    rescaler.SetWindowMaximum(val_ph)
    rescaler.SetWindowMinimum(val_pl)
    rescaler.SetOutputMaximum(val_ph)
    rescaler.SetOutputMinimum(val_pl)
    return rescaler.Execute(image)


def rescale_zero_one(image):
    image = sitk.RescaleIntensity(sitk.Cast(image, sitk.sitkFloat32), 0, 1)
    return image


def zero_pad(image, LB_pad, UB_pad, pad_value=0):
    """
    对 SimpleITK 图像在 (x,y,z) 方向做常数补零。
    LB_pad/UB_pad: 长度为3的数组(>=0)，分别是下界和上界的填充量。
    """
    LB_pad = np.maximum(0, np.asarray(LB_pad, dtype=int))
    UB_pad = np.maximum(0, np.asarray(UB_pad, dtype=int))
    f = sitk.ConstantPadImageFilter()
    f.SetConstant(pad_value)
    f.SetPadLowerBound(LB_pad.tolist())
    f.SetPadUpperBound(UB_pad.tolist())
    return f.Execute(image)

def crop_roi(image,
             center_ijk,
             patch_size,
             pad_value=0,
             center_order='zyx'):  # 'zyx' 表示传的是 IJK；若传的是 xyz 就改成 'xyz'
    """
    从 image 中以 center_ijk 为中心裁剪固定尺寸 patch。
    超出图像范围部分用 pad_value 补0。

    参数
    ----
    image : SimpleITK.Image
    center_ijk : (cz, cy, cx) 若 center_order='zyx'；或 (cx, cy, cz) 若 center_order='xyz'
    patch_size : (pz, py, px) 对应 IJK；内部会转成 (px, py, pz) 传给 SimpleITK
    pad_value : 常数填充值
    center_order : 'zyx' 或 'xyz'

    返回
    ----
    patch_img : SimpleITK.Image (固定尺寸)
    """

    # SimpleITK 用 (x,y,z)
    img_size_xyz = np.asarray(image.GetSize(), dtype=int)  # (x,y,z)

    # 把传入的中心/尺寸转换成 xyz 顺序
    if center_order.lower() == 'zyx':
        center_xyz = np.asarray(center_ijk)[::-1].astype(int)
        size_xyz   = np.asarray(patch_size)[::-1].astype(int)
    else:  # 已经是 xyz
        center_xyz = np.asarray(center_ijk, dtype=int)
        size_xyz   = np.asarray(patch_size, dtype=int)

    # 一半尺寸（向下取整），保证最终 size 固定
    half = size_xyz // 2

    # 以“左开右闭”的思路计算起止（用于判定是否需要 pad）
    start = center_xyz - half                   # (x,y,z)
    end   = start + size_xyz                    # 可能越界（右侧“终止索引+1”）

    # 每维需要的上下补零量
    LB_pad = np.maximum(0, -start)              # 起点<0 -> 需要在下边补
    UB_pad = np.maximum(0, end - img_size_xyz)  # 终点>size -> 在上边补

    # 若需要补零，先 pad，再把 start 右移 LB_pad
    if (LB_pad > 0).any() or (UB_pad > 0).any():
        image = zero_pad(image, LB_pad, UB_pad, pad_value=pad_value)
        start = start + LB_pad
        # pad 后图像尺寸也变化了（不过 start 已修正，下面直接用即可）

    # 现在一定能在图像内直接裁出 size_xyz 大小的 ROI
    start = np.maximum(start, 0).astype(int)
    size_xyz = size_xyz.astype(int)

    # 裁剪（SimpleITK 接受 xyz 顺序）
    patch_img = sitk.RegionOfInterest(image,
                                      size=size_xyz.tolist(),
                                      index=start.tolist())

    return patch_img