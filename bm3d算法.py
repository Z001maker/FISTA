import math
import os

import cv2
import numpy
import numpy as np
from skimage.metrics import structural_similarity as ssim
from common.utils import apply_random_mask, psnr

cv2.setUseOptimized(True)

# 参数初始化
sigma = 25
Threshold_Hard3D = 2.7 * sigma  # 硬阈值
First_Match_threshold = 2500  # 用于计算块之间相似度的阈值
Step1_max_matched_cnt = 16  # 组最大匹配的块数
Step1_Blk_Size = 8  # 块大小
Step1_Blk_Step = 3  # 滑动的步长
Step1_Search_Step = 3  # 块的搜索步长
Step1_Search_Window = 39  # 搜索窗口大小

Second_Match_threshold = 400  # 用于计算块之间相似度的阈值
Step2_max_matched_cnt = 32
Step2_Blk_Size = 8
Step2_Blk_Step = 3
Step2_Search_Step = 3
Step2_Search_Window = 39

Beta_Kaiser = 2.0  # 凯撒窗的参数

def init(img, _blk_size, _Beta_Kaiser):
    """初始化，返回用于记录过滤后图像以及权重的数组,还有构造凯撒窗"""
    m_shape = img.shape
    m_img = np.zeros(m_shape, dtype=float)
    m_wight = np.zeros(m_shape, dtype=float)
    K = np.matrix(np.kaiser(_blk_size, _Beta_Kaiser))
    m_Kaiser = np.array(K.T * K)  # 构造凯撒窗
    return m_img, m_wight, m_Kaiser

def Locate_blk(i, j, blk_step, block_size, width, height):
    """确保当前块不超出图像范围"""
    if i * blk_step + block_size < width:
        point_x = i * blk_step
    else:
        point_x = width - block_size

    if j * blk_step + block_size < height:
        point_y = j * blk_step
    else:
        point_y = height - block_size

    m_blockPoint = np.array((point_x, point_y), dtype=int)
    return m_blockPoint

def Define_SearchWindow(_noisyImg, _BlockPoint, _WindowSize, Blk_Size):
    """定义搜索窗口的顶点坐标"""
    point_x = _BlockPoint[0]
    point_y = _BlockPoint[1]

    LX = point_x + Blk_Size // 2 - _WindowSize // 2
    LY = point_y + Blk_Size // 2 - _WindowSize // 2
    RX = LX + _WindowSize
    RY = LY + _WindowSize

    if LX < 0:
        LX = 0
    elif RX > _noisyImg.shape[0]:
        LX = _noisyImg.shape[0] - _WindowSize
    if LY < 0:
        LY = 0
    elif RY > _noisyImg.shape[1]:
        LY = _noisyImg.shape[1] - _WindowSize

    return np.array((LX, LY), dtype=int)

def Step1_fast_match(_noisyImg, _BlockPoint):
    """快速匹配算法"""
    (present_x, present_y) = _BlockPoint
    Blk_Size = Step1_Blk_Size
    Search_Step = Step1_Search_Step
    Threshold = First_Match_threshold
    max_matched = Step1_max_matched_cnt
    Window_size = Step1_Search_Window

    blk_positions = np.zeros((max_matched, 2), dtype=int)
    Final_similar_blocks = np.zeros((max_matched, Blk_Size, Blk_Size), dtype=float)

    img = _noisyImg[present_x: present_x + Blk_Size, present_y: present_y + Blk_Size]
    if img.shape[0] != Blk_Size or img.shape[1] != Blk_Size:
        raise ValueError("Block size mismatch: expected {}x{}".format(Blk_Size, Blk_Size))

    dct_img = cv2.dct(img.astype(np.float64))
    Final_similar_blocks[0, :, :] = dct_img
    blk_positions[0, :] = _BlockPoint

    Window_location = Define_SearchWindow(_noisyImg, _BlockPoint, Window_size, Blk_Size)
    blk_num = (Window_size - Blk_Size) // Search_Step
    blk_num = int(blk_num)
    (present_x, present_y) = Window_location

    similar_blocks = np.zeros((blk_num ** 2, Blk_Size, Blk_Size), dtype=float)
    m_Blkpositions = np.zeros((blk_num ** 2, 2), dtype=int)
    Distances = np.zeros(blk_num ** 2, dtype=float)

    matched_cnt = 0
    for i in range(blk_num):
        for j in range(blk_num):
            tem_img = _noisyImg[present_x: present_x + Blk_Size, present_y: present_y + Blk_Size]
            if tem_img.shape[0] != Blk_Size or tem_img.shape[1] != Blk_Size:
                continue
            dct_Tem_img = cv2.dct(tem_img.astype(np.float64))
            m_Distance = np.linalg.norm((dct_img - dct_Tem_img)) ** 2 / (Blk_Size ** 2)

            if m_Distance < Threshold and m_Distance > 0:
                similar_blocks[matched_cnt, :, :] = dct_Tem_img
                m_Blkpositions[matched_cnt, :] = (present_x, present_y)
                Distances[matched_cnt] = m_Distance
                matched_cnt += 1
            present_y += Search_Step
        present_x += Search_Step
        present_y = Window_location[1]
    Distances = Distances[:matched_cnt]
    Sort = np.argsort(Distances)

    if matched_cnt < max_matched:
        Count = matched_cnt + 1
    else:
        Count = max_matched

    if Count > 0:
        for i in range(1, Count):
            similar_blocks[i, :, :] = similar_blocks[Sort[i - 1], :, :]
            blk_positions[i, :] = m_Blkpositions[Sort[i - 1], :]
    return Final_similar_blocks, blk_positions, Count

def Step1_3DFiltering(_similar_blocks):
    """3D变换及滤波处理"""
    statis_nonzero = 0
    m_Shape = _similar_blocks.shape

    for i in range(m_Shape[1]):
        for j in range(m_Shape[2]):
            tem_Vct_Trans = cv2.dct(_similar_blocks[:, i, j])
            tem_Vct_Trans[np.abs(tem_Vct_Trans) < Threshold_Hard3D] = 0.
            statis_nonzero += np.sum(np.abs(tem_Vct_Trans) > Threshold_Hard3D)
            _similar_blocks[:, i, j] = cv2.idct(tem_Vct_Trans)[0]
    return _similar_blocks, statis_nonzero

def Aggregation_hardthreshold(_similar_blocks, blk_positions, m_basic_img, m_wight_img, _nonzero_num, Count, Kaiser):
    """加权累加并应用硬阈值"""
    _shape = _similar_blocks.shape
    if _nonzero_num < 1:
        _nonzero_num = 1
    block_wight = (1. / _nonzero_num) * Kaiser
    for i in range(Count):
        point = blk_positions[i, :]
        tem_img = (1. / _nonzero_num) * cv2.idct(_similar_blocks[i, :, :]) * Kaiser
        m_basic_img[point[0]: point[0] + _shape[1], point[1]: point[1] + _shape[2]] += tem_img
        m_wight_img[point[0]: point[0] + _shape[1], point[1]: point[1] + _shape[2]] += block_wight

def BM3D_1st_step(_noisyImg):
    """第一步：基本去噪"""
    width, height = _noisyImg.shape
    block_Size = Step1_Blk_Size
    blk_step = Step1_Blk_Step
    width_num = (width - block_Size) // blk_step + 2
    height_num = (height - block_Size) // blk_step + 2

    Basic_img, m_Wight, m_Kaiser = init(_noisyImg, block_Size, Beta_Kaiser)

    for i in range(int(width_num)):
        for j in range(int(height_num)):
            m_blockPoint = Locate_blk(i, j, blk_step, block_Size, width, height)
            Similar_Blks, Positions, Count = Step1_fast_match(_noisyImg, m_blockPoint)
            Similar_Blks, statis_nonzero = Step1_3DFiltering(Similar_Blks)
            Aggregation_hardthreshold(Similar_Blks, Positions, Basic_img, m_Wight, statis_nonzero, Count, m_Kaiser)
    Basic_img[:, :] /= m_Wight[:, :]
    return Basic_img

def Step2_fast_match(_basicImg, _noisyImg, _BlockPoint):
    """快速匹配算法"""
    (present_x, present_y) = _BlockPoint
    Blk_Size = Step2_Blk_Size
    Search_Step = Step2_Search_Step
    Threshold = Second_Match_threshold
    max_matched = Step2_max_matched_cnt
    Window_size = Step2_Search_Window

    blk_positions = np.zeros((max_matched, 2), dtype=int)
    Final_similar_blocks = np.zeros((max_matched, Blk_Size, Blk_Size), dtype=float)
    Final_noisy_blocks = np.zeros((max_matched, Blk_Size, Blk_Size), dtype=float)

    img = _basicImg[present_x: present_x + Blk_Size, present_y: present_y + Blk_Size]
    if img.shape[0] != Blk_Size or img.shape[1] != Blk_Size:
        raise ValueError("Block size mismatch: expected {}x{}".format(Blk_Size, Blk_Size))

    dct_img = cv2.dct(img.astype(np.float32))
    Final_similar_blocks[0, :, :] = dct_img

    n_img = _noisyImg[present_x: present_x + Blk_Size, present_y: present_y + Blk_Size]
    dct_n_img = cv2.dct(n_img.astype(np.float32))
    Final_noisy_blocks[0, :, :] = dct_n_img

    blk_positions[0, :] = _BlockPoint

    Window_location = Define_SearchWindow(_noisyImg, _BlockPoint, Window_size, Blk_Size)
    blk_num = (Window_size - Blk_Size) // Search_Step
    blk_num = int(blk_num)
    (present_x, present_y) = Window_location

    similar_blocks = np.zeros((blk_num ** 2, Blk_Size, Blk_Size), dtype=float)
    m_Blkpositions = np.zeros((blk_num ** 2, 2), dtype=int)
    Distances = np.zeros(blk_num ** 2, dtype=float)

    matched_cnt = 0
    for i in range(blk_num):
        for j in range(blk_num):
            tem_img = _basicImg[present_x: present_x + Blk_Size, present_y: present_y + Blk_Size]
            if tem_img.shape[0] != Blk_Size or tem_img.shape[1] != Blk_Size:
                continue
            dct_Tem_img = cv2.dct(tem_img.astype(np.float32))
            m_Distance = np.linalg.norm((dct_img - dct_Tem_img)) ** 2 / (Blk_Size ** 2)

            if m_Distance < Threshold and m_Distance > 0:
                similar_blocks[matched_cnt, :, :] = dct_Tem_img
                m_Blkpositions[matched_cnt, :] = (present_x, present_y)
                Distances[matched_cnt] = m_Distance
                matched_cnt += 1
            present_y += Search_Step
        present_x += Search_Step
        present_y = Window_location[1]
    Distances = Distances[:matched_cnt]
    Sort = np.argsort(Distances)

    if matched_cnt < max_matched:
        Count = matched_cnt + 1
    else:
        Count = max_matched

    if Count > 0:
        for i in range(1, Count):
            similar_blocks[i, :, :] = similar_blocks[Sort[i - 1], :, :]
            blk_positions[i, :] = m_Blkpositions[Sort[i - 1], :]
            n_img = _noisyImg[present_x: present_x + Blk_Size, present_y: present_y + Blk_Size]
            Final_noisy_blocks[i, :, :] = cv2.dct(n_img.astype(np.float64))
    return Final_similar_blocks, Final_noisy_blocks, blk_positions, Count

def Step2_3DFiltering(_Similar_Bscs, _Similar_Imgs):
    """3D维纳变换的协同滤波"""
    m_Shape = _Similar_Bscs.shape
    Wiener_wight = np.zeros((m_Shape[1], m_Shape[2]), dtype=float)

    for i in range(m_Shape[1]):
        for j in range(m_Shape[2]):
            tem_vector = _Similar_Bscs[:, i, j]
            tem_Vct_Trans = np.matrix(cv2.dct(tem_vector))
            Norm_2 = np.float64(np.sum(tem_Vct_Trans.T * tem_Vct_Trans))
            m_weight = Norm_2 / (Norm_2 + sigma ** 2)
            if m_weight != 0:
                Wiener_wight[i, j] = 1. / (m_weight ** 2 * sigma ** 2)
            else:
                Wiener_wight[i, j] = 10000  # 防止除以零

            tem_vector = _Similar_Imgs[:, i, j]
            tem_Vct_Trans = m_weight * cv2.dct(tem_vector)
            _Similar_Bscs[:, i, j] = cv2.idct(tem_Vct_Trans)[0]

    return _Similar_Bscs, Wiener_wight

def Aggregation_Wiener(_Similar_Blks, _Wiener_wight, blk_positions, m_basic_img, m_wight_img, Count, Kaiser):
    """加权累加并应用维纳滤波"""
    _shape = _Similar_Blks.shape
    block_wight = _Wiener_wight  # * Kaiser

    for i in range(Count):
        point = blk_positions[i, :]
        tem_img = _Wiener_wight * cv2.idct(_Similar_Blks[i, :, :])  # * Kaiser
        m_basic_img[point[0]: point[0] + _shape[1], point[1]: point[1] + _shape[2]] += tem_img
        m_wight_img[point[0]: point[0] + _shape[1], point[1]: point[1] + _shape[2]] += block_wight

def BM3D_2nd_step(_basicImg, _noisyImg):
    """第二步协同滤波和最终估计"""
    width, height = _noisyImg.shape
    block_Size = Step2_Blk_Size
    blk_step = Step2_Blk_Step
    width_num = (width - block_Size) // blk_step + 1
    height_num = (height - block_Size) // blk_step + 1

    m_img, m_Wight, m_Kaiser = init(_noisyImg, block_Size, Beta_Kaiser)

    for i in range(int(width_num)):
        for j in range(int(height_num)):
            m_blockPoint = Locate_blk(i, j, blk_step, block_Size, width, height)
            Similar_Blks, Similar_Imgs, Positions, Count = Step2_fast_match(_basicImg, _noisyImg, m_blockPoint)
            Similar_Blks, Wiener_wight = Step2_3DFiltering(Similar_Blks, Similar_Imgs)
            Aggregation_Wiener(Similar_Blks, Wiener_wight, Positions, m_img, m_Wight, Count, m_Kaiser)
    m_img[:, :] /= m_Wight[:, :]
    return m_img

def calculate_psnr(original, denoised):
    """计算PSNR值"""
    mse = np.mean((original - denoised) ** 2)
    if mse == 0:
        return 100  # 两张图完全相同
    return 20 * math.log10(255.0 / math.sqrt(mse))

def calculate_ssim(original, denoised):
    """计算SSIM值"""
    return ssim(original, denoised, data_range=denoised.max() - denoised.min())

if __name__ == '__main__':
    cv2.setUseOptimized(True)
    img_name = "../0.3_noisy_image.png"

    # 检查文件是否存在
    if not os.path.exists(img_name):
        print(f"Error: File '{img_name}' does not exist.")
    else:
        print(f"File '{img_name}' exists.")

        # 尝试加载图像
        img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Error: Unable to load image. Please check the file path.")
    else:
        # 确保图像大小是块大小的整数倍
        height, width = img.shape
        if height % Step1_Blk_Size != 0 or width % Step1_Blk_Size != 0:
            new_height = (height // Step1_Blk_Size) * Step1_Blk_Size
            new_width = (width // Step1_Blk_Size) * Step1_Blk_Size
            img = cv2.resize(img, (new_width, new_height))

        e1 = cv2.getTickCount()
        Basic_img = BM3D_1st_step(img)
        e2 = cv2.getTickCount()
        time = (e2 - e1) / cv2.getTickFrequency()
        print("The Processing time of the First step is %.2f s" % time)

        # 保存图像
        cv2.imwrite("First2.jpg", Basic_img)

        # 第二步
        Final_img = BM3D_2nd_step(Basic_img, img)
        e3 = cv2.getTickCount()
        time = (e3 - e2) / cv2.getTickFrequency()
        print("The Processing time of the Second step is %.2f s" % time)

        # 保存最终图像
        cv2.imwrite("Final3.jpg", Final_img)

        time = (e3 - e1) / cv2.getTickFrequency()
        print("The total Processing time is %.2f s" % time)

        img_name1 = "../Set14/ppt3.png"
        img1 = cv2.imread(img_name1, cv2.IMREAD_GRAYSCALE)
        # 确保图像的值范围是 [0, 255]，并且数据类型是 uint8
        img1 = cv2.normalize(img1, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # 调整 Final_img 的大小以匹配 img1
        #Final_img = cv2.resize(Final_img, (img1.shape[1], img1.shape[0]))
        Final_img = cv2.resize(Final_img, (img1.shape[1], img1.shape[0]))
        print(f"Final_img shape: {Final_img.shape}")
        print(f"img1 shape: {img1.shape}")
        Final_img = cv2.normalize(Final_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        # 计算PSNR和SSIM
        #psnr_value = calculate_psnr(img1, Final_img)
        psnr_value = psnr(img1, Final_img)
        ssim_value = calculate_ssim(img1, Final_img)

        print(f"PSNR: {psnr_value:.2f} dB")
        print(f"SSIM: {ssim_value:.4f}")