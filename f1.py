import time
from skimage import io

import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.restoration import denoise_tv_chambolle
from skimage.metrics import structural_similarity as ssim, structural_similarity
from skimage.util import img_as_ubyte

from common.utils import apply_random_mask, psnr, load_image, print_progress, print_end_message, print_start_message
from common.operators import TV_norm, RepresentationOperator, p_omega, p_omega_t, l1_prox, norm1, norm2sq



def ISTA(fx, gx, gradf, proxg, params, verbose=False):
    method_name = 'ISTA'
    print_start_message(method_name)
    tic_start = time.time()

    # Initialize parameters.
    x0 = params['x0']
    maxit = params['maxit']
    lmbd = params['lambda']
    alpha = 1 / params['prox_Lips']

    info = {'itertime': np.zeros(maxit), 'fx': np.zeros(maxit), 'iter': maxit}

    # 记录 PSNR 值
    psnr_values = []

    x_k = x0
    for k in range(maxit):
        tic = time.time()

        # Update the iterate
        y = x_k - alpha * gradf(x_k)
        x_k_next = proxg(y, alpha * lmbd)
        x_k = x_k_next

        # 计算 PSNR 值并记录
        reconstruction = x_k.reshape(params['m'], params['m'])
        reconstruction = np.clip(reconstruction, 0, 1)  # 确保值范围在 [0, 1]
        psnr_value = psnr(params['original_image'], reconstruction)
        psnr_values.append(psnr_value)

        # Compute error and save data to be plotted later on.
        info['itertime'][k] = time.time() - tic
        info['fx'][k] = fx(x_k) + lmbd * gx(x_k)
        if k % params['iter_print'] == 0 and verbose:
            print_progress(k, maxit, info['fx'][k], fx(x_k), gx(x_k))

    if verbose:
        print_end_message(method_name, time.time() - tic_start)
    return x_k, info, psnr_values  # 返回 x_k、info 和 psnr_values


def FISTA(fx, gx, gradf, proxg, params, verbose=False):
    method_name = 'common'  # 方法名称
    print_start_message(method_name)
    tic_start = time.time()

    # 初始化参数
    x0 = params['x0']
    maxit = params['maxit']
    lmbd = params['lambda']
    alpha = 1 / params['prox_Lips']
    y_k = x0
    t_k = 1
    restart_fista = params['restart_criterion']

    # 记录 PSNR 值
    psnr_values = []

    info = {'itertime': np.zeros(maxit), 'fx': np.zeros(maxit), 'iter': maxit}

    x_k = x0
    for k in range(maxit):
        tic = time.time()

        # 更新迭代
        prox_argument = y_k - alpha * gradf(y_k)
        x_k_next = proxg(prox_argument, alpha)
        t_k_next = (1 + np.sqrt(4 * (t_k ** 2) + 1)) / 2
        y_k_next = x_k_next + ((t_k - 1) / t_k_next) * (x_k_next - x_k)
        if restart_fista and gradient_scheme_restart_condition(x_k.reshape(x_k.shape[0],), x_k_next.reshape(x_k_next.shape[0],), y_k.reshape(y_k.shape[0],)):
            y_k = x_k
        else:
            y_k = y_k_next
            t_k = t_k_next
            x_k = x_k_next

        # 在 FISTA 函数中，打印中间结果
        #print(f"Iteration {k}: Min = {np.min(x_k)}, Max = {np.max(x_k)}, PSNR = {psnr_value}")

        # 计算 PSNR 值并记录
        reconstruction = x_k.reshape(params['m'], params['m'])
        reconstruction = np.clip(reconstruction, 0, 1)  # 确保值范围在 [0, 1]
        psnr_value = psnr(params['original_image'], reconstruction)
        psnr_values.append(psnr_value)

        # 计算误差并保存数据
        info['itertime'][k] = time.time() - tic
        info['fx'][k] = fx(x_k) + lmbd * gx(x_k)
        if k % params['iter_print'] == 0:
            if verbose:
                print_progress(k, maxit, info['fx'][k], fx(x_k), gx(x_k))

    if verbose:
        print_end_message(method_name, time.time() - tic_start)
    return x_k, info, psnr_values  # 返回 x_k、info 和 psnr_values


def gradient_scheme_restart_condition(x_k, x_k_next, y_k):
    """
    Whether to restart
    """
    return (y_k - x_k_next) @ (x_k_next - x_k) > 0

def reconstructL1(image, indices, optimizer, params):
    # Wavelet 算子
    r = RepresentationOperator(m=params["m"])

    # 定义整体算子
    forward_operator = lambda x: p_omega(r.WT(x), indices)  # P_Omega.W^T
    adjoint_operator = lambda x: r.W(p_omega_t(x, indices, params['m']))  # W. P_Omega^T

    # 生成测量值
    b = p_omega(image, indices)

    fx = lambda x: norm2sq(b - forward_operator(x))
    gx = lambda x: norm1(x)
    proxg = lambda x, y: l1_prox(x, params['lambda'] * y)
    gradf = lambda x: adjoint_operator(forward_operator(x) - b)

    # 调用优化算法
    x, info, psnr_values = optimizer(fx, gx, gradf, proxg, params, verbose=params['verbose'])
    return r.WT(x).reshape((params['m'], params['m'])), info, psnr_values


def reconstructTV(image, indices, optimizer, params):
    # 如果图像是彩色的，转换为灰度
    if len(image.shape) == 3:  # 彩色图像的形状为 (height, width, 3)
        image = rgb2gray(image)  # 转换为灰度

    # 定义整体算子
    forward_operator = lambda x: p_omega(x, indices)  # P_Omega
    adjoint_operator = lambda x: p_omega_t(x, indices, params['m'])  # P_Omega^T

    # 生成测量值
    b = forward_operator(image)

    # 定义目标函数和正则化项
    fx = lambda x: norm2sq(b - forward_operator(x))
    gx = lambda x: TV_norm(x, optimizer)
    proxg = lambda x, y: denoise_tv_chambolle(x.reshape((params['m'], params['m'])),
                                          weight=params["lambda"] * y, eps=1e-5,
                                          max_num_iter=50).reshape((params['N'], 1))
    gradf = lambda x: adjoint_operator(forward_operator(x) - b).reshape(x.shape[0], 1)

    # 调用优化算法
    x, info, psnr_values = optimizer(fx, gx, gradf, proxg, params, verbose=params['verbose'])
    return x.reshape((params['m'], params['m'])), info, psnr_values

# 在 main 函数中添加 ISTA 算法的调用
if __name__ == "__main__":
    # Load image and sample mask
    shape = (256, 256)
    params = {
        'maxit': 200,
        'tol': 10e-15,
        'prox_Lips': 1,
        'lambda': 0.01,
        'x0': np.zeros((shape[0] * shape[1], 1)),
        'restart_criterion': True,
        'stopping_criterion': False,
        'iter_print': 50,
        'shape': shape,
        'restart_param': 50,
        'verbose': True,
        'm': shape[0],
        'rate': 0.5,
        'N': shape[0] * shape[1]
    }
    PATH = 'D:/learn/机器智能/Algorithm-Project/p2 ISTA算法/Set14/ppt3.png'
    image = load_image(PATH, params['shape'])

    # 在调用 reconstructL1 和 reconstructTV 之前，添加原始图像到 params 字典
    params['original_image'] = image

    im_us, mask = apply_random_mask(image, params['rate'])
    indices = np.nonzero(mask.flatten())[0]
    params['indices'] = indices

    # 保存原始图像丢失像素后的图片
    io.imsave('noisy_image_with_missing_pixels.png', img_as_ubyte(im_us))

    # Parameter sweep over lambda
    lambdas = np.logspace(-4, 0, 10)
    psnr_l1_list_ista = []
    psnr_tv_list_ista = []
    psnr_l1_list_fista = []
    psnr_tv_list_fista = []

    # 记录 ISTA 和 FISTA 算法的总时间和调用次数
    ista_total_time = 0
    ista_call_count = 0
    fista_total_time = 0
    fista_call_count = 0

    for lambda_ in lambdas:
        params['lambda'] = lambda_

        # 记录 ISTA 算法的开始时间
        t_start = time.time()
        reconstruction_l1_ista = reconstructL1(image, indices, ISTA, params)[0]
        reconstruction_tv_ista = reconstructTV(image, indices, ISTA, params)[0]
        t_end = time.time()
        ista_total_time += t_end - t_start
        ista_call_count += 2  # 每次循环调用两次 ISTA（L1 和 TV）

        # 记录 FISTA 算法的开始时间
        t_start = time.time()
        reconstruction_l1_fista = reconstructL1(image, indices, FISTA, params)[0]
        reconstruction_tv_fista = reconstructTV(image, indices, FISTA, params)[0]
        t_end = time.time()
        fista_total_time += t_end - t_start
        fista_call_count += 2  # 每次循环调用两次 FISTA（L1 和 TV）

        # 计算 PSNR 和 SSIM
        psnr_l1_ista = psnr(image, reconstruction_l1_ista)
        ssim_l1_ista = ssim(image, reconstruction_l1_ista, data_range=1.0)
        psnr_l1_list_ista.append(psnr_l1_ista)

        psnr_tv_ista = psnr(image, reconstruction_tv_ista)
        ssim_tv_ista = ssim(image, reconstruction_tv_ista, data_range=1.0)
        psnr_tv_list_ista.append(psnr_tv_ista)

        psnr_l1_fista = psnr(image, reconstruction_l1_fista)
        ssim_l1_fista = ssim(image, reconstruction_l1_fista, data_range=1.0)
        psnr_l1_list_fista.append(psnr_l1_fista)

        psnr_tv_fista = psnr(image, reconstruction_tv_fista)
        ssim_tv_fista = ssim(image, reconstruction_tv_fista, data_range=1.0)
        psnr_tv_list_fista.append(psnr_tv_fista)

    # 计算 ISTA 和 FISTA 算法的平均时间
    ista_avg_time = ista_total_time / ista_call_count
    fista_avg_time = fista_total_time / fista_call_count

    # 输出 ISTA 和 FISTA 算法的总时间和平均时间
    print(f"ISTA 算法总时间: {ista_total_time:.2f} 秒")
    print(f"ISTA 算法平均时间: {ista_avg_time:.2f} 秒")
    print(f"FISTA 算法总时间: {fista_total_time:.2f} 秒")
    print(f"FISTA 算法平均时间: {fista_avg_time:.2f} 秒")

    # 找到最佳 lambda 值
    max_psnr_l1_ista = max(psnr_l1_list_ista)
    best_lambda_l1_ista = lambdas[psnr_l1_list_ista.index(max_psnr_l1_ista)]
    max_psnr_tv_ista = max(psnr_tv_list_ista)
    best_lambda_tv_ista = lambdas[psnr_tv_list_ista.index(max_psnr_tv_ista)]

    max_psnr_l1_fista = max(psnr_l1_list_fista)
    best_lambda_l1_fista = lambdas[psnr_l1_list_fista.index(max_psnr_l1_fista)]
    max_psnr_tv_fista = max(psnr_tv_list_fista)
    best_lambda_tv_fista = lambdas[psnr_tv_list_fista.index(max_psnr_tv_fista)]

    # 使用最佳 lambda 值进行重建
    params['lambda'] = best_lambda_l1_ista
    reconstruction_l1_ista, _, psnr_values_l1_ista = reconstructL1(image, indices, ISTA, params)

    params['lambda'] = best_lambda_tv_ista
    reconstruction_tv_ista, _, psnr_values_tv_ista = reconstructTV(image, indices, ISTA, params)

    params['lambda'] = best_lambda_l1_fista
    reconstruction_l1_fista, _, psnr_values_l1_fista = reconstructL1(image, indices, FISTA, params)

    params['lambda'] = best_lambda_tv_fista
    reconstruction_tv_fista, _, psnr_values_tv_fista = reconstructTV(image, indices, FISTA, params)

    # 绘制 PSNR 随迭代次数变化的曲线
    #plt.plot(range(params['maxit']), psnr_values_l1_ista, label=f'L1 ISTA (λ={best_lambda_l1_ista:.4f})')
    plt.plot(range(params['maxit']), psnr_values_tv_ista, label=f'TV ISTA (λ={best_lambda_tv_ista:.4f})')
    #plt.plot(range(params['maxit']), psnr_values_l1_fista, label=f'L1 FISTA (λ={best_lambda_l1_fista:.4f})')
    #plt.plot(range(params['maxit']), psnr_values_tv_fista, label=f'TV FISTA (λ={best_lambda_tv_fista:.4f})')
    plt.xlabel("Iteration")
    plt.ylabel("PSNR")
    plt.title("PSNR vs Iteration (ISTA)")
    plt.grid()
    plt.legend()
    plt.show()

    #plt.plot(range(params['maxit']), psnr_values_l1_ista, label=f'L1 ISTA (λ={best_lambda_l1_ista:.4f})')
    #plt.plot(range(params['maxit']), psnr_values_tv_ista, label=f'TV ISTA (λ={best_lambda_tv_ista:.4f})')
    #plt.plot(range(params['maxit']), psnr_values_l1_fista, label=f'L1 FISTA (λ={best_lambda_l1_fista:.4f})')
    plt.plot(range(params['maxit']), psnr_values_tv_fista, label=f'TV FISTA (λ={best_lambda_tv_fista:.4f})')
    plt.xlabel("Iteration")
    plt.ylabel("PSNR")
    plt.title("PSNR vs Iteration (FISTA)")
    plt.grid()
    plt.legend()
    plt.show()

    # 归一化图像数据到 [0, 1] 范围
    reconstruction_l1_ista_normalized = (reconstruction_l1_ista - np.min(reconstruction_l1_ista)) / (
                np.max(reconstruction_l1_ista) - np.min(reconstruction_l1_ista))
    reconstruction_tv_ista_normalized = (reconstruction_tv_ista - np.min(reconstruction_tv_ista)) / (
                np.max(reconstruction_tv_ista) - np.min(reconstruction_tv_ista))
    reconstruction_l1_fista_normalized = (reconstruction_l1_fista - np.min(reconstruction_l1_fista)) / (
                np.max(reconstruction_l1_fista) - np.min(reconstruction_l1_fista))
    reconstruction_tv_fista_normalized = (reconstruction_tv_fista - np.min(reconstruction_tv_fista)) / (
                np.max(reconstruction_tv_fista) - np.min(reconstruction_tv_fista))

    # 转换为 uint8
    reconstruction_l1_ista_uint8 = img_as_ubyte(reconstruction_l1_ista_normalized)
    reconstruction_tv_ista_uint8 = img_as_ubyte(reconstruction_tv_ista_normalized)
    reconstruction_l1_fista_uint8 = img_as_ubyte(reconstruction_l1_fista_normalized)
    reconstruction_tv_fista_uint8 = img_as_ubyte(reconstruction_tv_fista_normalized)

    # 保存图像
    io.imsave('reconstruction_l1_ista.png', reconstruction_l1_ista_uint8)
    io.imsave('reconstruction_tv_ista.png', reconstruction_tv_ista_uint8)
    io.imsave('reconstruction_l1_fista.png', reconstruction_l1_fista_uint8)
    io.imsave('reconstruction_tv_fista.png', reconstruction_tv_fista_uint8)

    print(f"L1 ISTA 正则化结果: lambda = {best_lambda_l1_ista}, PSNR = {max_psnr_l1_ista}, SSIM = {ssim_l1_ista}")
    print(f"TV ISTA 正则化结果: lambda = {best_lambda_tv_ista}, PSNR = {max_psnr_tv_ista}, SSIM = {ssim_tv_ista}")
    print(f"L1 FISTA 正则化结果: lambda = {best_lambda_l1_fista}, PSNR = {max_psnr_l1_fista}, SSIM = {ssim_l1_fista}")
    print(f"TV FISTA 正则化结果: lambda = {best_lambda_tv_fista}, PSNR = {max_psnr_tv_fista}, SSIM = {ssim_tv_fista}")

    # Plot results
    plt.semilogx(lambdas, psnr_l1_list_ista, label='L1-norm ISTA')
    plt.semilogx(lambdas, psnr_tv_list_ista, label='TV-norm ISTA')
    plt.xlabel("Lambda")
    plt.ylabel("PSNR")
    plt.title("PSNR vs Lambda (ISTA)")
    plt.grid()
    plt.legend()
    plt.show()

    plt.semilogx(lambdas, psnr_l1_list_fista, label='L1-norm FISTA')
    plt.semilogx(lambdas, psnr_tv_list_fista, label='TV-norm FISTA')
    plt.xlabel("Lambda")
    plt.ylabel("PSNR")
    plt.title("PSNR vs Lambda (FISTA)")
    plt.grid()
    plt.legend()
    plt.show()