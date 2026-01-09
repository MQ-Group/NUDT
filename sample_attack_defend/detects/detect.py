import sys
import os
import torch
import torch.nn as nn

sys.path.insert(0, '..')
import torchattacks

sys.path.insert(0, '..')
import robustbench
from robustbench.data import load_cifar10, load_cifar100
from robustbench.utils import load_model, clean_accuracy

from robustbench.model_zoo.enums import BenchmarkDataset, ThreatModel

from torchattacks import FGSM, PGD, BIM, CW, DeepFool, MIFGSM, VMIFGSM, GN, Jitter, Boundary, ZOO, HSJA, NES
from torchdefends import YOPO, TRADES, FREE, FAST
from torchdetects import SS, FS, LID

from torchvision import transforms

from utils.sse import sse_print


def detect(args):
    device = args.device
    
    try:
        # print(args.data_name)
        if args.data_name == 'cifar10':
            images, labels = load_cifar10(
                                        n_examples=args.selected_samples, # 抽取样本数量
                                        batch_size=args.batch,
                                        num_workers=args.workers,
                                        data_dir=args.data_path)
            dataset = BenchmarkDataset.cifar_10
        elif args.data_name == 'cifar100':
            images, labels = load_cifar100(
                                        n_examples=args.selected_samples, # 抽取样本数量
                                        batch_size=args.batch,
                                        num_workers=args.workers,
                                        data_dir=args.data_path)
            dataset = BenchmarkDataset.cifar_100
        else:
            raise ValueError('数据集不支持, 仅支持数据集: cifar10, cifar100.')
        
        event = "data_load"
        data = {
            "status": "success",
            "message": "数据集加载完毕.",
            "data name": args.data_name,
            "data path": args.data_path
        }
        sse_print(event, data)
    except Exception as e:
        event = "data_load"
        data = {
            "status": "failure",
            "message": f"{e}",
            "data name": args.data_name,
            "data path": args.data_path
        }
        sse_print(event, data)
        
    # print(images.shape)
    # print(labels.shape)
    
    
    '''
    :param model_name: The name used in the model zoo.
    :param model_dir: The base directory where the models are saved.
    :param dataset: The dataset on which the model is trained.
    :param threat_model: The threat model for which the model is trained.
    :param norm: Deprecated argument that can be used in place of ``threat_model``. If specified, it overrides ``threat_model``
    '''
    model = load_model(
        model_name = args.model_name, # Modas2021PRIMEResNet18 -- cifar100
        model_path = args.model_path,
        dataset = dataset,
        threat_model = ThreatModel.Linf, # 默认值
        norm='Linf'
        ).to(device)
    
    try:
        # print(args.attack_method)
        if args.attack_method == 'fgsm':
            atk = FGSM(model, eps=args.epsilon)
        elif args.attack_method == 'mifgsm':
            atk = MIFGSM(model, eps=args.epsilon, alpha=args.step_size, steps=args.max_iterations, decay=args.decay)
        elif args.attack_method == 'vmifgsm':
            atk = VMIFGSM(model, eps=args.epsilon, alpha=args.step_size, steps=args.max_iterations, decay=args.decay, N=args.sampled_examples, beta=3/2)
        elif args.attack_method == 'pgd':
            atk = PGD(model, eps=args.epsilon, alpha=args.step_size, steps=args.max_iterations, random_start=args.random_start)
        elif args.attack_method == 'bim':
            atk = BIM(model, eps=args.epsilon, alpha=args.step_size, steps=args.max_iterations)
        elif args.attack_method == 'cw':
            atk = CW(model, c=1, kappa=0, steps=args.max_iterations, lr=args.lr)
        elif args.attack_method == 'deepfool':
            atk = DeepFool(model, steps=args.max_iterations, overshoot=0.02)
        elif args.attack_method == 'gn':
            atk = GN(model, std=args.std)
        elif args.attack_method == 'jitter':
            atk = Jitter(model, eps=args.epsilon, alpha=args.step_size, steps=args.max_iterations, scale=args.scale, std=args.std, random_start=args.random_start)
        elif args.attack_method == 'boundary':
            atk = Boundary(model, max_queries=args.max_queries, init_epsilon=args.epsilon, spherical_step=0.01, orthogonal_step=args.step_size, binary_search_steps=args.binary_search_steps)
        elif args.attack_method == 'zoo':
            atk = ZOO(model, max_iterations=args.max_iterations, learning_rate=args.lr, binary_search_steps=args.binary_search_steps, init_const=0.01, beta=0.001, batch_size=128, resolution=1, early_stop_iters=10, abort_early=True)
        elif args.attack_method == 'hsja':
            atk = HSJA(model, max_queries=args.max_queries, norm=args.norm, gamma=0.01, init_num_evals=10, max_num_evals=10, stepsize_search='geometric_progression', num_iterations=args.max_iterations, constraint='L2', batch_size=128)
        elif args.attack_method == 'nes':
            atk = NES(model, max_queries=args.max_queries, epsilon=args.epsilon, learning_rate=args.lr, samples_per_draw=10, sigma=0.001, decay_factor=0.9, norm=args.norm, early_stop=True, loss_func='cross_entropy')
        elif args.attack_method == 'yopo':
            atk = YOPO(model, eps=args.epsilon)
        elif args.attack_method == 'pgdrs':
            atk = PGDRS(model, eps=args.epsilon, alpha=args.step_size, steps=args.max_iterations, noise_type=args.noise_type, noise_sd=args.noise_sd, noise_batch_size=5, batch_max=2048)
        elif args.attack_method == 'trades':
            atk = TRADES(model, eps=args.epsilon, alpha=args.step_size, steps=args.max_iterations)
        elif args.attack_method == 'free':
            atk = FREE(model, eps=args.epsilon, alpha=args.step_size, steps=args.max_iterations)
        elif args.attack_method == 'fast':
            atk = FAST(model, eps=args.epsilon)
        else:
            raise ValueError('不支持的攻击方法.')

        # print(args.detect_method)
        if args.detect_method == 'spatial_smoothing':
            detect = SS(model, kernel_size=args.kernel_size, detection_threshold=args.detection_threshold)
        elif args.detect_method == 'feature_squeezing':
            detect = FS(model, kernel_size=args.kernel_size, bit_depth=args.bit_depth, detection_threshold=args.detection_threshold)
        elif args.detect_method == 'local_intrinsic_dimensionality':
            detect = LID(model, k_nearest=args.k_nearest, detection_threshold=args.detection_threshold)
            if args.k_nearest > args.selected_samples:
                raise ValueError('args.k_nearest must less than args.selected_samples for local intrinsic dimensionality detect method.')
        else:
            raise ValueError('不支持的检测方法.')
        
        event = "samples_detection_init"
        data = {
            "status": "success",
            "message": "检测初始化完成.",
            "attack_method": args.attack_method,
            "detect_method": args.detect_method
        }
        sse_print(event, data)
    except Exception as e:
        event = "samples_detection_init"
        data = {
            "status": "failure",
            "message": f"{e}",
            "attack_method": args.attack_method,
            "detect_method": args.detect_method
        }
        sse_print(event, data)
        import sys
        sys.exit()
        
    # print(atk)
    
    adv_images = atk(images, labels)
    
    if args.detect_method == 'local_intrinsic_dimensionality':
        images_ref, adv_images_ref = images, adv_images
        
        if args.data_name == 'cifar10':
            images, labels = load_cifar10(
                                        n_examples=args.selected_samples, # 抽取样本数量
                                        batch_size=args.batch,
                                        num_workers=args.workers,
                                        data_dir=args.data_path)
        elif args.data_name == 'cifar100':
            images, labels = load_cifar100(
                                        n_examples=args.selected_samples, # 抽取样本数量
                                        batch_size=args.batch,
                                        num_workers=args.workers,
                                        data_dir=args.data_path)

        adv_images = atk(images, labels)
        
    ori_images_flod = f"{args.output_path}/ori_images"
    adv_images_flod = f"{args.output_path}/adv_images"
    os.makedirs(ori_images_flod, exist_ok=True)
    os.makedirs(adv_images_flod, exist_ok=True)
    
    total_iamges = images.shape[0]
    adv_detect_success_count = 0
    adv_detect_failure_count = 0
    ori_detect_success_count = 0
    ori_detect_failure_count = 0
    
    # print(images.device)
    # print(adv_images.device)
    # print(images_ref.device)
    # print(adv_images_ref.device)
    
    for i in range(total_iamges):
        if args.detect_method == 'local_intrinsic_dimensionality':
            images, images_ref = images.to(device), images_ref.to(device)
            is_adversarial, adv_pred_diff = detect.detect_adversarial_sample(adv_images[i:i+1], reference_clean=images_ref, reference_adv=adv_images_ref)
            is_original, ori_pred_diff = detect.detect_adversarial_sample(images[i:i+1], reference_clean=images_ref, reference_adv=adv_images_ref)
        else:
            images = images.to(device)
            is_adversarial, adv_pred_diff = detect.detect_adversarial_sample(adv_images[i:i+1])
            is_original, ori_pred_diff = detect.detect_adversarial_sample(images[i:i+1])
        
        # print(is_adversarial.shape) # torch.Size([batch])
        # print(adv_pred_diff.shape) # torch.Size([batch])
        # print(is_adversarial)
        # print(adv_pred_diff)
            
        if is_adversarial.item() is True:
            adv_detect_success_count += 1
            # adv_detect_result = '成功'
            adv_detect_result = 'success'
        else:
            adv_detect_failure_count += 1
            # adv_detect_result = '失败'
            adv_detect_result = 'failure'
        
        if is_original.item() is False:
            ori_detect_success_count += 1
            # ori_detect_result = '成功'
            ori_detect_result = 'success'
        else:
            ori_detect_failure_count += 1
            # ori_detect_result = '失败'
            ori_detect_result = 'failure'
        
        ori_img_save_path = f"{ori_images_flod}/ori_img_{i}.jpg"
        adv_img_save_path = f"{adv_images_flod}/adv_img_{i}.jpg"
        
        to_pil = transforms.ToPILImage()
        pil_image = to_pil(images[0])
        pil_image = to_pil(adv_images[0])
        pil_image.save(ori_img_save_path)
        pil_image.save(adv_img_save_path)
        
        event = "samples_detection"
        data = {
            "message": "正在执行样本检测...",
            "progress": int(i/total_iamges*100),
            # "log": f"[{int(i/total_iamges*100)}%] 正在执行样本检测... 对抗样本: {adv_img_save_path}, 检测结果: {adv_detect_result}, 差异值: {adv_pred_diff.item():.2f}, 原始样本: {ori_img_save_path}, 检测结果: {ori_detect_result}, 差异值: {ori_pred_diff.item():.2f}"
            "log": f"[{int(i/total_iamges*100)}%] 正在执行样本检测...",
            "details": {
                "adversarial_sample": {
                    "file_path": adv_img_save_path,
                    "detect_result": adv_detect_result,
                    "difference": adv_pred_diff.item()
                },
                "original_sample": {
                    "file_path": ori_img_save_path,
                    "detect_result": ori_detect_result,
                    "difference": ori_pred_diff.item()
                }
            }
        }
        sse_print(event, data)
    

    # 基础指标
    tp = adv_detect_success_count  # 真阳性（对抗样本检测正确）
    tn = ori_detect_success_count # 真阴性（自然样本检测正确）
    fp = ori_detect_failure_count # 假阳性（自然样本误判为对抗）
    fn = adv_detect_failure_count  # 假阴性（对抗样本漏判）
    
    # 准确率、精确率、召回率、F1
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0  # 检测率（TPR）
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # 误检率（FPR）

    event = "final_result"
    data = {
        "message": "样本检测执行完成.",
        "progress": 100,
        "log": f"[100%] 样本检测执行完成.",
        "details": {
            "attack_method": args.attack_method,
            "detect_method": args.detect_method,
            "original_samples": ori_images_flod,
            "adversarial_samples": adv_images_flod,
            "summary": {
                "total_original_samples": total_iamges,
                "total_adversarial_samples": total_iamges,
                "task_success_count": adv_detect_success_count + ori_detect_success_count,
                "task_failure_count": adv_detect_failure_count + ori_detect_failure_count,
                "true_positive (TP)": tp,
                "true_negative (TN)": tn,
                "false_positive (FP)": fp,
                "false_negative (FN)": fn,
                "accuracy": accuracy,
                "precision": precision,
                "recall (detection_rate)": recall,
                "f1_score": f1,
                "false_positive_rate (FPR)": fpr,
                "detection_threshold": args.detection_threshold
            }
        }
    }
    sse_print(event, data)
    
    
    


